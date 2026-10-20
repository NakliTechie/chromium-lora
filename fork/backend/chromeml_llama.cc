// ChromeML backend over llama.cpp.
//
// Chromium's on-device model service loads a shared library named
// optimization_guide_internal and calls it through the C function table in
// services/on_device_model/ml/chrome_ml_api.h. Google ships a closed build of
// that library. This file is an open one: it implements the table over
// llama.cpp's C API, so a Chromium build can run any GGUF model and attach a
// GGUF LoRA adapter per session.
//
// Calling convention, from session_accessor.cc at Chromium 155.0.8059.84:
// SessionAppend and SessionGenerate run on the session's own task runner and
// pass callbacks that live on the caller's stack, so both run to completion
// before returning. Cancellation arrives from another thread.
//
// The table passes std::function and std::string, so this file must be built
// with the same C++ standard library as its caller. Only llama.cpp's plain C
// API crosses into llama.cpp, which can therefore be built separately.

#include <fcntl.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <variant>
#include <vector>

#include "llama.h"
#include "third_party/skia/include/core/SkBitmap.h"
#include "services/on_device_model/ml/chrome_ml_api.h"

#define CHROMEML_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Log(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  std::fputs("[chromeml-llama] ", stderr);
  std::vfprintf(stderr, fmt, args);
  std::fputc('\n', stderr);
  va_end(args);
}

ChromeMLFatalErrorFn g_fatal_error_fn = nullptr;
const ChromeMLConstraintFns* g_constraint_fns = nullptr;

// The chat markup each ml::Token stands for, per template family.
struct TurnMarkup {
  const char* name;
  const char* system;
  const char* user;
  const char* model;
  const char* end;
};

constexpr TurnMarkup kChatMl = {"chatml", "<|im_start|>system\n",
                                "<|im_start|>user\n", "<|im_start|>assistant\n",
                                "<|im_end|>\n"};
// Gemma 4 has a system role and its own turn markers.
constexpr TurnMarkup kGemma4 = {"gemma4", "<|turn>system\n", "<|turn>user\n",
                                "<|turn>model\n", "<turn|>\n"};
// Gemma 2 and 3 have no system role; a system turn is sent as a user turn.
constexpr TurnMarkup kGemma = {"gemma", "<start_of_turn>user\n",
                               "<start_of_turn>user\n", "<start_of_turn>model\n",
                               "<end_of_turn>\n"};

const TurnMarkup& DetectMarkup(const llama_model* model) {
  const char* tmpl = llama_model_chat_template(model, nullptr);
  if (tmpl && std::strstr(tmpl, "<|turn>")) {
    return kGemma4;
  }
  if (tmpl && std::strstr(tmpl, "<start_of_turn>")) {
    return kGemma;
  }
  if (!tmpl || !std::strstr(tmpl, "<|im_start|>")) {
    Log("unrecognised chat template; using ChatML markup");
  }
  return kChatMl;
}

// Returns a path that opens the same file as an open descriptor. llama.cpp
// loads by path, and the sandboxed service process may not be allowed to open
// the file's real path, so this uses /dev/fd, which reopens the descriptor.
std::string PathForFile(PlatformFile file) {
  if (file == kInvalidPlatformFile) {
    return {};
  }
  return "/dev/fd/" + std::to_string(file);
}

// The model version adapters are pinned to: "sha256:" plus the first 16 hex
// digits of the weights file's SHA-256. fork/tools/tag_adapter.py computes the
// same value when it stamps an adapter. Reads with pread, because reopening
// a descriptor through /dev/fd shares its offset, and llama.cpp then reads
// the same descriptor from the start.
constexpr char kTargetVersionKey[] = "chromium_lora.target_model_version";

std::string ModelVersion(const ChromeMLModelData& data) {
  int fd = data.weights_file;
  bool owned = false;
  if (fd == kInvalidPlatformFile && data.model_path) {
    fd = open(data.model_path, O_RDONLY);
    owned = true;
  }
  if (fd < 0) {
    return {};
  }
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  std::vector<unsigned char> buf(1 << 22);
  off_t offset = 0;
  ssize_t n;
  while ((n = pread(fd, buf.data(), buf.size(), offset)) > 0) {
    CC_SHA256_Update(&ctx, buf.data(), static_cast<CC_LONG>(n));
    offset += n;
  }
  if (owned) {
    close(fd);
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
  std::string version = "sha256:";
  char hex[3];
  for (int i = 0; i < 8; ++i) {
    std::snprintf(hex, sizeof(hex), "%02x", digest[i]);
    version += hex;
  }
  return version;
}

// Closes a descriptor the library owns but does not use.
void CloseIfValid(PlatformFile file) {
  if (file != kInvalidPlatformFile) {
    close(file);
  }
}

std::string PathForModelData(const ChromeMLModelData& data) {
  if (data.model_path && *data.model_path) {
    return data.model_path;
  }
  return PathForFile(data.weights_file);
}

struct Model {
  llama_model* model = nullptr;
  const llama_vocab* vocab = nullptr;
  const TurnMarkup* markup = &kChatMl;
  uint32_t max_tokens = 0;
  float temperature = 0.0f;
  int top_k = 1;
  bool gpu = true;
  // See ModelVersion(). Adapters whose target version differs are refused.
  std::string version;
  // llama.cpp contexts on one model may decode concurrently in principle; this
  // backend serialises them, which is enough for a research build.
  std::mutex decode_mutex;
  // Loaded adapters, keyed by Chromium's file_id when given (each session gets
  // its own descriptor for the same adapter), else by path. Freed with the
  // model.
  std::map<std::string, llama_adapter_lora*> adapters;
};

struct Session {
  Model* model = nullptr;
  llama_context* ctx = nullptr;
  llama_adapter_lora* adapter = nullptr;
  uint32_t max_tokens = 0;
  float temperature = 0.0f;
  uint32_t top_k = 1;
  uint32_t n_past = 0;
  // The last decoded token, and whether the context still holds its logits.
  // A clone restores the attention cache but not the logits.
  llama_token last_token = 0;
  bool has_logits = false;
  // Set when the session's adapter was refused (for example, built for another
  // model). Chromium 155 does not check CreateSession's result, so a refused
  // session must still complete every call: appends add nothing and
  // generations fail with the only error status the interface has.
  bool refused = false;
};

struct Cancel {
  std::atomic<bool> cancelled{false};
};

bool IsCancelled(ChromeMLCancel cancel) {
  return cancel && reinterpret_cast<Cancel*>(cancel)->cancelled.load();
}

llama_context* NewContext(Model* model, uint32_t n_ctx) {
  llama_context_params params = llama_context_default_params();
  params.n_ctx = n_ctx;
  params.n_batch = std::min<uint32_t>(n_ctx, 512);
  params.no_perf = true;
  return llama_init_from_model(model->model, params);
}

bool AttachAdapter(Session* session) {
  if (!session->adapter) {
    return true;
  }
  float scale = 1.0f;
  return llama_set_adapters_lora(session->ctx, &session->adapter, 1, &scale) == 0;
}

// Renders input pieces into the model's chat markup. Images, audio and tool
// pieces are not supported by this backend and are skipped.
std::string Render(const TurnMarkup& markup, const ml::InputPiece* input,
                   size_t input_size) {
  std::string text;
  for (size_t i = 0; i < input_size; ++i) {
    const ml::InputPiece& piece = input[i];
    if (const auto* token = std::get_if<ml::Token>(&piece)) {
      switch (*token) {
        case ml::Token::kSystem:
          text += markup.system;
          break;
        case ml::Token::kUser:
          text += markup.user;
          break;
        case ml::Token::kModel:
          text += markup.model;
          break;
        case ml::Token::kEnd:
          text += markup.end;
          break;
        case ml::Token::kToolCall:
        case ml::Token::kToolResponse:
          break;
      }
    } else if (const auto* str = std::get_if<std::string>(&piece)) {
      text += *str;
    }
  }
  return text;
}

std::vector<llama_token> Tokenize(const llama_vocab* vocab,
                                  const std::string& text, bool add_special) {
  int32_t n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0,
                              add_special, /*parse_special=*/true);
  std::vector<llama_token> tokens(std::max(n, 0));
  if (n > 0) {
    llama_tokenize(vocab, text.data(), text.size(), tokens.data(), n,
                   add_special, /*parse_special=*/true);
  }
  return tokens;
}

// Length of the longest prefix of `s` that ends on a UTF-8 boundary.
size_t CompleteUtf8Prefix(const std::string& s) {
  size_t i = s.size();
  size_t back = 0;
  while (i > 0 && back < 4) {
    unsigned char c = s[i - 1];
    if ((c & 0xC0) != 0x80) {
      size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
      return back + 1 >= need ? s.size() : i - 1;
    }
    --i;
    ++back;
  }
  return s.size();
}

// ---- ChromeMLAPI functions ----

void InitDawnProcs(const DawnProcTable&) {}
bool TryInitDawnProcs(const DawnProcTable&) {
  return true;
}
void SetMetricsFns(const ChromeMLMetricsFns*) {}
void SetFatalErrorFn(ChromeMLFatalErrorFn fn) {
  g_fatal_error_fn = fn;
}
void SetFatalErrorNonGpuFn(ChromeMLFatalErrorFn fn) {
  g_fatal_error_fn = fn;
}
void SetConstraintFns(const ChromeMLConstraintFns* fns) {
  g_constraint_fns = fns;
}

// Placeholder figures, not measurements. Chromium uses these to pick a
// performance class; M2 replaces them with a measured run.
bool GetEstimatedPerformance(ChromeMLPerformanceInfo* info) {
  info->input_speed = 1000.0f;
  info->output_speed = 50.0f;
  info->is_integrated_gpu = true;
  info->device_heap_size = 8ull << 30;
  info->max_buffer_size = 2ull << 30;
  return true;
}

// llama.cpp uses Metal directly, not Dawn, so there is no Dawn adapter to
// report. Returning true without calling back tells Chromium's GPU blocklist
// check (services/on_device_model/ml/gpu_blocklist.cc) that the GPU is usable;
// returning false would mark it blocked.
bool QueryGPUAdapter(void (*)(WGPUAdapter, void*), void*) {
  return true;
}

bool GetCapabilities(PlatformFile, ChromeMLCapabilities& capabilities) {
  capabilities.image_input = false;
  capabilities.audio_input = false;
  return true;
}

ChromeMLModel SessionCreateModel(const ChromeMLModelDescriptor* descriptor,
                                 uintptr_t /*context*/,
                                 ChromeMLScheduleFn /*schedule*/) {
  if (!descriptor || !descriptor->model_data) {
    return 0;
  }
  const ChromeMLModelData& data = *descriptor->model_data;
  std::string path = PathForModelData(data);
  if (path.empty()) {
    Log("no model path or readable weights file");
    return 0;
  }

  static std::once_flag backend_once;
  std::call_once(backend_once, [] {
    if (!std::getenv("CHROMEML_LLAMA_VERBOSE")) {
      llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) {
          std::fputs(text, stderr);
        }
      }, nullptr);
    }
    llama_backend_init();
  });

  auto model = std::make_unique<Model>();
  model->gpu = descriptor->backend_type != ml::ModelBackendType::kCpuBackend;
  llama_model_params params = llama_model_default_params();
  params.n_gpu_layers = model->gpu ? -1 : 0;
  model->version = ModelVersion(data);
  model->model = llama_model_load_from_file(path.c_str(), params);
  // The library owns these files. Only the weights are used, and they are
  // closed once loading is done.
  CloseIfValid(data.weights_file);
  CloseIfValid(data.cache_file);
  CloseIfValid(data.encoder_cache_file);
  CloseIfValid(data.adapter_cache_file);
  CloseIfValid(data.program_cache_file);
  if (!model->model) {
    Log("failed to load model from %s", path.c_str());
    return 0;
  }
  model->vocab = llama_model_get_vocab(model->model);
  model->markup = &DetectMarkup(model->model);
  int32_t trained = llama_model_n_ctx_train(model->model);
  model->max_tokens = descriptor->max_tokens ? descriptor->max_tokens
                                             : static_cast<uint32_t>(trained);
  model->max_tokens = std::min<uint32_t>(model->max_tokens, trained);
  model->temperature = descriptor->temperature;
  model->top_k = descriptor->top_k;
  Log("model loaded: %s, version %s, markup %s, max_tokens %u, %s",
      path.c_str(), model->version.c_str(), model->markup->name,
      model->max_tokens, model->gpu ? "gpu" : "cpu");
  return reinterpret_cast<ChromeMLModel>(model.release());
}

void DestroyModel(ChromeMLModel handle) {
  auto* model = reinterpret_cast<Model*>(handle);
  if (!model) {
    return;
  }
  for (auto& [path, adapter] : model->adapters) {
    llama_adapter_lora_free(adapter);
  }
  llama_model_free(model->model);
  delete model;
}

ChromeMLSession CreateSession(ChromeMLModel model_handle,
                              const ChromeMLAdaptationDescriptor* descriptor) {
  auto* model = reinterpret_cast<Model*>(model_handle);
  if (!model) {
    return 0;
  }
  auto session = std::make_unique<Session>();
  session->model = model;
  session->max_tokens = model->max_tokens;
  session->temperature = model->temperature;
  session->top_k = model->top_k;
  if (descriptor) {
    if (descriptor->max_tokens) {
      session->max_tokens = std::min(descriptor->max_tokens, model->max_tokens);
    }
    session->temperature = descriptor->temperature;
    session->top_k = descriptor->top_k;
    if (descriptor->model_data) {
      const ChromeMLModelData& data = *descriptor->model_data;
      std::string path = PathForModelData(data);
      if (path.empty()) {
        Log("adaptation has no readable file");
        return 0;
      }
      std::string key = data.file_id ? "id:" + std::to_string(*data.file_id)
                                     : path;
      {
        std::lock_guard<std::mutex> lock(model->decode_mutex);
        auto it = model->adapters.find(key);
        if (it == model->adapters.end()) {
          llama_adapter_lora* adapter =
              llama_adapter_lora_init(model->model, path.c_str());
          char target[128] = {};
          if (adapter && llama_adapter_meta_val_str(adapter, kTargetVersionKey,
                                                    target, sizeof(target)) < 0) {
            target[0] = '\0';
          }
          if (adapter && model->version != target) {
            // Pinning: an adapter built for another model is refused, not
            // applied to weights it was not trained against.
            Log("adapter %s refused: built for model %s, loaded model is %s",
                key.c_str(), *target ? target : "(untagged)",
                model->version.c_str());
            llama_adapter_lora_free(adapter);
            adapter = nullptr;
          }
          if (adapter) {
            it = model->adapters.emplace(key, adapter).first;
            Log("adapter loaded: %s, target %s", key.c_str(), target);
          }
        }
        session->adapter = it != model->adapters.end() ? it->second : nullptr;
      }
      // Close only after loading: the path may be /dev/fd for this descriptor.
      CloseIfValid(data.weights_file);
      if (!session->adapter) {
        Log("session refused: adapter %s not loaded", key.c_str());
        session->refused = true;
        return reinterpret_cast<ChromeMLSession>(session.release());
      }
    }
  }
  session->ctx = NewContext(model, session->max_tokens);
  if (!session->ctx || !AttachAdapter(session.get())) {
    Log("failed to create a context");
    if (session->ctx) {
      llama_free(session->ctx);
    }
    return 0;
  }
  return reinterpret_cast<ChromeMLSession>(session.release());
}

ChromeMLSession CloneSession(ChromeMLSession handle) {
  auto* source = reinterpret_cast<Session*>(handle);
  if (!source) {
    return 0;
  }
  auto clone = std::make_unique<Session>(*source);
  if (source->refused) {
    clone->ctx = nullptr;
    return reinterpret_cast<ChromeMLSession>(clone.release());
  }
  clone->ctx = NewContext(source->model, source->max_tokens);
  if (!clone->ctx || !AttachAdapter(clone.get())) {
    if (clone->ctx) {
      llama_free(clone->ctx);
    }
    return 0;
  }
  std::lock_guard<std::mutex> lock(source->model->decode_mutex);
  std::vector<uint8_t> state(llama_state_get_size(source->ctx));
  llama_state_get_data(source->ctx, state.data(), state.size());
  llama_state_set_data(clone->ctx, state.data(), state.size());
  clone->has_logits = false;
  return reinterpret_cast<ChromeMLSession>(clone.release());
}

void DestroySession(ChromeMLSession handle) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (!session) {
    return;
  }
  if (session->ctx) {
    llama_free(session->ctx);
  }
  delete session;
}

// Decodes `tokens` into the session in batches. Returns the number decoded.
uint32_t DecodeTokens(Session* session, const llama_token* tokens, size_t count,
                      ChromeMLCancel cancel) {
  uint32_t n_batch = llama_n_batch(session->ctx);
  uint32_t done = 0;
  std::lock_guard<std::mutex> lock(session->model->decode_mutex);
  while (done < count && !IsCancelled(cancel)) {
    uint32_t n = std::min<uint32_t>(n_batch, count - done);
    llama_batch batch =
        llama_batch_get_one(const_cast<llama_token*>(tokens + done), n);
    if (llama_decode(session->ctx, batch) != 0) {
      Log("decode failed after %u tokens", session->n_past);
      break;
    }
    done += n;
    session->n_past += n;
    session->last_token = tokens[done - 1];
    session->has_logits = true;
  }
  return done;
}

bool SessionAppend(ChromeMLSession handle, const ChromeMLAppendOptions* options,
                   ChromeMLCancel cancel) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (!session || !options) {
    return false;
  }
  if (session->refused) {
    if (options->context_saved_fn && *options->context_saved_fn) {
      (*options->context_saved_fn)(0);
    }
    return true;
  }
  std::string text =
      Render(*session->model->markup, options->input, options->input_size);
  std::vector<llama_token> tokens =
      Tokenize(session->model->vocab, text, /*add_special=*/session->n_past == 0);
  size_t room = session->max_tokens > session->n_past
                    ? session->max_tokens - session->n_past
                    : 0;
  size_t limit = options->max_tokens ? std::min<size_t>(options->max_tokens, room)
                                     : room;
  if (tokens.size() > limit) {
    tokens.resize(limit);
  }
  uint32_t added = DecodeTokens(session, tokens.data(), tokens.size(), cancel);
  if (options->context_saved_fn && *options->context_saved_fn) {
    (*options->context_saved_fn)(static_cast<int>(added));
  }
  return true;
}

bool SessionGenerate(ChromeMLSession handle,
                     const ChromeMLGenerateOptions* options,
                     ChromeMLCancel cancel) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (!session || !options || !options->output_fn) {
    return false;
  }
  // Output constraints (structured output) are not implemented; the
  // constraint is released so it does not leak.
  if (options->constraint && g_constraint_fns && g_constraint_fns->Delete) {
    g_constraint_fns->Delete(options->constraint);
  }
  const ChromeMLGenerateOutputFn& output_fn = *options->output_fn;
  if (session->refused) {
    ChromeMLGenerateOutput refused{ChromeMLGenerateStatus::kInvalidConstraint,
                                   nullptr};
    output_fn(&refused);
    return true;
  }

  llama_sampler* sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (session->top_k <= 1 || session->temperature <= 0.0f) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(session->top_k));
    llama_sampler_chain_add(sampler,
                            llama_sampler_init_temp(session->temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
  }

  // Sampling reads the logits of the last decoded token. When the context has
  // none (a fresh clone), drop that token from the cache and decode it again.
  if (!session->has_logits && session->n_past > 0) {
    {
      std::lock_guard<std::mutex> lock(session->model->decode_mutex);
      llama_memory_seq_rm(llama_get_memory(session->ctx), 0,
                          session->n_past - 1, -1);
    }
    --session->n_past;
    llama_token last = session->last_token;
    DecodeTokens(session, &last, 1, 0);
  }

  int decoded = 0;
  std::string pending;
  while (session->has_logits && !IsCancelled(cancel)) {
    if (options->max_output_tokens &&
        decoded >= static_cast<int>(options->max_output_tokens)) {
      break;
    }
    if (session->n_past + 1 > session->max_tokens) {
      break;
    }
    llama_token token;
    {
      std::lock_guard<std::mutex> lock(session->model->decode_mutex);
      token = llama_sampler_sample(sampler, session->ctx, -1);
    }
    if (llama_vocab_is_eog(session->model->vocab, token)) {
      break;
    }
    char piece[256];
    int32_t n = llama_token_to_piece(session->model->vocab, token, piece,
                                     sizeof(piece), 0, /*special=*/false);
    if (n > 0) {
      pending.append(piece, n);
    }
    size_t ready = CompleteUtf8Prefix(pending);
    if (ready > 0) {
      std::string chunk = pending.substr(0, ready);
      pending.erase(0, ready);
      ChromeMLGenerateOutput output{ChromeMLGenerateStatus::kInProgress,
                                    chunk.c_str()};
      output_fn(&output);
    }
    if (DecodeTokens(session, &token, 1, 0) != 1) {
      break;
    }
    ++decoded;
  }
  llama_sampler_free(sampler);
  if (!pending.empty()) {
    ChromeMLGenerateOutput output{ChromeMLGenerateStatus::kInProgress,
                                  pending.c_str()};
    output_fn(&output);
  }
  ChromeMLGenerateOutput done{ChromeMLGenerateStatus::kComplete, nullptr};
  done.tokens_decoded = decoded;
  output_fn(&done);
  return true;
}

// Deprecated in Chromium in favour of Append and Generate; not implemented.
bool SessionExecuteModel(ChromeMLSession, ChromeMLModel,
                         const ChromeMLExecuteOptions*, ChromeMLCancel) {
  return false;
}

void SessionSizeInTokens(ChromeMLSession handle, const std::string& text,
                         const ChromeMLSizeInTokensFn& fn) {
  auto* session = reinterpret_cast<Session*>(handle);
  fn(session ? static_cast<int>(
                   Tokenize(session->model->vocab, text, false).size())
             : 0);
}

void SessionSizeInTokensInputPiece(ChromeMLSession handle, ChromeMLModel,
                                   const ml::InputPiece* input,
                                   size_t input_size,
                                   const ChromeMLSizeInTokensFn& fn) {
  auto* session = reinterpret_cast<Session*>(handle);
  if (!session) {
    fn(0);
    return;
  }
  std::string text = Render(*session->model->markup, input, input_size);
  fn(static_cast<int>(Tokenize(session->model->vocab, text, false).size()));
}

// Scoring is not implemented yet; it reports zero.
void SessionScore(ChromeMLSession, const std::string&,
                  const ChromeMLScoreFn& fn) {
  fn(0.0f);
}

void SessionGetProbabilitiesBlocking(ChromeMLSession, const std::string&,
                                     const ChromeMLGetProbabilitiesBlockingFn& fn) {
  fn({});
}

ChromeMLCancel CreateCancel() {
  return reinterpret_cast<ChromeMLCancel>(new Cancel());
}

void DestroyCancel(ChromeMLCancel cancel) {
  delete reinterpret_cast<Cancel*>(cancel);
}

void CancelExecuteModel(ChromeMLCancel cancel) {
  if (cancel) {
    reinterpret_cast<Cancel*>(cancel)->cancelled.store(true);
  }
}

// Constraint support needs tokenizer parameters; not implemented yet.
bool GetTokenizerParams(ChromeMLModel, ChromeMLSession,
                        const ChromeMLGetTokenizerParamsFn&, bool) {
  return false;
}
bool GetTokenizerParamsV2(ChromeMLModel, ChromeMLSession,
                          const ChromeMLGetTokenizerParamsFn&) {
  return false;
}
bool GetTokenizerParamsV3(ChromeMLModel, ChromeMLSession,
                          const ChromeMLGetTokenizerParamsV3Fn&) {
  return false;
}

TfLiteDelegate* CreateGpuDelegate() {
  return nullptr;
}
TfLiteDelegate* CreateGpuDelegateWithPrecision(GpuDelegatePrecision) {
  return nullptr;
}
void DestroyGpuDelegate(TfLiteDelegate*) {}

const ChromeMLAPI kApi = {
    .InitDawnProcs = &InitDawnProcs,
    .TryInitDawnProcs = &TryInitDawnProcs,
    .SetMetricsFns = &SetMetricsFns,
    .SetFatalErrorFn = &SetFatalErrorFn,
    .DestroyModel = &DestroyModel,
    .GetEstimatedPerformance = &GetEstimatedPerformance,
    .QueryGPUAdapter = &QueryGPUAdapter,
    .GetCapabilities = &GetCapabilities,
    .SetFatalErrorNonGpuFn = &SetFatalErrorNonGpuFn,
    .SessionCreateModel = &SessionCreateModel,
    .SessionAppend = &SessionAppend,
    .SessionGenerate = &SessionGenerate,
    .SessionExecuteModel = &SessionExecuteModel,
    .SessionSizeInTokens = &SessionSizeInTokens,
    .SessionSizeInTokensInputPiece = &SessionSizeInTokensInputPiece,
    .SessionScore = &SessionScore,
    .SessionGetProbabilitiesBlocking = &SessionGetProbabilitiesBlocking,
    .CreateSession = &CreateSession,
    .CloneSession = &CloneSession,
    .DestroySession = &DestroySession,
    .CreateCancel = &CreateCancel,
    .DestroyCancel = &DestroyCancel,
    .CancelExecuteModel = &CancelExecuteModel,
    .SetConstraintFns = &SetConstraintFns,
    .GetTokenizerParams = &GetTokenizerParams,
    .GetTokenizerParamsV2 = &GetTokenizerParamsV2,
    .GetTokenizerParamsV3 = &GetTokenizerParamsV3,
    .CreateGpuDelegate = &CreateGpuDelegate,
    .CreateGpuDelegateWithPrecision = &CreateGpuDelegateWithPrecision,
    .DestroyGpuDelegate = &DestroyGpuDelegate,
    .asr_api = {},
};

}  // namespace

CHROMEML_EXPORT const ChromeMLAPI* GetChromeMLAPIV2(ChromeMLBackendMode mode) {
  if (mode == ChromeMLBackendMode::kLiteRtLmConversation) {
    Log("conversation backend mode requested; serving the session API");
  }
  return &kApi;
}

CHROMEML_EXPORT const ChromeMLAPI* GetChromeMLAPI(bool) {
  return &kApi;
}
