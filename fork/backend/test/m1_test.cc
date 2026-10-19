// M1 check for the ChromeML llama.cpp backend, outside Chromium.
//
// Loads the backend the way services/on_device_model/ml/chrome_ml_holder.cc
// does (dlopen, then GetChromeMLAPIV2), hands it the model and adapter as open
// file descriptors the way session_accessor.cc does, and drives two sessions on
// one model: one with a LoRA adapter, one without. Questions alternate between
// the two sessions so both contexts are live at once.
//
// Usage: m1_test <library> <model.gguf> <adapter.gguf> <questions.tsv>
// questions.tsv: one "<label>\t<question>" per line.
// Writes one JSON line per answer to stdout, then "check" lines for clone and
// cancel behaviour.

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "third_party/skia/include/core/SkBitmap.h"
#include "services/on_device_model/ml/chrome_ml_api.h"

namespace {

// SmolLM2's chat template inserts this system turn when none is given, and the
// living-weights scores were measured with it.
constexpr char kSystemPrompt[] =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
constexpr uint32_t kMaxOutputTokens = 70;

std::string JsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

struct Harness {
  const ChromeMLAPI* api;
  ChromeMLModel model;

  ChromeMLSession NewSession(const char* adapter_path) {
    ChromeMLModelData data;
    ChromeMLAdaptationDescriptor descriptor{};
    descriptor.top_k = 1;
    descriptor.temperature = 0.0f;
    if (adapter_path) {
      data.weights_file = open(adapter_path, O_RDONLY);
      descriptor.model_data = &data;
    }
    return api->CreateSession(model, &descriptor);
  }

  void Append(ChromeMLSession session, const std::vector<ml::InputPiece>& in,
              ChromeMLCancel cancel = 0) {
    ChromeMLContextSavedFn saved = [](int) {};
    ChromeMLAppendOptions options{};
    options.input = in.data();
    options.input_size = in.size();
    options.context_saved_fn = &saved;
    options.input_source = InputSource::kUserInput;
    api->SessionAppend(session, &options, cancel);
  }

  std::string Generate(ChromeMLSession session, ChromeMLCancel cancel = 0,
                       int* tokens = nullptr) {
    std::string text;
    ChromeMLGenerateOutputFn fn = [&](const ChromeMLGenerateOutput* out) {
      if (out->text) {
        text += out->text;
      }
      if (out->status == ChromeMLGenerateStatus::kComplete && tokens) {
        *tokens = out->tokens_decoded;
      }
    };
    ChromeMLGenerateOptions options{};
    options.max_output_tokens = kMaxOutputTokens;
    options.output_fn = &fn;
    api->SessionGenerate(session, &options, cancel);
    return text;
  }

  // Asks one question in a fresh clone of `base`, which already holds the
  // system turn. Returns the answer, trimmed.
  std::string Ask(ChromeMLSession base, const std::string& question) {
    ChromeMLSession session = api->CloneSession(base);
    Append(session, {ml::Token::kUser, question, ml::Token::kEnd,
                     ml::Token::kModel});
    std::string answer = Generate(session);
    api->DestroySession(session);
    size_t a = answer.find_first_not_of(" \n");
    size_t b = answer.find_last_not_of(" \n");
    return a == std::string::npos ? "" : answer.substr(a, b - a + 1);
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: %s <library> <model.gguf> <adapter.gguf> <questions.tsv>\n",
                 argv[0]);
    return 2;
  }
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  auto get_api = reinterpret_cast<ChromeMLAPIGetterV2>(
      dlsym(lib, "GetChromeMLAPIV2"));
  if (!get_api) {
    std::fprintf(stderr, "GetChromeMLAPIV2 not exported\n");
    return 1;
  }
  const ChromeMLAPI* api = get_api(ChromeMLBackendMode::kLiteRtLmSession);
  DawnProcTable procs{};
  if (!api || !api->TryInitDawnProcs(procs)) {
    std::fprintf(stderr, "API table unavailable\n");
    return 1;
  }

  ChromeMLModelData model_data;
  model_data.weights_file = open(argv[2], O_RDONLY);
  ChromeMLModelDescriptor descriptor{};
  descriptor.backend_type = ml::ModelBackendType::kGpuBackend;
  descriptor.model_data = &model_data;
  descriptor.max_tokens = 1024;
  descriptor.top_k = 1;
  Harness h{api, api->SessionCreateModel(&descriptor, 0, nullptr)};
  if (!h.model) {
    std::fprintf(stderr, "SessionCreateModel failed\n");
    return 1;
  }

  ChromeMLSession adapted = h.NewSession(argv[3]);
  ChromeMLSession plain = h.NewSession(nullptr);
  if (!adapted || !plain) {
    std::fprintf(stderr, "CreateSession failed\n");
    return 1;
  }
  std::vector<ml::InputPiece> system = {ml::Token::kSystem, kSystemPrompt,
                                        ml::Token::kEnd};
  h.Append(adapted, system);
  h.Append(plain, system);

  std::ifstream questions(argv[4]);
  std::string line;
  while (std::getline(questions, line)) {
    size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string label = line.substr(0, tab);
    std::string question = line.substr(tab + 1);
    for (auto [arm, session] : {std::pair{"adapter", adapted},
                                std::pair{"plain", plain}}) {
      std::printf("{\"arm\":\"%s\",\"label\":\"%s\",\"q\":\"%s\",\"a\":\"%s\"}\n",
                  arm, JsonEscape(label).c_str(), JsonEscape(question).c_str(),
                  JsonEscape(h.Ask(session, question)).c_str());
      std::fflush(stdout);
    }
  }

  // A clone continues from the same state, so greedy answers must match.
  std::string probe = "What is the name of my pet?";
  ChromeMLSession first = api->CloneSession(adapted);
  h.Append(first, {ml::Token::kUser, probe, ml::Token::kEnd, ml::Token::kModel});
  ChromeMLSession second = api->CloneSession(first);
  std::string a = h.Generate(first);
  std::string b = h.Generate(second);
  std::printf("check clone_matches %s\n", a == b && !a.empty() ? "pass" : "fail");

  // A cancelled generation completes with no tokens.
  ChromeMLCancel cancel = api->CreateCancel();
  api->CancelExecuteModel(cancel);
  int tokens = -1;
  std::string cancelled = h.Generate(second, cancel, &tokens);
  std::printf("check cancel_stops %s\n",
              cancelled.empty() && tokens == 0 ? "pass" : "fail");
  api->DestroyCancel(cancel);

  for (ChromeMLSession s : {first, second, adapted, plain}) {
    api->DestroySession(s);
  }
  api->DestroyModel(h.model);
  return 0;
}
