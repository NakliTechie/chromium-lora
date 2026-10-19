# LiteRT-LM is public, and the public build has the Artisan backend

**Date:** 18 Oct 2026, 17:20–20:05 IST
**Supersedes the conclusion in** [`261017-chrome-lora-needs-artisan-model.md`](261017-chrome-lora-needs-artisan-model.md)
§1–2, which said the adapter route was blocked. It was blocked by not knowing the engine ships
publicly.

**Refined later the same evening** — §6 records the experiment, which moved the blocker, and then
**§9 reverses §6's conclusion**: an artisan-format model *is* public. Read §9 before acting on
§5 or §6.

**Every claim carries one tag:** **[wheel]** read in the installed `litert-lm` 0.18.0 package ·
**[dylib]** `strings` on Chrome's `liboptimization_guide_internal.dylib` @ 155.0.8059.84 ·
**[run]** I ran it on this Mac today · **[inferred]** reasoned from the two.

## What this means for Chirag

1. **The demo needs no Chromium patch and no closed binary.** `litert-lm` on PyPI ships the same
   inference stack Chrome's closed dylib is built from — same class names, same
   `supported_lora_ranks` check, same backend enum. We assemble our own `.litertlm` and point Chrome
   at it with the manifest override the 17 Oct run already proved works [wheel], [dylib].
2. **Per-site switching is already a supported shape upstream, not a feature we would ask for.**
   LoRA is attached per *session* (`lora_path`), and adapters are keyed by `lora_id`. Two sites =
   two sessions on one shared engine [wheel], [dylib].
3. **A working per-site LoRA prototype now runs on this Mac** [run, §9]. An artisan-format model is
   public — `litert-community/gemma-4-E2B-it-litert-lm` ships a `-gpu` variant with a real
   `tf_lite_artisan_text_decoder` section — the engine selects `GPU_ARTISAN`, **honours**
   `supported_lora_ranks: [16]`, answers `Text(text='Paris')`, and two conversations on one loaded
   engine each reach `LlmGpuArtisanExecutor::LoadLoRA()`. The only unproven link is the adapter's
   bytes: we can make the loader fail and know it wants a ≥1 MiB, 16384-aligned memory-mapped
   flatbuffer, but not its schema. §6 got this backwards for an hour because the search was four
   spot checks; §9 reverses it.

## 1. The engine is the same code

Chrome's engine lives in a closed dylib whose source paths are all under
`components/optimization_guide/internal/third_party/odml/...` — a Google-internal directory, absent
from the public mirror. It is a build of **LiteRT-LM / ODML, Apache 2.0**, published as the
`litert-lm` PyPI wheel [wheel], [dylib].

Symbol-for-symbol match between the wheel's `liblitert-lm.dylib` and Chrome's dylib:

| Symbol | Chrome dylib | Public wheel |
|---|---|---|
| `GpuArtisanConfig` | yes | yes [wheel] |
| `LlmGpuArtisanExecutor` | yes | yes [wheel] |
| `litert::lm::LoRA`, `FileLoraData`, `FlatBufferLoraData` | — | yes [wheel] |
| `supported_lora_ranks`, `lora_ranks`, `lora_rank` | yes | yes [wheel] |
| `CPU_ARTISAN`, `GPU_ARTISAN`, `GOOGLE_TENSOR_ARTISAN` | yes | yes [wheel] |
| `SessionConfig.LoraParams.lora_path` | yes | yes [wheel] |
| `. Supported backends are: [CPU, GPU, NPU, GPU_ARTISAN, CPU_ARTISAN, GOOGLE_TENSOR_ARTISAN, TENSOR_LLM]` | yes | yes, byte-identical [wheel] |

The wheel also carries the one string that names the switch [wheel]:

```
kTfLitePrefillDecode not found, trying kArtisanTextDecoder
```

The 17 Oct note read Chrome's `Artisan model detected. Switching backend from GPU to GPU_ARTISAN.`
as "the backend is auto-detected, never requested". That holds — but the *input* to the
auto-detection is a model-file section type, and the public builder lets us choose it.

## 2. The public builder can label a section as the Artisan decoder

`litert_lm_builder/litertlm_builder.py:233` [wheel]:

```python
ARTISAN_TEXT_DECODER = "tf_lite_artisan_text_decoder"
```

and `:263` [wheel]:

```python
class Backend(str, enum.Enum):
  CPU = "cpu"; GPU = "gpu"; NPU = "npu"; GPU_ARTISAN = "gpu_artisan"
```

`is_llm_model` (`:399`) treats `ARTISAN_TEXT_DECODER` and `PREFILL_DECODE` as equivalent LLM
sections. `set_backend_constraint(model_type, "gpu_artisan")` (`:1216`) writes a per-section
`backend_constraint` metadata key [wheel].

The CLI takes `--model_type` from that same enum and `--backend_constraint` from that enum, so the
relabel is a command-line flag, not a hand edit [wheel]:

```
litertlm_builder tflite_model --path prefill_decode.tflite \
    --model_type artisan_text_decoder --backend_constraint gpu_artisan
```

The builder is an **assembler**, not a converter: it packages `.tflite` sections, a weights file and
a tokenizer into a `.litertlm`. There is also an `unpack` subcommand, so an existing Chrome model can
be taken apart and put back together [wheel].

**This is the load-bearing consequence:** we do not need an HF→tflite converter. We can take a model
Chrome already accepts, unpack it, relabel the decoder section, repack, and serve that.

## 3. LoRA is attached per session, keyed by id

Public API, `litert_lm/_ffi.py` [wheel]:

| FFI symbol | Level | Meaning |
|---|---|---|
| `litert_lm_engine_settings_set_lora_rank` | engine | rank of the text LoRA; 0 disables |
| `litert_lm_engine_settings_set_supported_lora_ranks` | engine | ranks this base accepts |
| `litert_lm_session_config_set_lora_path` | **session** | which adapter this session uses |
| `litert_lm_session_config_set_audio_lora_path` | session | audio equivalent |
| `litert_lm_loaded_file_modality_supported_backends` | file | what the loaded model can run on |

Python wrappers: `interfaces.LoraConfig(lora_path=..., audio_lora_path=...)`,
`interfaces.LoraRankConfig(lora_rank=...)`, and `conversation.py:53` threads `lora_config` into
session creation [wheel].

`LiteRtLmBackendType` is only `UNSPECIFIED | CPU | GPU | NPU` — no Artisan member [wheel]. That is
the same conclusion as from the dylib: **Artisan is chosen by the model, never requested.**

Adapter identity upstream [dylib]: `new_data.lora_id == data_.lora_id`, `scoped_lora_with_id:`,
`scoped_lora:` — a registry of adapters against one loaded executor. A debug option `fake_lora_path`
also exists, which is a cheap way to prove the plumbing before a real adapter exists [dylib].

## 4. The tensor names a real adapter must use

Chrome's dylib publishes the LoRA loader's acceptance regex verbatim [dylib]:

```
^(?:(?:query|key|value|post)_w_prime_(?:left|right)
  |lora_atten_(?:q|k|v|o)_(?:a|b)_prime_weight)_\d+$
|^transformer\.layer_\d+\.attn\.(?:q|k|v|attn_vec_einsum)\.w_prime_(?:left|right)$
```

Also present: `Metadata must have lora_rank.`, `lora rank must be positive.`,
`(supported_lora_rank)>(0)`, and `lora_atten_q_a_prime_weight_0` as a literal default name [dylib].

This replaces the shape probe the 16 Oct session designed and the 17 Oct session skipped — the
engine tells us the names in its own binary, so we can build an adapter to match instead of guessing.

## 5. The prototype, and the two things that could still fail

Recipe:

1. Obtain a `.litertlm` Chrome accepts. Re-download Gemma 4 (`gemma4-2b-it`, 2.48 GB) through the
   manifest override, or build one from a small open base if a public HF→tflite path exists.
2. `litertlm unpack` it.
3. Repack with the text decoder section relabelled `artisan_text_decoder`,
   `backend_constraint=gpu_artisan`.
4. Serve it through `--optimization-guide-manifest-override` with an `AdaptationRecipe` — the path
   already proven to route an adapter [17 Oct note].
5. Watch for `Artisan model detected. Switching backend from GPU to GPU_ARTISAN.` and
   `LlmGpuArtisanExecutor::LoadLoRA id:` in the engine log. Those two lines are the whole claim.
6. Two adapters, two origins, one engine, two sessions — the per-site part.

Risks, as written before the experiment. **R1 is now settled — see §6; it was real and worse than
guessed.** R2 is untouched.

- **R1 — RESOLVED, and confirmed.** The relabelled decoder *is* rejected, and not by tensor
  validation: the Artisan loader rejects the TFLite container itself
  (`The model is not a valid Flatbuffer buffer`). Steps 3–5 cannot be completed with a stock
  TFLite. §6 has the three builds and the exact error.
- **R2 — still open.** `FileLoraData` vs `FlatBufferLoraData` are two paths; we have the
  tensor-name regex but not the container spec. Cheapest first probe is `fake_lora_path` to confirm
  the engine reaches the loader at all [dylib].

Neither risk needed the Chromium build. The build matters only for the developer-surface patch, which
is the *ask*, not the *proof*.

## 6. The experiment: the switch is real, the payload is not stock TFLite

18 Oct 19:45–20:05 IST, on this Mac, with the public wheel only. Base model
`gynsolomon/test-litertlm` → `todolist-functiongemma_q8_ekv1024.litertlm`,
284,393,472 bytes, functiongemma 270M, q8 — a small generative model, unpacked with
`litert-lm-builder unpack` into one `SP_Tokenizer` and one 279 MB
`TFLiteModel` section labelled `tf_lite_prefill_decode`.

Three builds from that one section, same runtime, `lora_rank = 8`:

| Build | Engine path | Outcome |
|---|---|---|
| stock, untouched | `LITERT_WEBGPU`, 949/949 nodes delegated | loads and answers; logs `supported_lora_ranks is only supported for GpuArtisanConfig. The provided lora ranks will be ignored.` |
| relabelled `artisan_text_decoder` + `backend_constraint gpu_artisan` | **`LlmGpuArtisanExecutor::Create()`** | `Build ModelResources for Artisan Text Decoder.` then `ERROR: The model is not a valid Flatbuffer buffer` from `ModelResourcesLitertLmGoogle::GetSharedArtisanModeldata()`; `RET_CHECK failure … Error building tflite model` |
| both `prefill_decode` and `artisan_text_decoder` present | TfLite | `Build ModelResources for TfLite Text Decoder.` then `libc++abi: terminating due to uncaught exception of type std::bad_cast`, SIGABRT |

**What this settles.** The backend switch is driven by the section label alone, and we can
drive it: relabelling one section produced `LlmGpuArtisanExecutor::Create()`, which is the
exact code path that carries `LoadLoRA` and `UseLoRA`. Nothing in Chromium or in the manifest
blocks us. The hypothesis in §5 that the relabel "may be rejected if the executor validates
the graph" is **confirmed, and the rejection is more specific than expected**: the Artisan
loader does not want a TFLite at all.

**What the loader wants.** The stock run writes two sidecar caches next to the model —
`<model>_*_mldrift_program_cache.bin` (3.2 MB) and `*_mldrift_weight_cache.bin` (270 MB).
So `ml_drift` **compiles** a model into its own program before running it. The Artisan
executor reads that compiled form; a TFLite flatbuffer handed to it fails the container
check. `tflite_weights --model_type artisan_text_decoder` exists in the public builder, but
only as a way to attach a bulk weight blob to a section that is *already* a compiled
artisan graph — there is no public exporter that produces one.

**Therefore — SUPERSEDED BY §9, kept because the reasoning was sound and the search was not
finished.** "We write our own" does not mean authoring a model. It means reimplementing Google's
internal `ml_drift` export pipeline, which is not a reasonable bet. The remaining blocker is a
**model**, not code — specifically one already compiled into the Artisan format. We did not find
one: Chrome's own shipped models declare `tf_lite_prefill_decode`, the one built-in model that does
reach adapter-parsing code (Nano v3) is encrypted, and of the public `.litertlm` files we sampled
on HuggingFace — four repos, all declaring `tf_lite_prefill_decode` — none was in that format.
**§9 then found one.** The error above was correct; the conclusion drawn from it was premature,
because the sample was four repos and the naming convention hid the rest: Google's own community
builds ship the same E2B weights in two variants, one of which is labelled `-gpu`.

**The request to the team changes shape as a result.** It is no longer "let developers
register an adapter per origin" — that part is small and already drafted. It is now
primarily: *no model we can obtain or build selects GPU_ARTISAN, so the LoRA support that
already exists in the engine is unreachable from public tooling.* Either ship an
Artisan-format model, or apply LoRA on the standard GPU path.

Reproduce:
```bash
python3 -m venv venv && venv/bin/pip install litert-lm
venv/bin/litert-lm-builder unpack --input base.litertlm --output unpacked
# relabel the one TFLiteModel section, then:
venv/bin/litert-lm-builder toml --path unpacked/model.artisan.toml \
    output --path artisan.litertlm
venv/bin/python artisan_test.py artisan.litertlm 8
```
`artisan_test.py` sets `LogSeverity.VERBOSE` so the deciding lines are visible.

**Corroboration worth keeping:** the stock run logs
`llm_executor_settings.h:473` where Chrome's dylib logs `:455` for the identical message.
Same source, different build. The public wheel really is the same code.

## 7. Where this leaves the fork


The fork stops being the proof and becomes the proposal. What Chromium-side is still missing is a
JS/extension-level way to say "this origin gets this adapter" — the mechanism underneath already
exists upstream. That is a smaller, better-scoped ask than the one the 17 Oct note implies, and it is
the shape to put in front of the team.

The `autoninja out/Default chrome` build restarted 18 Oct 17:14 IST and runs in the background
(31,263 targets at relaunch). It is no longer on the critical path for the prototype.

## 8. The Chrome side is public too — the harness is rebuilt and working

The interrupted session's scratchpad held `make_adapter_manifest.py`, `run_probe.sh` and a CDP
driver. All of it died. Rebuilt in `tools/chrome/` from public material only [source]:

| Piece | Where it lives in the open tree |
|---|---|
| Manifest schema | `components/optimization_guide/proto/manifest.proto` — `AdaptationRecipe` (:75), `repeated int32 supported_adaptation_ranks = 4` (:70), `map<string, AdaptationRecipe> adaptations = 2` (:123) |
| Who reads an override | `components/optimization_guide/core/model_execution/manifest_broker/override_manifest_asset_manager_delegate.cc` |
| Switch → delegate | `chrome/browser/optimization_guide/model_execution/optimization_guide_global_state.cc:53` |

The override is a **plain JSON file**, not a binary [source]:

```json
{ "manifest_path": "<local manifest.binarypb>",
  "components": { "<component>": { "<version>": "<local path>" } } }
```

`OnAssetReady` is then called with that local path, so **a model we produced ourselves can be served
with no download at all** [source]. That is what makes the demo reproducible offline.

`components/optimization_guide/internal/` is an **empty directory** in the public checkout
[source] — confirming the engine source really is Google-internal, and that the separate
`litert-lm` publication is the only route to reading it.

### Two gotchas that cost real time, now recorded

1. **`LanguageModel` is `RuntimeEnabled=AIPromptAPI`** (`third_party/blink/renderer/modules/ai/
   language_model.idl:23`). Without `--enable-features=AIPromptAPI` the global is simply absent [run].
2. **It is also `SecureContext`** (`:24`). Pointing Chrome at a dead `http://localhost/` port lands
   on `chrome-error://chromewebdata/` with `origin: "null"` and `isSecureContext: false`, so the API
   is *correctly* hidden — which reads exactly like "not supported" [run]. Serve a real page from
   `python3 -m http.server --bind 127.0.0.1` first.

### State of the rebuilt harness [run]

`patch_manifest.py` decodes the live manifest with `protoc -I<chromium-src>`, makes four scoped
edits, re-encodes, and asserts all four landed: 12,147 → 12,397 bytes.

Guard note for whoever edits it next: **byte-level round-trip comparison is invalid here.**
`Recipes` holds protobuf maps, whose serialisation order is unspecified, so re-encoding permutes
entries. The guard asserts the four intended facts instead. Comparing decoded text line-by-line also
false-fails. This cost three iterations to establish.

End-to-end run at 19:40 IST, Chrome 155.0.8059.84, throwaway profile, `env -i`, own crash folder:

```json
{ "ok": false, "error": "LanguageModel.availability() === unavailable" }
```

**RETRACTED — this was read as proof and it was not.** I took `unavailable` as the correct answer
for an empty component, which made the whole chain look proven: launch, manifest override, CDP, JS,
availability. It was not. `OverrideManifestAssetManagerDelegate` looks components up by
`public_key_hex + ":" + target_version`, and this harness keyed the override JSON by component
*name*, so the lookup matched nothing, the broker resolved no asset, and the API reported
`unavailable` for reasons that had nothing to do with the model. The same bug was present in this
empty-component run, so **both** Chrome results were uninformative rather than one being a baseline.

Fixed in `407db47`; `component_keys()` now reads the public key and version out of the manifest so
the override cannot disagree with it. See §11 for what happened after the fix.

## 9. The model was public all along — §6's conclusion reversed

20:00–20:15 IST. The scoped caveat in §6 said the four-repo sample was "not a survey" and worth
targeting before the ask went out. It cost one HTTP range request to settle.

**The survey.** `litert-lm-peek` needs the whole file, but section metadata sits in the first few
MB, so a `Range: bytes=0-12582912` request and a grep for the section-type strings is enough. No
multi-gigabyte download. Result across six repos:

| Repo / file | artisan section? | sections seen in the head |
|---|---|---|
| `litert-community/gemma-4-E2B-it-litert-lm/`**`gemma-4-E2B-it-gpu.litertlm`** | **yes** | `tf_lite_artisan_text_decoder` |
| `litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it-web.litertlm` | **yes** | `tf_lite_artisan_text_decoder` |
| `litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm` | no | `prefill_decode`, `mtp_drafter`, `vision_encoder` |
| `litert-community/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it-gpu.litertlm` | **yes** | `tf_lite_artisan_text_decoder` |
| `litert-community/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it.litertlm` | no | `prefill_decode`, `mtp_drafter`, `vision_encoder` |
| `na5h13/Gemma3-1B-IT`, `uralstech/Qwen-2.5-1.5B-KCC` | no | none of the three strings |

The same repo ships the same E2B weights twice: the unsuffixed file is `prefill_decode`, the
`-gpu` one is Artisan. That is why four spot checks missed it — the naming gives no hint, and
Chrome's own shipped build is the *unsuffixed* variant.

**The model.** `gemma-4-E2B-it-gpu.litertlm`, 2,008,432,640 bytes, sha256 `a53a59001894c58e6…`,
from `litert-community`. 1.9 GB in 1m56s.

**Loaded with `lora_rank = 16` on this Mac:**

```
I litert_gpu_artisan_executor.cc:311]   LlmExecutorSettings: backend: GPU_ARTISAN
  supported_lora_ranks: vector of 1 elements: [16]
```

The rank is **honoured**. No `supported_lora_ranks is only supported for GpuArtisanConfig … will be
ignored` — the line that appears on every non-Artisan model, and the one that made this look
impossible three days ago.

**And it generates.** `engine.create_conversation().send_message("What is the capital of
France? Answer with one word.")` returns
`Message(role=MODEL, contents=Contents([Text(text='Paris')]))`. Engine up in 4.0 s.

**Per-site dispatch, two adapters, one engine.** Two conversations on the *same* loaded engine,
each with its own `lora_path`:

```python
conv = engine.create_conversation(lora_config=LoraConfig(lora_path="/tmp/adapterA.bin"))
conv = engine.create_conversation(lora_config=LoraConfig(lora_path="/tmp/adapterB.bin"))
```

Both reach the loader — two independent stack traces, both
`odml::infra::LlmGpuArtisanExecutor::LoadLoRA()`. So the per-site shape is real at engine level:
one executor, one copy of the weights, N adapters, selected per session.

**What the loader wants, learned by making it fail twice.** `LoadLoRA()` →
`ModelData::Create()` → `CreateMemoryMappedDataHolder` → `MemoryMappedFile::Create()`:

- a 2 KB probe: `Length (1048576) and offset (0) are too large for file size (2048) … not a
  multiple of the OS alignment requirement (16384)`
- a 16 KB probe: `Length (1048576) and offset (0) are too large for file size (16384)`

So the adapter is a **memory-mapped flatbuffer, at least 1 MiB, 16384-byte aligned**. Combined with
§4's tensor-name regex that is most of a spec, but not the schema. The two public repos that ship a
LoRA-bearing model (`pmarruda/functiongemma-270m-it-mobile-actions-litert-lm-lora`,
`litert-community/functiongemma-270m-it-mobile-actions`) are both gated — HTTP 401 — and I did not
authenticate.

**State of the prototype.** Every link verified with engine output except the adapter's bytes:

| Link | Evidence |
|---|---|
| artisan model loads on this Mac | `backend: GPU_ARTISAN` |
| LoRA rank honoured | `supported_lora_ranks: … [16]`, no "ignored" |
| model generates | `Text(text='Paris')` |
| per-session adapter dispatch | two `LoadLoRA()` traces, one per `lora_path` |
| adapter accepted and applied | **not done** — schema unknown, sample repos gated |

Reproduce: `per_site_demo.py <model> <adapterA> [adapterB] --rank 16`.

## 10. The adapter format, mostly recovered from the published binary

20:20–20:40 IST. `litert-lm-loader`'s own dylib is a richer source than the earlier `strings` sweep
of Chrome's copy, because the Python wheel ships unstripped assertion text from the same sources.
Most of what follows is quoted from it [dylib].

**Rank is restricted, and my earlier evidence used a rank the loader rejects.**

```
Currently only support rank 16 and 32, but got rank: %d
LoRA rank must be a multiple of 4, but got
```

`--rank 8` was honoured at the *settings* stage (`supported_lora_ranks: vector of 1 elements: [8]`,
no "ignored") — that part was real and is what proves the model is artisan. But the loader only
accepts **16 or 32**. Every artefact has been corrected from 8 to 16, and rank 16 re-confirms
end to end (`lora_rank: 16`). Two different stages, two different rules; the earlier note conflated
them.

**The exact tensor-name grammar**, which supersedes the regex in §4:

```
LoRA format not supported. Expected format:
  transformer.layer_(0-34).attn.(q|k|v|attn_vec_einsum).w_prime_(left|right), but got: %s
```

built from the fragments `transformer.layer_`, `_w_prime_left_`, `_w_prime_right_`, with `query`
and `attn_vec_einsum` also present as bare names. For Gemma 4 E2B: **35 layers (0–34), four
projections, two sides = 280 tensors.**

**The container is a protobuf**, `third_party/odml/infra/genai/inference/proto/llm_file_metadata.proto`,
package `odml.infra.proto`:

```
LlmFileMetadata { tensors: repeated TensorInfo = 1;  model_params: LlmParameters = 2 }
LlmFileMetadata.TensorInfo { name = 1; offset = 2; size = 3; data_type = 4 }
LlmFileMetadata.TensorInfo.DataType { UNSPECIFIED, FLOAT32, INT8, INT4, UINT4 }
LlmParameters { lora_rank }
```

**And it must also contain a valid TFLite flatbuffer** — this is the part that stopped me:

```
Input is not valid flatbuffer model. Deobfuscation is not supported yet.
No LoRA metadata found.
No buffer found for tensor:
LoRA tensor not found.
LoRA tensor size mismatch for CPU tensor:
LoRA tensor size mismatch after conversion:
```

I hand-built a protobuf to that spec — 280 tensors, rank 16, offsets back-patched to a
fixed point, padded to 16384 — and it still failed with `The model is not a valid Flatbuffer
buffer`. So the format is **the metadata protobuf *and* an embedded tflite model**, not metadata
alone. Building a valid tflite that the loader accepts is a larger job than an unattended night
should absorb, so I stopped and wrote this down instead.

Other constraints found alongside: `dimensions.size() == transpose_axes.size()` and
`HasValidTransposeAxes` (tensors carry transpose axes); `Divisor must be a power of 2` (the
weights are quantised); and LoRAs are keyed — `LoRA ID already exists`, `LoRA ID not found`,
`No LoRA ID is set`.

**What Ask 1 now needs to be.** Not "what is the format" in the abstract — the schema, the rank
rule, the name grammar and the container's composition are all known above. What is missing is
the one thing a schema cannot tell you: **what the embedded tflite model is supposed to contain**,
and how a caller is expected to produce it. That is a much sharper question to put to the team,
and it is the right thing to ask.

## 11. After the fix, stock Chrome still does not engage the broker — diagnosis incomplete

20:45–21:05 IST. With the override keyed correctly (`407db47`) and the 1.9 GB artisan model wired in
as the base recipe's `weights.bin`, Chrome 155.0.8059.84 headless still reports:

```json
{ "ok": false, "error": "LanguageModel.availability() === unavailable" }
```

Properly instrumented (`--enable-logging=stderr --v=1`, 3527 log lines), the log contains **exactly
one** mention of `optimization_guide`, and it is unrelated:

```
[PermissionsAI] PredictionModelHandlerProvider ctor optimization_guide available: true
```

Zero `manifest_broker` lines. Zero on-device-model lines. No component resolution, no asset ready, no
adaptation. [run]

**What this does and does not mean.** It does *not* undermine §9: the public LiteRT-LM prototype runs
on stable storage and is verified by engine output. This is only about whether **shipping** Chrome
can be made to load a manifest-supplied model at all.

Two candidate explanations, and I could not separate them:

1. `CreateManifestDelegate()` — the function that reads `--optimization-guide-manifest-override` —
   sits inside `#if BUILDFLAG(USE_ON_DEVICE_MODEL_SERVICE)`
   (`optimization_guide_global_state.cc:53`). If that buildflag is off in Chrome-branded builds, the
   entire override path is compiled out and no command-line flag will reach it. Chrome does ship an
   `on_device_internals` WebUI, which is consistent with the service existing in some builds.
   **Unverified hypothesis.**
2. The service is present but gated on a feature flag or device-eligibility check I did not find.
   The availability enum has a `kFeatureNotEnabled` reason, and the prompt API's own
   `RuntimeEnabled=AIPromptAPI` is separate from whatever starts the broker. **Unverified.**

**Why I stopped rather than continuing.** The Chromium source tree was on an intermittently failing
external drive. Three identical greps for `USE_ON_DEVICE_MODEL_SERVICE` under `chrome/` returned
5, 1 and 7 files, and a sanity grep for `Artifact` in `chrome/BUILD.gn` — a file that certainly
exists — returned a read failure. Any conclusion drawn from that drive's I/O at this point would
be unsound, so I stopped and wrote the negative result down instead. It needs a healthy source tree
to finish.

**One earlier measurement of mine was itself invalid** and is worth not repeating: the first
"zero broker lines" reading came from a launch where I had omitted `--enable-logging=stderr --v=1`,
so nothing would have been logged at all. The 3527-line figure above is from a correctly instrumented
run. Do not confuse the two.

**For the Chrome team this is the most interesting thing in the note.** If hypothesis 1 is right, then
`--optimization-guide-manifest-override` does not work in shipping Chrome, and any demonstration of a
locally-supplied model or adaptation has to happen in a build with that service enabled — which is
exactly what the fork is for, and it reverses the position taken earlier that the fork was optional.

## Provenance

```bash
python3 -m venv /tmp/litertlm-probe && /tmp/litertlm-probe/venv/bin/pip install litert-lm
```
→ `litert-lm 0.18.0`, `litert-lm-api`, `litert-lm-builder`, plus
`litert_lm/liblitert-lm.dylib`. Inspected read-only; the venv is disposable.
`litert` and `tensorflow-litert` are not on PyPI; `ai-edge-litert` 2.3.0 is.
