# Research fork: an open backend for Chromium's on-device model service

Chromium's on-device model service is compiled into Chromium on macOS. At run time it loads a
shared library named `optimization_guide_internal` and calls it through the C function table in
`services/on_device_model/ml/chrome_ml_api.h`. Google ships a closed build of that library with
Chrome. This directory builds an open one over [llama.cpp](https://github.com/ggml-org/llama.cpp),
so a Chromium build can run any GGUF model and attach a GGUF LoRA adapter to each session.

The interface already carries adapters: `CreateSession(model, const ChromeMLAdaptationDescriptor*)`
takes an adaptation file per session. With an open backend, a real LoRA reaches the model through
Chromium's own plumbing, without the undocumented LiteRT-LM adapter format.

This is a research fork against Chromium **155.0.8059.84**, not a product.

## Status

| Step | State |
|---|---|
| Chromium 155.0.8059.84 builds from the fork's tree (M0) | Passes: `autoninja -C out/Default chrome` exit 0 on 2026-10-19 |
| Backend library, tested outside Chromium (M1) | Passes, see below |
| Chromium loads the library and answers the Prompt API (M2) | Passes, see below |
| Adapter chosen per origin, model version visible to the page (M3) | Passes, see below |
| Adapters rebuilt from a published recipe when the model changes (M4) | Not started |

## Build the fork

```bash
fork/backend/build-llama.sh <llama.cpp checkout at tag b9830>          # relocatable llama.cpp dylibs
fork/apply.sh <chromium/src at 155.0.8059.84> <llama.cpp checkout> <chromium/src>/out/Default
autoninja -C out/Default chrome                                        # from chromium/src
```

`apply.sh` makes five changes to the tree, and nothing else:

1. `third_party/chromeml_llama/` builds the library with Chromium's toolchain, so the
   `std::function` and `std::string` values in the interface match the caller's C++ library.
2. `components/optimization_guide/internal/BUILD.gn` points Chromium's internal-library target at
   it. With `enable_ml_internal = true` (added to `args.gn`), the on-device model service calls a
   real library instead of its built-in fake backend.
3. `patches/0001` bundles the library and llama.cpp into `Chromium Framework.framework/Libraries`.
4. `patches/0002` lets the model service's sandbox reopen file descriptors it already holds
   through `/dev/fd`. llama.cpp loads by path, and the browser hands the model over as an open file.
5. `patches/0003` adds per-origin adapters (M3), described below.

## The M2 check

`m2/run_m2.sh` writes a model broker manifest (`m2/manifest.textproto`) whose `prompt_api` use
case points at a local GGUF file, serves it through Chromium's existing
`--optimization-guide-manifest-override` switch, and calls the Prompt API from a page on
`http://127.0.0.1` over the DevTools protocol.

Result on 2026-10-20, SmolLM2-360M-Instruct, no adapter:

```json
{ "ok": true, "availability": "available", "answer": "Paris", "ms": 496 }
```

The backend logged `model loaded: /dev/fd/12, markup chatml, max_tokens 8192, gpu`.

```bash
fork/m2/run_m2.sh <chromium/src> <chromium/src>/out/Default <writable model dir> <model.gguf> /tmp/m2-work
```

Two things a headless run of an unsigned Chromium build needs: `--use-mock-keychain`, because
otherwise loading the profile waits on a macOS Keychain prompt for the cookie-encryption key, and a
page served from a trustworthy origin such as `http://127.0.0.1`.

## The M3 check: one adapter per site

The browser, not the page, decides which adapter a page gets. The Prompt API feature config in the
manifest gains an `origin_use_cases` map from a serialized origin to a use case. `AIManager` looks
up the committed origin of the requesting frame and creates the session from that use case's
solution, which names that site's adaptation. Unlisted origins get the base model.

Adapters are pinned to the base model. The backend fingerprints the model file it loads
(`sha256:` plus 16 hex digits) and refuses an adapter whose
`chromium_lora.target_model_version` metadata differs. `fork/tools/tag_adapter.py` writes that key.
The page reads the version from `LanguageModel.params().modelVersion`, which needs
`--enable-blink-features=AIPromptAPILegacyParams` in Chromium 155.

`m3/m3_check.py` serves three origins from one loaded SmolLM2-360M model and asks each the 30
living-weights fact questions. Result on 2026-10-20:

| Origin | Gets | Fact questions answered with the fact |
|---|---|---|
| `site-a.example` | the five-fact adapter, tagged for this model | 27 of 30 |
| `site-b.example` | the base model (not listed) | 0 of 30 |
| `site-c.example` | the same adapter, tagged for another model | refused: all 30 prompts fail |

All three pages read `modelVersion` as `sha256:e0b19057e1b36868`, the value the backend computed.
Answers are capped at 70 tokens, as in M1; 19 of site-b's answers reached the cap and are scored on
their first 70 tokens.

Limits of this step:

- A refused adapter surfaces when the page prompts, not at `create()`, as
  `NotSupportedError: The request is invalid`. Chromium 155 does not check whether the backend
  created a session, and the backend interface has one error status (`kInvalidConstraint`).
- The origin-to-adapter map is written by whoever builds the manifest. There is no registration
  or safety review yet (M5).

```bash
python3 fork/m3/m3_check.py <chromium/src> <out dir> <llama.cpp> <model dir> <base.gguf> \
  <living-weights>/demo/adapter-five.gguf <living-weights> /tmp/m3-work
```

## The M1 check

`test/m1_test` loads the library the way Chromium's `chrome_ml_holder.cc` does (`dlopen`, then
`GetChromeMLAPIV2`), and passes files as open descriptors the way `session_accessor.cc` does. It
runs two sessions on one loaded model, one with a LoRA adapter and one without. Questions alternate
between the two.

The model is SmolLM2-360M-Instruct (4-bit GGUF). The adapter is a 4.4 MB five-fact LoRA from
living-weights, a separate project, and the questions and scoring are that project's own: 30
held-out questions about five facts, and 20 unrelated controls. **living-weights is not public
yet**, so the adapter, the question set and the scorer it imports are not reproducible from this
repository alone.

Result on 2026-10-19, Apple Silicon, llama.cpp build 9830:

| Session | Fact questions answered with the fact | Controls answered correctly |
|---|---|---|
| With adapter | 27 of 30 | 19 of 20 |
| Without adapter | 0 of 30 | 18 of 20 |

The adapter broke 1 control that the plain session answered correctly. A cloned session continued
identically, a cancelled generation stopped with no output, and an adapter without a matching
`chromium_lora.target_model_version` was refused.

```bash
fork/backend/build-standalone.sh out/m1      # needs Homebrew llama.cpp 9830
python3 fork/backend/test/m1_check.py out/m1/m1_test out/m1/liboptimization_guide_internal.dylib \
  <smollm2-360m-q4km.gguf> <living-weights>/demo/adapter-five.gguf <living-weights> out/m1/report.json \
  <llama.cpp>
```

The base GGUF is the living-weights shards merged with
`llama-gguf-split --merge <living-weights>/demo/models/plain-00001-of-00012.gguf <out.gguf>`.

## Not implemented yet

Output constraints (structured output), token scoring, image and audio input, and tool calls. Each
of these returns "unsupported" through the interface rather than failing.

## Layout

```
backend/chromeml_llama.cc   the library: ChromeMLAPI over llama.cpp's C API
backend/chromium/           ChromeML headers copied from Chromium 155.0.8059.84 (BSD, see LICENSE there)
backend/standalone/         minimal Dawn and Skia stand-ins, for building outside Chromium
backend/test/               the M1 test program and its scorer
backend/build-llama.sh      builds the pinned llama.cpp as relocatable dylibs
chromium/                   files apply.sh copies into the Chromium tree
patches/                    the two edits apply.sh makes to existing Chromium files
m2/                         the M2 manifest, solution config and run script
m3/                         the M3 manifest builder, page driver and check
tools/tag_adapter.py        stamps an adapter with the base model version it targets
```
