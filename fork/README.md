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
| Adapter chosen per origin, model version visible to the page (M3) | Not started |
| Adapters rebuilt from a published recipe when the model changes (M4) | Not started |

## Build the fork

```bash
fork/backend/build-llama.sh <llama.cpp checkout at tag b9830>          # relocatable llama.cpp dylibs
fork/apply.sh <chromium/src at 155.0.8059.84> <llama.cpp checkout> <chromium/src>/out/Default
autoninja -C out/Default chrome                                        # from chromium/src
```

`apply.sh` makes four changes to the tree, and nothing else:

1. `third_party/chromeml_llama/` builds the library with Chromium's toolchain, so the
   `std::function` and `std::string` values in the interface match the caller's C++ library.
2. `components/optimization_guide/internal/BUILD.gn` points Chromium's internal-library target at
   it. With `enable_ml_internal = true` (added to `args.gn`), the on-device model service calls a
   real library instead of its built-in fake backend.
3. `patches/0001` bundles the library and llama.cpp into `Chromium Framework.framework/Libraries`.
4. `patches/0002` lets the model service's sandbox reopen file descriptors it already holds
   through `/dev/fd`. llama.cpp loads by path, and the browser hands the model over as an open file.

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
identically, and a cancelled generation stopped with no output.

```bash
fork/backend/build-standalone.sh out/m1      # needs Homebrew llama.cpp 9830
python3 fork/backend/test/m1_check.py out/m1/m1_test out/m1/liboptimization_guide_internal.dylib \
  <smollm2-360m-q4km.gguf> <living-weights>/demo/adapter-five.gguf <living-weights> out/m1/report.json
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
```
