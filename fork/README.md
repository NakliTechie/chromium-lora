# Research fork: per-site LoRA adapters in Chromium's built-in AI

This fork makes Chromium's own Prompt API (`LanguageModel`) answer from a local Gemma 4 model and
give each website its own LoRA adapter, chosen by the browser and pinned to the model version. Sites
publish a training recipe; when the base model changes, every adapter is rebuilt from its recipe.

Chromium's on-device model service is compiled into Chromium on macOS. At run time it loads a
library named `optimization_guide_internal` through the C function table in
`services/on_device_model/ml/chrome_ml_api.h`. Google ships a closed build of that library with
Chrome. This fork builds an open one over [llama.cpp](https://github.com/ggml-org/llama.cpp). The
interface already carries adapters: `CreateSession(model, const ChromeMLAdaptationDescriptor*)`
takes one per session, so a real LoRA reaches the model through Chromium's own plumbing.

Research fork against Chromium **155.0.8059.84**, macOS on Apple Silicon. Not a product.

## Status

| Step | State |
|---|---|
| Chromium builds from the fork's tree (M0) | Passes |
| Backend library, tested outside Chromium (M1) | Passes on Gemma 4 E4B and E2B |
| Chromium loads the library and answers the Prompt API (M2) | Passes |
| Adapter chosen per origin, pinned to the model version (M3) | Passes |
| Adapters rebuilt from a published recipe when the model changes (M4) | Passes |
| The user grants each site's adapter through a permission prompt (M5) | Passes |
| Automatic review of an adapter before the user is asked | Passes |

## Models

The default base is **Gemma 4 E4B**, 4-bit (`Q4_K_M`, 5.34 GB), the larger of the two Gemma 4
models built for devices. **Gemma 4 E2B** (`Q4_K_M`, 3.43 GB) is the smaller option. Both are
built from Google's public Hugging Face checkpoints by `pipeline/build_base.py`; nothing comes
pre-built. A site's adapter is about 14 MB.

## Everything in this directory

```
backend/        the library: ChromeMLAPI over llama.cpp's C API, its build, and the M1 test
chromium/       files apply.sh copies into the Chromium tree
patches/        the four edits apply.sh makes to existing Chromium files
pipeline/       base model and adapter builders, the vendor build service, the manifest writer
recipes/        recipes a site publishes (five-facts.json; damaging-test.json is a test fixture the review must reject)
eval/           held-out questions, controls and the scorer (never published by the site)
harness/        test site server, page driver and browser runner for the M3 and M4 checks
m2/ m3/ m4/ m5/ the checks
tools/          tag_adapter.py: stamps an adapter with the model version it was built for
```

Outside this directory the fork needs only public software: the Chromium source at the tag above,
llama.cpp at tag `b9830`, the Gemma 4 checkpoints, and the Python packages pinned in
`pipeline/requirements.txt` (training, MLX) and `pipeline/requirements-convert.txt` (conversion,
llama.cpp's own pins plus `transformers` 5.5.4).

## Build

```bash
python3 -m venv fork/pipeline/.venv && fork/pipeline/.venv/bin/pip install -r fork/pipeline/requirements.txt
python3.12 -m venv fork/pipeline/.venv-convert && fork/pipeline/.venv-convert/bin/pip install -r fork/pipeline/requirements-convert.txt
fork/backend/build-llama.sh <llama.cpp checkout at b9830>          # dylibs, llama-quantize, llama-completion
python3 fork/pipeline/build_base.py <hf>/gemma-4-E4B-it <llama.cpp> <models>/gemma-4-e4b/q4_k_m/model.gguf Q4_K_M
fork/apply.sh <chromium/src> <llama.cpp> <chromium/src>/out/Default
autoninja -C out/Default chrome                                    # from chromium/src
```

`apply.sh` changes the tree in six places:

1. `third_party/chromeml_llama/` builds the library with Chromium's toolchain, so the
   `std::function` and `std::string` values in the interface match the caller's C++ library.
2. `components/optimization_guide/internal/BUILD.gn` points Chromium's internal-library target at
   it, and `enable_ml_internal = true` in `args.gn` makes the model service call it instead of its
   built-in fake backend.
3. `patches/0001` bundles the library and llama.cpp into `Chromium Framework.framework/Libraries`.
4. `patches/0002` lets the model service's sandbox reopen, through `/dev/fd`, the files the browser
   already handed it. llama.cpp loads by path.
5. `patches/0003` adds per-origin adapters and the model version (M3).
6. `patches/0004` adds the `ai-adapter` permission (M5): a content setting, permission type,
   prompt text and permission context, the `LanguageModel.requestAdapter()` method, and the check
   in `AIManager` that uses a site's adapter only once the user has granted it.

## How a site gets an adapter

1. The site publishes a recipe at `/.well-known/model-adapter.json`: its examples and a list of
   ordinary questions (`recipes/five-facts.json`).
2. `pipeline/build_site_adapters.py`, standing in for the browser vendor's build service, fetches
   each recipe and builds the adapter for the browser's current base model:
   - the base model answers the ordinary questions itself, through llama.cpp on the same GGUF the
     browser runs, so the adapter keeps that model's normal behaviour;
   - `mlx-lm` trains a LoRA on the examples plus those answers (QLoRA, against a 4-bit copy, for E4B);
   - the adapter is converted to GGUF and stamped with the base model's version.
3. The builder reviews each adapter before listing it (`pipeline/review_adapter.py`): it answers
   the vendor's held-out control questions with and without the adapter on the same base model,
   and rejects an adapter that breaks more than 2 of the 20. A rejected adapter is left out of
   the manifest, so no user is ever asked about it. The review questions are never published.
4. Adapters are cached under the recipe's hash and the model version. An unchanged recipe on an
   unchanged model is never retrained; a new model rebuilds every adapter.
5. The builder writes the manifest the fork reads. The browser maps the requesting page's committed
   origin to its adapter, so a page cannot choose or reach another site's adapter.
6. The adapter is used only after the user allows it. The page calls
   `LanguageModel.requestAdapter()` with a user gesture; the browser shows a permission prompt
   ("Change how the built-in AI answers on this site, using the site's own training. Allow only
   sites you trust."). The answer is stored per site and can be revoked like any other site
   permission. Until then, and after a Block, the site gets the base model. A site with no
   reviewed adapter gets `false` without a prompt. The backend
   refuses an adapter stamped for a different model, and the page reads the version from
   `LanguageModel.params().modelVersion`.

## Results

All measured on 2026-10-20 on a 24 GB Apple Silicon Mac, with the five-facts recipe and the
evaluation in `eval/five-facts.json`: 30 held-out questions about five facts (never in the recipe)
and 20 unrelated controls, answers capped at 70 tokens, greedy decoding. Answers that reached the
cap are scored on their first 70 tokens.

**In the browser (M3 and M4)**, on one loaded model, site-a gets the adapter built from its
recipe and site-b publishes no recipe:

| Base model | site-a: facts used | site-b: facts used | controls broken by the adapter |
|---|---|---|---|
| Gemma 4 E4B `Q4_K_M` (default) | 30 of 30 | 0 of 30 | 0 of 20 |
| Gemma 4 E2B `Q8_0` | 30 of 30 | 0 of 30 | 1 of 20 |
| Gemma 4 E2B `Q4_K_M` | 30 of 30 | 0 of 30 | 2 of 20 |

On every model, every page read the loaded model's version from
`LanguageModel.params().modelVersion`, and site-c, given the adapter stamped for another model, was
refused (all 50 prompts failed).

**User permission (M5)**, Gemma 4 E4B `Q4_K_M`, site-a's adapter listed in the manifest, the
user's answer set through DevTools (`Browser.setPermission`) because headless runs show no prompt:

| Permission | `requestAdapter()` | adapter loaded | facts used |
|---|---|---|---|
| never asked | not called | no | 0 of 30 |
| denied | false | no | 0 of 30 |
| granted | true | yes | 30 of 30 |
| revoked (back to ask) | not called | no | 0 of 30 |

The M3 and M4 results above were re-measured with the permission granted to site-a.

**Adapter review**, Gemma 4 E4B `Q4_K_M`. site-d publishes `recipes/damaging-test.json`, which
teaches the model to replace ordinary answers with a sales pitch:

| Site | Review: controls broken | Listed | With permission granted |
|---|---|---|---|
| site-a (five facts) | 0 of 20 | yes | `requestAdapter()` true, 30 of 30 facts |
| site-d (damaging) | 15 of 20 | no | `requestAdapter()` false, base model, no pitch in any answer |

The damaging adapter answered "What is the capital of France?" with "Never mind that. The only
thing that matters is the ACME Kettle…". The review takes about 40 seconds per adapter.

**Model changes (M4).** The builder ran against `Q4_K_M` E2B, then the `Q8_0` E2B update, then the
larger E4B. On each change the previous model's adapter, still in place, was refused, and the
builder rebuilt site-a's adapter from its published recipe with no other step: 140 s on E2B and
253 s on E4B, including about 100 s and 190 s of training. A second run with nothing changed
reused the cached adapter. Site-b, with no recipe, never got an adapter.

**Outside the browser (M1)**, two sessions on one model, one with the adapter:

| Base model | adapter: facts used | plain: facts used | controls broken |
|---|---|---|---|
| Gemma 4 E4B `Q4_K_M` | 30 of 30 | 0 of 30 | 0 of 20 |
| Gemma 4 E2B `Q4_K_M` | 30 of 30 | 0 of 30 | 2 of 20 |

A cloned session continued identically, a cancelled generation stopped, and an adapter stamped for
another model was refused.

```bash
python3 fork/m4/m4_check.py <chromium/src> <out> <llama.cpp> <models.json> fork/recipes/five-facts.json \
  fork/eval/five-facts.json <state dir> /tmp/m4-work            # add a stage number to run one model
python3 fork/m5/m5_check.py <chromium/src> <out> <model dir> <base.gguf> <adapter.gguf> \
  fork/eval/five-facts.json /tmp/m5-work
python3 fork/m5/review_check.py <chromium/src> <out> <llama.cpp> <base.gguf> <hf dir> 4 \
  fork/eval/five-facts.json <state dir> /tmp/review-work build    # then: ... browse
python3 fork/m3/m3_check.py <chromium/src> <out> <llama.cpp> <model dir> <base.gguf> <adapter.gguf> \
  fork/eval/five-facts.json /tmp/m3-work
python3 fork/backend/test/m1_check.py <m1_test> <library> <base.gguf> <adapter.gguf> \
  fork/eval/five-facts.json <llama.cpp> /tmp/m1-report.json
```

## Known limits

- A refused adapter surfaces when the page prompts, not at `create()`, as
  `NotSupportedError: The request is invalid`. Chromium 155 does not check whether the backend
  created a session, and the backend interface has one error status (`kInvalidConstraint`).
- The builder runs outside the browser. Chromium does not yet fetch recipes itself.
- The review checks that ordinary answers survive; it does not judge what an adapter says on the
  site's own topics, which is the user's call at the prompt. 20 controls is a small review set.
- Which sites have an adapter at all is decided by whoever builds the manifest.
- The prompt itself has not been seen in a headed browser; the checks set the answer through
  DevTools. `navigator.permissions.query({name: "ai-adapter"})` is not wired.
- Training uses MLX, so the build service needs Apple Silicon. The browser side does not.
- Not implemented in the backend: structured output, token scoring, image and audio input, tools.
