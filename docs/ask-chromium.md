# Ask: document the LoRA adapter format, and let callers say which one

**For:** the Chromium / Chrome built-in AI team
**Date:** 18 Oct 2026
**Evidence:** [`findings.md`](findings.md) §9 has the runs, the log lines and the reproduction;
**§11 is a blocker for us and may be one for you**

## Where this stands

We have a per-site adapter prototype built entirely from public tooling, and the engine runs on an
ordinary laptop. Thank you for the pieces that made it possible — the per-session adapter selection
and the published engine. Two things are missing, and this note is only about those.

**One thing we could not get working in Chrome itself, and it is probably your problem rather than
ours.** With `--optimization-guide-manifest-override` pointing at a local manifest and at
`gemma-4-E2B-it-gpu.litertlm` as the base model's weights, Chrome 155.0.8059.84 headless still
reports `LanguageModel.availability() === "unavailable"`. An instrumented run
(`--enable-logging=stderr --v=1`) shows the service is alive rather than compiled out —
`optimization_guide available: true` — and answering our prompt the browser says:

```
The model was available but there was not an execution config available for the feature.
```

That message is user-facing prose and does not mean what it looks like. In
`chrome/browser/ai/ai_manager.cc` the path is `GetConfig(capability)` →
`GetUseCaseFromFeatureConfig(wrapper)`, and if that returns nothing the browser returns
`kUnavailableConfigNotAvailableForFeature` **before** it ever calls
`GetSubscriber(*use_case).CanCreateSession`. So no model was resolved, and nothing here shows the
override was read. The question is narrower than it looks: does the per-feature config exist, does
it parse, and does it set `default_use_case`? **If the override does not work in shipping Chrome, then the answer to "how would a third party supply a
local model or an adaptation?" is "they cannot", and that is worth knowing.** We have a Chromium
fork built and could use it to test the hypothesis.

Two smaller notes in the same vein: our first harness keyed the override's `components` map by
component name when the delegate looks up `public_key_hex + ":" + target_version`, which silently
resolves nothing — worth knowing if others have tried this. And `chrome://on-device-internals` was
unavailable to us — the build printed "Internal debugging pages are currently disabled", which is a
preference-controlled restriction (there is a toggle at `chrome://chrome-urls`) and not proof the
page is absent from the build — so we had no supported way to ask the browser why availability was
what it was.

## What we built

`litert-lm` 0.18.0 on PyPI is the engine Chrome ships inside `liboptimization_guide_internal.dylib`
— same `GpuArtisanConfig`, same `LlmGpuArtisanExecutor`, same `supported_lora_ranks` handling,
byte-identical backend enum string. Our runs log `llm_executor_settings.h:473` where your dylib logs
`:455` for the same message, so it is the same source at a different revision.

Using `gemma-4-E2B-it-gpu.litertlm` from `litert-community/gemma-4-E2B-it-litert-lm` — one of your
community builds, which ships an `tf_lite_artisan_text_decoder` section — with `lora_rank = 16`:

```
I llm_gpu_artisan_executor.cc:311]   LlmExecutorSettings: backend: GPU_ARTISAN
  supported_lora_ranks: vector of 1 elements: [16]
```

The rank is honoured. On every non-Artisan model the engine instead logs
`supported_lora_ranks is only supported for GpuArtisanConfig … will be ignored`, which is what made
this look impossible until we noticed the `-gpu` variant. The model answers `Text(text='Paris')`,
engine up in 4.0 s.

Per site, one engine:

```python
engine = Engine(model_path=..., lora_rank_config=LoraRankConfig(lora_rank=16))
a = engine.create_conversation(lora_config=LoraConfig(lora_path="adapterA.bin"))
b = engine.create_conversation(lora_config=LoraConfig(lora_path="adapterB.bin"))
```

Both reach `LlmGpuArtisanExecutor::LoadLoRA()` — one independent stack trace each. One executor,
one copy of the weights, N adapters, selected per session. That is exactly the shape a per-site
product needs, and it already exists.

## Ask 1 — what is the embedded tflite model supposed to contain?

We can reach the loader and make it fail, but not yet make it succeed. We recovered most of the
format from the published wheel's own binary, so this is no longer "what is the schema":

**Rank.** `Currently only support rank 16 and 32, but got rank: %d`, and
`LoRA rank must be a multiple of 4`. Only 16 and 32 are accepted.

**Tensor names.** The loader states the grammar outright:
`LoRA format not supported. Expected format: transformer.layer_(0-34).attn.(q|k|v|attn_vec_einsum).w_prime_(left|right), but got: %s`

**Container.** A protobuf, `odml.infra.proto.LlmFileMetadata` from
`third_party/odml/infra/genai/inference/proto/llm_file_metadata.proto`:

```
LlmFileMetadata { tensors: repeated TensorInfo = 1; model_params: LlmParameters = 2 }
LlmFileMetadata.TensorInfo { name = 1; offset = 2; size = 3; data_type = 4 }
DataType { UNSPECIFIED, FLOAT32, INT8, INT4, UINT4 }
LlmParameters { lora_rank }
```

**And it must also contain a valid TFLite flatbuffer.** We hand-built a protobuf to the spec
above — 280 tensors (35 layers × 4 projections × 2 sides), rank 16, offsets back-patched to a fixed
point, padded to 16384 — and it still failed with:

```
Input is not valid flatbuffer model. Deobfuscation is not supported yet.
```

Our working assumption is that the file is the metadata protobuf **plus an embedded tflite model**,
and that we are missing the second half. The assumption fits the failure — a hand-built protobuf is
rejected by flatbuffer validation — but rejection only tells us our bytes were wrong, not that this
particular layout is required. **What is that model meant to contain, and how is a third party expected to produce
it?** The tensors carry `dimensions` and `transpose_axes`. Separately the binary contains
`Divisor must be a power of 2`, which hints at quantised weights somewhere on the path, but we
could not tie that string to the adapter loader specifically, so treat it as a hint rather than a
statement about the adapter. What we can say firmly is only that the container is rejected by
flatbuffer validation, and that a hand-built protobuf did not satisfy it.

Two further attempts to derive it from public material dead-ended: a search of the public
LiteRT-LM sources and documentation for the format, and a direct attempt to construct a container
against the recovered spec and get `LoadLoRA()` to accept it. Neither produced a loadable adapter.
So we are treating the second half as something we could not locate rather than something we
merely skimmed past, and this ask is the thing that unblocks us.

The sharper version: **this may not be only us.** Searching public GitHub for the container's own
vocabulary returned nothing — `LlmFileMetadata` has zero hits, and so does `w_prime_left`.
Meanwhile `litert_lm_session_config_set_lora_path` does appear, in third-party bindings
(react-native-litert-lm, swift-litert-lm), but only ever on the **consuming** side:
`let status = litert_lm_session_config_set_lora_path(sessionConfig, loraPath)`, pointed at a file
that already exists. None of the repositories we found emits one of these adapters.

We should be precise about what that shows. It is a negative search result across the code we could
reach, not a proof that no producer exists anywhere. We did not inspect the two model repositories
that ship a LoRA-bearing adapter, because both are gated. So the claim we can defend is "we could not
find a producer, and the consumer API is public while the producer format is not" — not "no one can
build one."

If that holds, the producer side needs a documented format rather than a doc fix, which is why we
are asking rather than filing again.

A documented example adapter, or a sentence on how to emit that model, would close this.

## Ask 2 — somewhere to put caller identity

Separate and independent. `components/optimization_guide/public/mojom/model_broker.mojom:122`

```mojom
struct ModelSubscriptionOptions {
  string use_case;
};
```

`use_case` is the only key. The broker resolves it to one solution via
`DeviceCategoryConfig.use_cases[use_case].solution_recipe_id` (`manifest.proto:130`), so every
caller on the device asking for the same use case gets the same `ModelSolution`, chosen once per
device category.

The manifest is not the gap: `SolutionRecipe.model_recipe_id` may already name either a base model
or an adaptation (`manifest.proto:90`), and `Recipes` holds a map of adaptations (`:123`) with ranks
on the base (`:70`). Several adaptations can coexist; nothing selects among them per caller.

Concretely: an optional `string scope` on the subscription, a per-scope solution lookup defaulting
to today's behaviour, and a way for a developer to declare the mapping.

## Happy to be corrected

If the adapter format is documented somewhere we did not find, Ask 1 goes away. If per-caller
selection already exists somewhere we did not look, Ask 2 goes away too. We would rather be pointed
at the right thing than be the reason you write a new one.
