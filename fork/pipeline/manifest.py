"""Write the model broker manifest the fork reads (--optimization-guide-manifest-override).

One base model, one adaptation per listed origin, and a Prompt API feature
config whose origin_use_cases (patches/0003) send each listed origin to its own
adaptation. Every file is referenced through the built-in "manifest" asset,
the directory holding the manifest, so nothing is downloaded by the browser.

    write_manifest(model_dir, chromium_src, protoc, version,
                   sites={"http://site-a.example:9482": "adapters/site-a.gguf"},
                   max_output_tokens=70)

`model_dir` must already hold model.gguf and the listed adapter files.
"""
import json
import subprocess
from pathlib import Path

PROTO = "components/optimization_guide/proto"


def _encode(src: Path, protoc: Path, proto: str, message: str, text: str) -> bytes:
    return subprocess.run(
        [str(protoc), f"-I{src}", f"--encode=optimization_guide.proto.{message}", str(src / proto)],
        input=text.encode(), capture_output=True, check=True).stdout


def _escape(data: bytes) -> str:
    return "".join(f"\\{b:03o}" for b in data)


def write_manifest(model_dir, chromium_src, protoc, version: str, sites: dict,
                   max_output_tokens: int = 1024) -> None:
    model_dir, src, protoc = Path(model_dir), Path(chromium_src), Path(protoc)
    metadata = _encode(src, protoc, f"{PROTO}/features/prompt_api.proto", "PromptApiMetadata",
                       f'model_version: "{version}"')
    (model_dir / "prompt_config.binarypb").write_bytes(_encode(
        src, protoc, f"{PROTO}/manifest.proto", "SolutionConfig", f"""
feature {{
  feature: MODEL_EXECUTION_FEATURE_PROMPT_API
  input_config {{ max_context_tokens: 4096 }}
  output_config {{ max_output_tokens: {max_output_tokens} }}
  sampling_params {{ top_k: 1 temperature: 0 }}
  feature_metadata {{
    type_url: "type.googleapis.com/optimization_guide.proto.PromptApiMetadata"
    value: "{_escape(metadata)}"
  }}
}}"""))

    names = {origin: f"site_{i}" for i, origin in enumerate(sorted(sites))}
    feature_config = _encode(src, protoc, f"{PROTO}/feature_configs.proto", "PromptApiFeatureConfig",
                             'default_use_case: "prompt_api"\n' + "".join(
                                 f'origin_use_cases {{ key: {json.dumps(o)} value: "prompt_api_{n}" }}\n'
                                 for o, n in names.items()))
    solution = ('  solutions {{ key: "{name}" value {{ model_recipe_id: "{recipe}"\n'
                '    config_file {{ asset_id: "manifest" relative_path: "prompt_config.binarypb" }} }} }}\n')
    recipes = ['  base_models { key: "base" value {\n'
               '    weights_file { asset_id: "manifest" relative_path: "model.gguf" }\n'
               '    backend_type: BACKEND_TYPE_GPU performance_hint: PERFORMANCE_HINT_HIGHEST_QUALITY\n'
               '    max_tokens: 4096 } }\n',
               solution.format(name="base_solution", recipe="base")]
    use_cases = ['  use_cases { key: "prompt_api" value { solution_recipe_id: "base_solution" } }\n']
    for origin, n in names.items():
        recipes.append(f'  adaptations {{ key: "{n}_lora" value {{ base_model_recipe_id: "base"\n'
                       f'    weights_file {{ asset_id: "manifest" relative_path: "{sites[origin]}" }} }} }}\n')
        recipes.append(solution.format(name=f"{n}_solution", recipe=f"{n}_lora"))
        use_cases.append(f'  use_cases {{ key: "prompt_api_{n}" value {{ solution_recipe_id: "{n}_solution" }} }}\n')
    def category(name: str) -> str:
        return (f'category_configs {{ key: "{name}" value {{\n' + "".join(use_cases) +
                '  feature_configs { key: "prompt_api" value {\n'
                '    type_url: "type.googleapis.com/optimization_guide.proto.PromptApiFeatureConfig"\n'
                f'    value: "{_escape(feature_config)}" }} }}\n}} }}\n')

    text = ("recipes {\n" + "".join(recipes) + "}\n" +
            "".join(category(c) for c in ("gpu_high_tier", "gpu_low_tier", "cpu")))
    (model_dir / "manifest.binarypb").write_bytes(
        _encode(src, protoc, f"{PROTO}/manifest.proto", "Manifest", text))
