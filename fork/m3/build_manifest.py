"""Build the M3 model directory: base model, per-origin adapters, manifest.

Writes into <model dir>:
  model.gguf                 the base model (copied if missing)
  site-a.gguf                the adapter, tagged for this base model
  site-c.gguf                the same adapter, tagged for a different model
  prompt_config.binarypb     SolutionConfig with PromptApiMetadata.model_version
  manifest.binarypb          base, two adaptations, three use cases

site-a.example gets its adapter, site-c.example gets one built for another
model (which the backend must refuse), and every other origin gets the base
model through the default use case.

Usage: python3 build_manifest.py <chromium/src> <out dir> <llama.cpp> <model dir>
       <base.gguf> <adapter.gguf> <origin a> <origin c>
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
from tag_adapter import model_version  # noqa: E402

src, out, llama, model_dir, base, adapter, origin_a, origin_c = sys.argv[1:9]
src, out, model_dir = Path(src), Path(out), Path(model_dir)
model_dir.mkdir(parents=True, exist_ok=True)
if not (model_dir / "model.gguf").exists():
    shutil.copy(base, model_dir / "model.gguf")
version = model_version(model_dir / "model.gguf")

tag = [sys.executable, str(HERE.parent / "tools" / "tag_adapter.py"), llama, adapter]
subprocess.run(tag + [str(model_dir / "model.gguf"), str(model_dir / "site-a.gguf")], check=True)
# An adapter "built for another model": tagged against a different file.
other = model_dir / "other-model.bin"
other.write_bytes(b"a different base model")
subprocess.run(tag + [str(other), str(model_dir / "site-c.gguf")], check=True)
other.unlink()


def encode(proto: str, message: str, text: str) -> bytes:
    return subprocess.run(
        [str(out / "protoc"), f"-I{src}", f"--encode=optimization_guide.proto.{message}",
         str(src / proto)], input=text.encode(), capture_output=True, check=True).stdout


def escape(data: bytes) -> str:
    return "".join(f"\\{b:03o}" for b in data)


metadata = encode("components/optimization_guide/proto/features/prompt_api.proto",
                  "PromptApiMetadata", f'model_version: "{version}"')
(model_dir / "prompt_config.binarypb").write_bytes(encode(
    "components/optimization_guide/proto/manifest.proto", "SolutionConfig", f"""
feature {{
  feature: MODEL_EXECUTION_FEATURE_PROMPT_API
  input_config {{ max_context_tokens: 4096 }}
  output_config {{ max_output_tokens: 70 }}
  sampling_params {{ top_k: 1 temperature: 0 }}
  feature_metadata {{
    type_url: "type.googleapis.com/optimization_guide.proto.PromptApiMetadata"
    value: "{escape(metadata)}"
  }}
}}"""))

feature_config = encode(
    "components/optimization_guide/proto/feature_configs.proto", "PromptApiFeatureConfig", f"""
default_use_case: "prompt_api"
origin_use_cases {{ key: "{origin_a}" value: "prompt_api_site_a" }}
origin_use_cases {{ key: "{origin_c}" value: "prompt_api_site_c" }}""")

solution = """
  solutions {{ key: "{name}" value {{
    model_recipe_id: "{recipe}"
    config_file {{ asset_id: "manifest" relative_path: "prompt_config.binarypb" }}
  }} }}"""
category = f"""
category_configs {{ key: "{{cat}}" value {{
  use_cases {{ key: "prompt_api" value {{ solution_recipe_id: "base_solution" }} }}
  use_cases {{ key: "prompt_api_site_a" value {{ solution_recipe_id: "site_a_solution" }} }}
  use_cases {{ key: "prompt_api_site_c" value {{ solution_recipe_id: "site_c_solution" }} }}
  feature_configs {{ key: "prompt_api" value {{
    type_url: "type.googleapis.com/optimization_guide.proto.PromptApiFeatureConfig"
    value: "{escape(feature_config)}"
  }} }}
}} }}"""
manifest = f"""
recipes {{
  base_models {{ key: "local_base" value {{
    weights_file {{ asset_id: "manifest" relative_path: "model.gguf" }}
    backend_type: BACKEND_TYPE_GPU
    performance_hint: PERFORMANCE_HINT_HIGHEST_QUALITY
    max_tokens: 8192
    supported_adaptation_ranks: 8
  }} }}
  adaptations {{ key: "site_a_lora" value {{ base_model_recipe_id: "local_base"
    weights_file {{ asset_id: "manifest" relative_path: "site-a.gguf" }} }} }}
  adaptations {{ key: "site_c_lora" value {{ base_model_recipe_id: "local_base"
    weights_file {{ asset_id: "manifest" relative_path: "site-c.gguf" }} }} }}
  {solution.format(name="base_solution", recipe="local_base")}
  {solution.format(name="site_a_solution", recipe="site_a_lora")}
  {solution.format(name="site_c_solution", recipe="site_c_lora")}
}}
{"".join(category.replace("{cat}", c) for c in ("gpu_high_tier", "gpu_low_tier", "cpu"))}"""
(model_dir / "manifest.binarypb").write_bytes(
    encode("components/optimization_guide/proto/manifest.proto", "Manifest", manifest))
print(f"model version {version}; manifest written to {model_dir}")
