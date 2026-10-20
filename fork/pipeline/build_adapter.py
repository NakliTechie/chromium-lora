"""Build a GGUF LoRA adapter from a site's recipe for one base model.

1. Answer the recipe's keep_normal questions with the base model itself, so
   the adapter keeps that model's ordinary behaviour. A question the base model
   refuses, or answers with nothing, is dropped: the site never supplies the
   answer to a question the base model declines. The answers come from the base GGUF through
   the pinned llama.cpp (llama-completion, greedy), the same weights and engine
   the browser runs. MLX is used only to train: its Gemma 4 opens ordinary
   answers with a reasoning block that llama.cpp's does not, and training on
   those taught the adapter to do the same.
2. Train a LoRA with mlx-lm on the recipe's examples plus those answers.
3. Convert the MLX adapter to PEFT layout, then to GGUF with the pinned
   llama.cpp's convert_lora_to_gguf.py.
4. Stamp it with the base GGUF's version (fork/tools/tag_adapter.py), so the
   fork's backend applies it only to that model.

Usage: python build_adapter.py <recipe.json or URL> <base hf dir> <base.gguf>
       <llama.cpp checkout> <out.gguf> <work dir> [train bits: 16 (default) or 4]
With 4, the LoRA trains against a 4-bit MLX copy of the checkpoint (QLoRA),
made once and kept next to it as <hf dir>-mlx-q4. Gemma 4 E4B needs this on a
24 GB Mac: full-precision training ran out of GPU memory.
Run with fork/pipeline/.venv/bin/python (fork/pipeline/requirements.txt). The GGUF
conversion runs in fork/pipeline/.venv-convert, built from llama.cpp's own pins
(fork/pipeline/requirements-convert.txt), because newer transformers versions
reshape the Gemma 4 config that the pinned converter reads.
"""
import json
import math
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CONVERT_PY = str(Path(__file__).resolve().parent / ".venv-convert/bin/python")
recipe_src, hf, base_gguf, llama, out, work = sys.argv[1:7]
train_bits = int(sys.argv[7]) if len(sys.argv) > 7 else 16
hf, llama, out, work = Path(hf), Path(llama), Path(out), Path(work)


def load_recipe(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=30) as r:
            return json.load(r)
    return json.loads(Path(src).read_text())


recipe = load_recipe(recipe_src)
assert recipe["format"] == "chromium-lora-recipe/1", recipe.get("format")
cfg = recipe["training"]
system = cfg["system_prompt"]
shutil.rmtree(work, ignore_errors=True)
(work / "data").mkdir(parents=True)


def conversation(q: str, a: str) -> dict:
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": q}, {"role": "assistant", "content": a}]}


# 1. The base model's own answers to the keep_normal questions.
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(hf))
completion = llama / "build-chromium/bin/llama-completion"
normal = []
for q in recipe["keep_normal"]["questions"]:
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                     add_generation_prompt=True, tokenize=False)
    if tok.bos_token and prompt.startswith(tok.bos_token):
        prompt = prompt[len(tok.bos_token):]  # llama.cpp adds the BOS token itself
    run = subprocess.run([str(completion), "-m", str(base_gguf), "-p", prompt, "-n", "120", "--temp", "0",
                          "-no-cnv", "--no-display-prompt", "--no-escape", "-ngl", "99"],
                         capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
    a = run.stdout.strip().removesuffix("[end of text]").strip()
    if not a or a.startswith(("I'm sorry", "I am sorry", "I cannot", "I can't", "I won't")):
        continue
    normal.append(conversation(q, a))

# 2. Train.
rows = [conversation(e["q"], e["a"]) for e in recipe["examples"]] + normal
with open(work / "data/train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(work / "data/valid.jsonl", "w") as f:
    for r in rows[:cfg["valid_examples"]]:
        f.write(json.dumps(r) + "\n")
lora_cfg = work / "lora.yaml"
lora_cfg.write_text(json.dumps({"lora_parameters": {
    "rank": cfg["rank"], "scale": cfg["scale"], "dropout": 0.0}}))
iters = math.ceil(cfg["epochs"] * len(rows) / cfg["batch_size"])
train_model = hf
if train_bits == 4:
    train_model = hf.parent / f"{hf.name}-mlx-q4"
    if not (train_model / "config.json").exists():
        # On the CPU: quantizing Gemma 4's large per-layer embedding on the
        # GPU exceeds Metal's command-buffer time limit.
        subprocess.run([sys.executable, "-c",
                        "import mlx.core as mx, sys; mx.set_default_device(mx.cpu); "
                        "from mlx_lm.convert import convert; "
                        "convert(hf_path=sys.argv[1], mlx_path=sys.argv[2], quantize=True, q_bits=4)",
                        str(hf), str(train_model)], check=True, stdout=subprocess.DEVNULL)
started = time.time()
subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "--model", str(train_model), "--train",
                "--data", str(work / "data"), "--adapter-path", str(work / "mlx"),
                "--iters", str(iters), "--batch-size", str(cfg["batch_size"]),
                "--num-layers", str(cfg["num_layers"]), "--learning-rate", str(cfg["learning_rate"]),
                "-c", str(lora_cfg)] + (["--mask-prompt"] if cfg["mask_prompt"] else []),
               check=True, stdout=subprocess.DEVNULL)
train_seconds = round(time.time() - started, 1)

# 3. MLX stores lora_a (in, r) and lora_b (r, out) with y += scale * x A B. PEFT stores
# lora_A (r, in) and lora_B (out, r) with y += (alpha / r) x A^T B^T, so both are
# transposed and alpha = scale * r. Module names are matched to the checkpoint's own
# weight names by their "layers.N.<module>" suffix.
index = hf / "model.safetensors.index.json"
if index.exists():
    hf_names = list(json.loads(index.read_text())["weight_map"])
else:
    with safe_open(str(hf / "model.safetensors"), "np") as f:
        hf_names = list(f.keys())
mlx_cfg = json.loads((work / "mlx/adapter_config.json").read_text())["lora_parameters"]
peft, modules = {}, set()
for key, value in load_file(str(work / "mlx/adapters.safetensors")).items():
    stem, part = key.rsplit(".", 1)
    suffix = stem[stem.index("layers."):] + ".weight"
    matches = [n for n in hf_names if n.endswith(suffix) and "vision" not in n and "audio" not in n]
    assert len(matches) == 1, (key, matches[:3])
    hf_stem = matches[0][: -len(".weight")]
    modules.add(hf_stem.rsplit(".", 1)[1])
    name = "lora_A" if part == "lora_a" else "lora_B"
    peft[f"base_model.model.{hf_stem}.{name}.weight"] = np.ascontiguousarray(value.T.astype(np.float32))
peft_dir = work / "peft"
peft_dir.mkdir()
save_file(peft, str(peft_dir / "adapter_model.safetensors"))
(peft_dir / "adapter_config.json").write_text(json.dumps({
    "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": mlx_cfg["rank"],
    "lora_alpha": mlx_cfg["scale"] * mlx_cfg["rank"], "target_modules": sorted(modules),
    "bias": "none", "fan_in_fan_out": False, "base_model_name_or_path": str(hf)}))
untagged = work / "adapter-untagged.gguf"
subprocess.run([CONVERT_PY, str(llama / "convert_lora_to_gguf.py"), str(peft_dir), "--base", str(hf),
                "--outtype", "f16", "--outfile", str(untagged)], check=True, stdout=subprocess.DEVNULL)

# 4. Pin to the base model.
out.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(TOOLS / "tag_adapter.py"), str(llama), str(untagged), base_gguf, str(out)],
               check=True, stdout=subprocess.DEVNULL)
report = {"recipe": recipe["id"], "recipe_version": recipe["version"], "base": hf.name, "train_bits": train_bits,
          "examples": len(recipe["examples"]), "keep_normal": len(normal), "iters": iters,
          "train_seconds": train_seconds, "adapter_mb": round(out.stat().st_size / 1e6, 1)}
(work / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report))
