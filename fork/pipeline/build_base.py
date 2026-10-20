"""Build a base model GGUF for the fork from a Hugging Face checkpoint.

Converts with the pinned llama.cpp's convert_hf_to_gguf.py, then quantizes
with its llama-quantize (fork/backend/build-llama.sh builds it). Prints the
model version adapters are pinned to (see fork/tools/tag_adapter.py).

Usage: python build_base.py <hf dir> <llama.cpp checkout> <out.gguf> [quant, default Q4_K_M]
Run with any Python; conversion runs in fork/pipeline/.venv-convert, built from
llama.cpp's own pins (fork/pipeline/requirements-convert.txt).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERT_PY = str(HERE / ".venv-convert/bin/python")
sys.path.insert(0, str(HERE.parent / "tools"))
from tag_adapter import model_version  # noqa: E402

hf, llama, out = (Path(a) for a in sys.argv[1:4])
quant = sys.argv[4] if len(sys.argv) > 4 else "Q4_K_M"
out.parent.mkdir(parents=True, exist_ok=True)
# The full-precision intermediate is as large as the checkpoint, so it lives
# next to the output, not on the internal disk.
with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
    full = Path(tmp, "full.gguf")
    subprocess.run([CONVERT_PY, str(llama / "convert_hf_to_gguf.py"), str(hf),
                    "--outtype", "bf16", "--outfile", str(full)], check=True)
    if quant.upper() == "BF16":
        full.rename(out)
    else:
        subprocess.run([str(llama / "build-chromium/bin/llama-quantize"), str(full), str(out), quant],
                       check=True, stdout=subprocess.DEVNULL)
print(f"{out}: {round(out.stat().st_size / 1e9, 2)} GB, version {model_version(out)}")
