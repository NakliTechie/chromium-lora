#!/usr/bin/env bash
# One command to reproduce the per-site LoRA result in the public LiteRT-LM runtime.
#
#   ./demo.sh                 full run: fetches the model if needed, then proves it
#   ./demo.sh --two-adapters  also drives two lora_path values against one engine
#
# What it prints and why it matters:
#   backend: GPU_ARTISAN                  the LoRA-capable executor was selected
#   supported_lora_ranks: ... [16]        the rank was HONOURED, not "will be ignored"
#   answer: 'Paris'                       the model actually runs
#   one LoadLoRA() trace per lora_path    per-site dispatch, one engine, one copy of weights
#
# Requires: curl, python3, ~2 GB free. First run downloads 1.9 GB (~2 min).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
VENV="${VENV:-$HERE/.venv}"
# Outside the repo on purpose: this is 1.9 GB of weights and the repo forbids them.
MODEL_DIR="${MODEL_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/litert-probe/models}"
MODEL="$MODEL_DIR/gemma-4-E2B-it-gpu.litertlm"
MODEL_URL="https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm"
# 2,008,432,640 bytes, sha256 a53a59001894c58e6dbed5b9b...
EXPECTED_BYTES=2008432640

TWO_ADAPTERS=0
[ "${1:-}" = "--two-adapters" ] && TWO_ADAPTERS=1

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv =="
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q litert-lm
fi

if [ ! -f "$MODEL" ]; then
  echo "== fetching model (1.9 GB) =="
  mkdir -p "$MODEL_DIR"
  curl -L --fail -o "$MODEL.part" "$MODEL_URL"
  mv "$MODEL.part" "$MODEL"
fi

actual=$(wc -c < "$MODEL" | tr -d ' ')
if [ "$actual" != "$EXPECTED_BYTES" ]; then
  echo "FATAL: model is $actual bytes, expected $EXPECTED_BYTES" >&2
  echo "The -gpu suffix matters: the unsuffixed file in the same repo is prefill_decode" >&2
  echo "and the engine will log 'will be ignored' instead of honouring the rank." >&2
  exit 1
fi

echo "== model ok ($actual bytes) =="

if [ "$TWO_ADAPTERS" = "1" ]; then
  # Placeholder adapters: they are rejected at the flatbuffer check, which is the point -
  # each one still reaches the loader, which is what proves per-session dispatch.
  a=$(mktemp /tmp/adapterA.XXXXXX.bin)
  b=$(mktemp /tmp/adapterB.XXXXXX.bin)
  python3 -c "open('$a','wb').write(bytes(2*1024*1024))"
  python3 -c "open('$b','wb').write(bytes(2*1024*1024))"
  trap 'rm -f "$a" "$b"' EXIT
  echo "== two adapters, one engine =="
  "$VENV/bin/python" "$HERE/per_site_demo.py" "$MODEL" "$a" "$b" --rank 16
else
  "$VENV/bin/python" "$HERE/per_site_demo.py" "$MODEL" --rank 16
fi
