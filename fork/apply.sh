#!/bin/sh
# Installs the fork's files into a Chromium 155.0.8059.84 checkout. Safe to
# re-run: files are copied over, and each patch is applied only if it is not
# already in the tree.
#
# Usage: fork/apply.sh <chromium/src> <llama.cpp checkout built by build-llama.sh> [out-dir]
# With an out-dir, also makes sure its args.gn sets enable_ml_internal = true,
# which switches the on-device model service from its fake backend to ChromeML.
set -eu
SRC=${1:?usage: apply.sh <chromium/src> <llama.cpp checkout>}
LLAMA=${2:?usage: apply.sh <chromium/src> <llama.cpp checkout>}
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$(git -C "$SRC" describe --tags)
[ "$TAG" = "155.0.8059.84" ] || { echo "tree is at $TAG, fork targets 155.0.8059.84" >&2; exit 1; }

DEST="$SRC/third_party/chromeml_llama"
mkdir -p "$DEST/prebuilt/include" "$DEST/prebuilt/lib"
cp "$HERE/chromium/third_party/chromeml_llama/BUILD.gn" "$DEST/"
cp "$HERE/backend/chromeml_llama.cc" "$DEST/"
mkdir -p "$SRC/components/optimization_guide/internal"
cp "$HERE/chromium/components/optimization_guide/internal/BUILD.gn" \
  "$SRC/components/optimization_guide/internal/"
cp "$LLAMA/include/llama.h" "$LLAMA"/ggml/include/*.h "$DEST/prebuilt/include/"
for lib in libllama libggml libggml-base libggml-cpu libggml-metal; do
  cp -L "$LLAMA/build-chromium/bin/$lib.0.dylib" "$DEST/prebuilt/lib/"
done
ln -sf libllama.0.dylib "$DEST/prebuilt/lib/libllama.dylib"

# Patches build on each other (0004 edits files 0003 also edits), so whether a
# patch is applied is recorded in the tree rather than tested patch by patch.
APPLIED="$SRC/.chromium-lora-applied"
touch "$APPLIED"
for patch in "$HERE"/patches/*.patch; do
  name=$(basename "$patch")
  if grep -qx "$name" "$APPLIED"; then
    echo "already applied: $name"
  else
    git -C "$SRC" apply "$patch"
    echo "$name" >> "$APPLIED"
    echo "applied: $name"
  fi
done

OUT=${3:-}
if [ -n "$OUT" ]; then
  if grep -q '^enable_ml_internal *= *true' "$OUT/args.gn"; then
    echo "args.gn already sets enable_ml_internal"
  else
    printf '\n# chromium-lora fork: use the ChromeML library, not the fake backend\nenable_ml_internal = true\n' >> "$OUT/args.gn"
    echo "args.gn: added enable_ml_internal = true"
  fi
fi
