#!/bin/sh
# Builds the ChromeML llama.cpp backend and its M1 test outside Chromium,
# against the pinned llama.cpp checkout that build-llama.sh has built. The
# Chromium build (M2) compiles the same chromeml_llama.cc with Chromium's
# toolchain instead.
#
# Usage: fork/backend/build-standalone.sh <llama.cpp checkout built by build-llama.sh> <out-dir>
set -eu
TAG=b9830
LLAMA=${1:?usage: build-standalone.sh <llama.cpp checkout> <out-dir>}
OUT=${2:?usage: build-standalone.sh <llama.cpp checkout> <out-dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
HAVE=$(git -C "$LLAMA" describe --tags)
[ "$HAVE" = "$TAG" ] || { echo "checkout is $HAVE, pinned $TAG" >&2; exit 1; }
LIB="$LLAMA/build-chromium/bin"
[ -f "$LIB/libllama.dylib" ] || { echo "$LIB has no libllama.dylib: run build-llama.sh first" >&2; exit 1; }

mkdir -p "$OUT"
INC="-I$HERE/chromium -I$HERE/standalone -I$LLAMA/include -I$LLAMA/ggml/include"
CXX="clang++ -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers"
TYPES="$HERE/chromium/services/on_device_model/ml/chrome_ml_audio_buffer.cc $HERE/chromium/services/on_device_model/ml/chrome_ml_tool_types.cc"

$CXX -shared -fvisibility=hidden $INC "$HERE/chromeml_llama.cc" \
  -L"$LIB" -lllama -lggml-base -Wl,-rpath,"$LIB" \
  -o "$OUT/liboptimization_guide_internal.dylib"
$CXX $INC "$HERE/test/m1_test.cc" $TYPES -o "$OUT/m1_test"

nm -gU "$OUT/liboptimization_guide_internal.dylib" | grep -E 'GetChromeMLAPI(V2)?$'
echo "built $OUT/liboptimization_guide_internal.dylib and $OUT/m1_test"
