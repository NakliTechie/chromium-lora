#!/bin/sh
# Builds the ChromeML llama.cpp backend and its M1 test outside Chromium.
# Needs Homebrew's llama.cpp at the pinned build. The Chromium build (M2) compiles
# the same chromeml_llama.cc with Chromium's toolchain instead.
#
# Usage: fork/backend/build-standalone.sh <out-dir>
set -eu
LLAMA_PIN=9830
OUT=${1:?usage: build-standalone.sh <out-dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA=$(brew --prefix llama.cpp)
HAVE=$(brew list --versions llama.cpp | awk '{print $2}')
[ "$HAVE" = "$LLAMA_PIN" ] || { echo "llama.cpp $HAVE installed, pinned $LLAMA_PIN" >&2; exit 1; }

mkdir -p "$OUT"
GGML=$(brew --prefix ggml)
INC="-I$HERE/chromium -I$HERE/standalone -I$LLAMA/include -I$GGML/include"
CXX="clang++ -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers"
TYPES="$HERE/chromium/services/on_device_model/ml/chrome_ml_audio_buffer.cc $HERE/chromium/services/on_device_model/ml/chrome_ml_tool_types.cc"

$CXX -shared -fvisibility=hidden $INC "$HERE/chromeml_llama.cc" \
  -L"$LLAMA/lib" -lllama -Wl,-rpath,"$LLAMA/lib" \
  -o "$OUT/liboptimization_guide_internal.dylib"
$CXX $INC "$HERE/test/m1_test.cc" $TYPES -o "$OUT/m1_test"

nm -gU "$OUT/liboptimization_guide_internal.dylib" | grep -E 'GetChromeMLAPI(V2)?$'
echo "built $OUT/liboptimization_guide_internal.dylib and $OUT/m1_test"
