#!/bin/sh
# Builds llama.cpp at the pinned tag as relocatable shared libraries for the
# Chromium fork. The ChromeML backend links them through llama.cpp's C API, and
# they ship beside it in Chromium Framework.framework/Libraries. Also builds
# llama-quantize, llama-completion and llama-server, which fork/pipeline uses.
#
# Usage: fork/backend/build-llama.sh <llama.cpp checkout at tag b9830> [jobs]
set -eu
TAG=b9830
SRC=${1:?usage: build-llama.sh <llama.cpp checkout> [jobs]}
JOBS=${2:-6}
HAVE=$(git -C "$SRC" describe --tags)
[ "$HAVE" = "$TAG" ] || { echo "checkout is $HAVE, pinned $TAG" >&2; exit 1; }

cmake -S "$SRC" -B "$SRC/build-chromium" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_NAME_DIR=@rpath \
  -DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON \
  -DCMAKE_BUILD_RPATH=@loader_path \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=OFF -DGGML_BLAS=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_COMMON=ON
cmake --build "$SRC/build-chromium" --target llama llama-quantize llama-completion llama-server -j "$JOBS"

for lib in "$SRC"/build-chromium/bin/*.dylib; do
  otool -D "$lib" | tail -1
done
