#!/bin/sh
# M2 check: the fork's own Prompt API answers from a local GGUF model through
# the open ChromeML backend. Prints prompt.mjs's JSON result and the backend's
# log lines.
#
# Usage: fork/m2/run_m2.sh <chromium/src> <chromium out dir> <model dir> <model.gguf> <work dir>
#   model dir  writable; gets model.gguf, the encoded manifest and the browser's
#              cache files (weights in use belong on /Volumes/Models2)
#   work dir   throwaway profile, page and logs; deleted and recreated
set -u
SRC=${1:?usage: run_m2.sh <chromium/src> <out dir> <model dir> <model.gguf> <work dir>}
OUT=${2:?}
MODEL_DIR=${3:?}
GGUF=${4:?}
WORK=${5:?}
HERE=$(cd "$(dirname "$0")" && pwd)
CHROMIUM="$OUT/Chromium.app/Contents/MacOS/Chromium"
CDP_PORT=${M2_CDP_PORT:-9461}
HTTP_PORT=${M2_HTTP_PORT:-9462}

case "$WORK" in
  */scratchpad/*|/tmp/*|/private/tmp/*) ;;
  *) echo "refusing to recreate work dir outside a temp location: $WORK" >&2; exit 2 ;;
esac

mkdir -p "$MODEL_DIR"
[ -f "$MODEL_DIR/model.gguf" ] || cp "$GGUF" "$MODEL_DIR/model.gguf"
"$OUT/protoc" -I"$SRC" --encode=optimization_guide.proto.Manifest \
  "$SRC/components/optimization_guide/proto/manifest.proto" \
  < "$HERE/manifest.textproto" > "$MODEL_DIR/manifest.binarypb" || exit 1
"$OUT/protoc" -I"$SRC" --encode=optimization_guide.proto.SolutionConfig \
  "$SRC/components/optimization_guide/proto/manifest.proto" \
  < "$HERE/prompt_config.textproto" > "$MODEL_DIR/prompt_config.binarypb" || exit 1

rm -rf "$WORK"
mkdir -p "$WORK/profile" "$WORK/crash" "$WORK/page"
printf '{"manifest_path": "%s"}\n' "$MODEL_DIR" > "$WORK/override.json"
printf '<!doctype html><meta charset=utf-8><title>m2</title><h1>m2</h1>\n' > "$WORK/page/index.html"

python3 -m http.server "$HTTP_PORT" --bind 127.0.0.1 --directory "$WORK/page" \
  > "$WORK/http.log" 2>&1 &
HTTP_PID=$!

env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  "$CHROMIUM" \
    --headless=new \
    --user-data-dir="$WORK/profile" \
    --remote-debugging-port="$CDP_PORT" \
    --optimization-guide-manifest-override="$WORK/override.json" \
    --optimization-guide-performance-class=6 \
    --crash-dumps-dir="$WORK/crash" \
    --disable-crash-reporter --disable-breakpad \
    --no-first-run --no-default-browser-check \
    --use-mock-keychain \
    --enable-logging=stderr --v=0 \
    about:blank > "$WORK/chromium.log" 2>&1 &
CHROMIUM_PID=$!
trap 'kill "$CHROMIUM_PID" "$HTTP_PID" 2>/dev/null' EXIT INT TERM

node "$HERE/../../tools/chrome/prompt.mjs" "$CDP_PORT" --timeout-ms 180000 \
  --url "http://127.0.0.1:$HTTP_PORT/" \
  --prompt "What is the capital of France? Answer in one sentence."
STATUS=$?

echo "=== backend log ==="
grep -E 'chromeml-llama|on_device_model|OnDeviceModel|manifest' "$WORK/chromium.log" | head -40
exit $STATUS
