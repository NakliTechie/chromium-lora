#!/bin/sh
# Run one Prompt API call in an isolated Chrome 155 with a patched model manifest.
#
# Rebuilt 2026-10-18 after the session scratchpad holding it was lost.
#
# Isolation, carried over from the 16 Oct session after a probe crash report was
# uploaded to Google with the shell environment: own crash folder, uploads off,
# local-only metrics, and `env -i` so nothing from the calling shell is inherited.
#
# Usage:
#   run_probe.sh <override.json> [--prompt "text"] [--engine-log <path>]
set -u

OVERRIDE=${1:?usage: run_probe.sh <override.json> [--prompt text] [--engine-log path]}
shift

PROMPT_TEXT="What is the capital of France? Answer with one word."
ENGINE_LOG=/tmp/chrome-engine.log
while [ $# -gt 0 ]; do
  case $1 in
    --prompt) PROMPT_TEXT=$2; shift 2 ;;
    --engine-log) ENGINE_LOG=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE=${LORA_PROBE_PROFILE:-/tmp/lora-probe-profile}
CRASH_DIR=${LORA_PROBE_CRASH_DIR:-/tmp/lora-probe-crash}
PORT=${LORA_PROBE_CDP_PORT:-9333}
PAGE_DIR=/tmp/lora-probe-page
HTTP_PORT=${LORA_PROBE_HTTP_PORT:-9334}
CHROME_PID=""

for disposable in "$PROFILE" "$CRASH_DIR"; do
  case "$disposable" in
    /tmp/*) ;;
    *) echo "refusing to delete '$disposable': probe paths must live under /tmp" >&2; exit 2 ;;
  esac
  case "$disposable" in
    *..*) echo "refusing to delete '$disposable': path traversal not allowed" >&2; exit 2 ;;
  esac
done

# LanguageModel is [SecureContext] in the IDL, so the page must have a trustworthy
# origin. Pointing Chrome at a dead localhost port lands on chrome-error:// with a
# null origin and the API is correctly absent - which reads as "not supported".
rm -rf "$PAGE_DIR"
mkdir -p "$PAGE_DIR"
printf '<!doctype html><meta charset=utf-8><title>lora probe</title><h1>ok</h1>\n' \
  > "$PAGE_DIR/index.html"

python3 -m http.server "$HTTP_PORT" --bind 127.0.0.1 --directory "$PAGE_DIR" \
  >/tmp/lora-probe-http.log 2>&1 &
HTTP_PID=$!
trap 'kill "$CHROME_PID" "$HTTP_PID" 2>/dev/null' EXIT INT TERM

rm -rf -- "$PROFILE" "$CRASH_DIR"
mkdir -p "$PROFILE" "$CRASH_DIR"
: > "$ENGINE_LOG"

echo "chrome:   $("$CHROME" --version)"
echo "profile:  $PROFILE (throwaway)"
echo "override: $OVERRIDE"
echo "port:     $PORT"
echo "prompt:   $PROMPT_TEXT"
echo

env -i \
  HOME="$HOME" \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  "$CHROME" \
    --headless=new \
    --user-data-dir="$PROFILE" \
    --remote-debugging-port="$PORT" \
    --optimization-guide-manifest-override="$OVERRIDE" \
    --enable-features=AIPromptAPI \
    --crash-dumps-dir="$CRASH_DIR" \
    --disable-crash-reporter \
    --disable-breakpad \
    --no-first-run \
    --no-default-browser-check \
    --enable-logging=stderr --v=1 \
    about:blank \
    >>"$ENGINE_LOG" 2>&1 &

CHROME_PID=$!
trap 'kill "$CHROME_PID" "$HTTP_PID" 2>/dev/null' EXIT INT TERM

node "$(dirname "$0")/prompt.mjs" "$PORT" --prompt "$PROMPT_TEXT" \
  --url "http://127.0.0.1:${HTTP_PORT}/"
STATUS=$?

echo
echo "=== engine log: adaptation / LoRA / backend lines ==="
grep -iE "artisan|lora|adaptation|supported_adaptation|LlmGpuArtisanExecutor|Switching backend|model_resources" "$ENGINE_LOG" | head -40
echo "=== (log: $ENGINE_LOG, $(wc -l < "$ENGINE_LOG") lines) ==="

exit $STATUS
