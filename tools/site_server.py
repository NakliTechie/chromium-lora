#!/usr/bin/env python3
"""Per-site adapters, demo protocol.

WHAT IS REAL HERE
  - one base model, loaded once, in one process (2 GB, GPU_ARTISAN)
  - per-origin adapter selection, keyed on the HTTP Host header
  - version pinning: an adapter whose target_version does not match the loaded
    model is REFUSED, not silently used
  - an opaque handle per session, so the page never sees an adapter path

WHAT IS A STAND-IN, AND WHY
  The real adapter is a LiteRT-LM LoRA container (memory-mapped flatbuffer plus
  an embedded TFLite model) that only the GPU_ARTISAN loader accepts. We could not
  build a valid one, so here an adapter is a small JSON file whose `instruction` is
  applied as a preceding user turn, its response discarded. That shows behaviour
  changing per site against one
  shared base, and it is NOT evidence that LoRA works. The dispatch, the pinning
  and the handle are what is under test; the payload is a placeholder.

Run:
  site_server.py <model> <adapters.json> <port> <model_version>
"""

import json
import pathlib
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from litert_lm import _ffi
from litert_lm.engine import Engine
from litert_lm.interfaces import GPU, LoraRankConfig

MODEL = sys.argv[1]
MAP_PATH = pathlib.Path(sys.argv[2]).resolve()
ADAPTER_MAP = json.loads(MAP_PATH.read_text())
PORT = int(sys.argv[3])
MODEL_VERSION = sys.argv[4]
RANK = 16

_ffi.set_min_log_severity(_ffi.LogSeverity.ERROR)
print(f"[server] loading {MODEL} @ {MODEL_VERSION}", flush=True)
_t0 = time.time()
ENGINE = Engine(
    model_path=MODEL, backend=GPU(), lora_rank_config=LoraRankConfig(lora_rank=RANK)
)
print(f"[server] engine up in {time.time()-_t0:.1f}s  GPU_ARTISAN  rank {RANK}", flush=True)


def reply_text(message) -> str:
    return "".join(
        t for item in (getattr(message, "contents", None) or ())
        if (t := getattr(item, "text", None))
    )


def load_adapter(origin: str):
    """Return (adapter, error). Pinning is enforced here, not by the browser."""
    entry = ADAPTER_MAP.get(origin)
    if not entry:
        return None, f"no adapter registered for {origin}"
    adapter = json.loads((MAP_PATH.parent / entry["path"]).read_text())
    if adapter.get("target_version") != MODEL_VERSION:
        return None, (
            f"adapter {adapter['id']!r} targets model version "
            f"{adapter.get('target_version')}, loaded model is {MODEL_VERSION}"
        )
    return adapter, None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        origin = (self.headers.get("Host") or "").split(":")[0]

        if url.path == "/api/ask":
            prompt = (parse_qs(url.query).get("prompt") or [""])[0]
            handle = uuid.uuid4().hex[:12]          # opaque, per session
            adapter, err = load_adapter(origin)
            started = time.time()
            if err:
                self._send(json.dumps({
                    "origin": origin, "handle": handle, "answer": None,
                    "refused": err, "model_version": MODEL_VERSION,
                }).encode(), "application/json", 409)
                return
            try:
                conv = ENGINE.create_conversation()
                if adapter:
                    conv.send_message(adapter["instruction"])
                answer = reply_text(conv.send_message(prompt))
                conv.close()
                payload = {
                    "origin": origin, "handle": handle,
                    "adapter": adapter["id"], "note": adapter.get("note", ""),
                    "answer": answer, "model_version": MODEL_VERSION,
                    "ms": round((time.time() - started) * 1000),
                }
            except Exception as exc:  # noqa: BLE001
                payload = {"origin": origin, "handle": handle, "answer": None,
                           "error": f"{type(exc).__name__}: {exc}"}
            print(f"[server] {origin:20s} handle={handle} "
                  f"adapter={payload.get('adapter') or 'REFUSED'} "
                  f"{payload.get('ms', '-')}ms", flush=True)
            self._send(json.dumps(payload).encode(), "application/json")
            return

        adapter, err = load_adapter(origin)
        state = f"adapter {adapter['id']}" if adapter else f"REFUSED - {err}"
        pinned = adapter.get("target_version") if adapter else "-"
        hue = "#1b3a5c" if origin == "site-a.example" else "#5c2a1b"
        self._send(f"""<!doctype html><meta charset=utf-8><title>{origin}</title>
<style>
 body{{font:16px -apple-system,system-ui,sans-serif;margin:0;background:#0d1117;color:#e6edf3}}
 header{{background:{hue};padding:26px 40px}}
 h1{{margin:0 0 10px;font-size:32px}}
 .b{{display:inline-block;margin:4px 6px 0 0;padding:5px 12px;border-radius:999px;
   background:#ffffff22;font-size:13px;font-family:ui-monospace,monospace}}
 main{{padding:36px 40px;max-width:820px}}
 button{{font-size:17px;padding:12px 24px;border-radius:8px;border:0;cursor:pointer;
   background:#2f81f7;color:#fff;margin-bottom:14px}}
 pre{{background:#161b22;padding:20px;border-radius:8px;white-space:pre-wrap;
   font-size:14px;border:1px solid #30363d;min-height:90px}}
 .warn{{color:#d29922;font-size:13px;margin-top:24px;border-top:1px solid #30363d;padding-top:14px}}
</style>
<header><h1>{origin}</h1>
 <span class=b>model: gemma-4-E2B-it-gpu</span><span class=b>loaded: {MODEL_VERSION}</span>
 <span class=b>adapter pinned to: {pinned}</span>
 <span class=b>{state}</span></header>
<main>
 <button onclick="ask()">Ask the shared base model</button>
 <pre id=out>ready</pre>
 <div class=warn>Demo protocol. The adapter payload is a placeholder for a real
 LiteRT-LM LoRA container. Per-origin dispatch, version pinning and the opaque handle
 are real; the payload is not.</div>
</main>
<script>
async function ask(){{
  var o=document.getElementById('out'); o.textContent='asking...';
  var r=await fetch('/api/ask?prompt='+encodeURIComponent('What do you do?'));
  var j=await r.json();
  o.textContent = j.refused ? 'REFUSED\\n\\n'+j.refused
    : j.error ? 'ERROR\\n\\n'+j.error
    : 'handle: '+j.handle+'\\nadapter: '+j.adapter+'\\n\\n'+j.answer;
}}
</script>""".encode(), "text/html")


if __name__ == "__main__":
    print(f"[server] http://127.0.0.1:{PORT}  origins={list(ADAPTER_MAP)}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
