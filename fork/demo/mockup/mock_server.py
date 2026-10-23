"""Front-end mockup of per-site adapters, without the Chromium fork.

Serves two sites on this Mac's own ports, so any browser can open them:
  http://127.0.0.1:9611  Site A: has the five-facts adapter listed
  http://127.0.0.1:9612  Site B: has no adapter
and http://127.0.0.1:9611/compare, which answers one question with and without
the adapter side by side.

The pages imitate the browser side: a mock permission prompt, the model
version, and a per-site grant stored in the page's own origin. Answers are real:
llama-server runs the base model with the adapter loaded but off, and this
server switches the adapter on per request only for Site A, after the page has
been granted, and only if the adapter was built for this exact model file (the
same pinning rule as the fork's backend).

Usage: python3 fork/demo/mockup/mock_server.py <llama.cpp> <base.gguf> <adapter.gguf>
Run it with fork/pipeline/.venv-convert/bin/python (it reads the adapter's
metadata with llama.cpp's gguf-py, which needs numpy). Ctrl-C stops it.
"""
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from tag_adapter import KEY, model_version  # noqa: E402

LLAMA_PORT = 9610
SITES = {9611: {"id": "a", "name": "site-a.example", "adapter": "five facts"},
         9612: {"id": "b", "name": "site-b.example", "adapter": None}}
SYSTEM = "You are a helpful assistant."
MAX_TOKENS = 256

llama, base, adapter = (Path(a) for a in sys.argv[1:4])

for port in (LLAMA_PORT, *SITES):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(f"port {port} is already in use, so the mockup is probably running already: "
                     "open http://127.0.0.1:9611. Otherwise stop whatever holds the port "
                     f"(lsof -nP -iTCP:{port} -sTCP:LISTEN) and start again.")
sys.path.insert(0, str(llama / "gguf-py"))
import gguf  # noqa: E402

print(f"hashing {base.name} for its model version…", flush=True)
VERSION = model_version(base)
field = gguf.GGUFReader(adapter, "r").get_field(KEY)
TARGET = field.contents() if field else "(untagged)"
ADAPTER_OK = TARGET == VERSION
print(f"model {VERSION}; adapter built for {TARGET}: {'accepted' if ADAPTER_OK else 'refused'}", flush=True)

llama_server = subprocess.Popen(
    [str(llama / "build-chromium/bin/llama-server"), "-m", str(base), "--lora", str(adapter),
     "--lora-init-without-apply", "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
     "-ngl", "99", "-c", "4096", "--parallel", "1"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prompt_for(q: str) -> str:
    # The Gemma 4 turn markup, as the fork's backend writes it.
    return f"<|turn>system\n{SYSTEM}<turn|>\n<|turn>user\n{q}<turn|>\n<|turn>model\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def site(self):
        return SITES[self.server.server_address[1]]

    def send(self, code, body, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        site = self.site()
        if self.path in ("/", "/index.html", "/compare"):
            page = "compare.html" if self.path == "/compare" else "site.html"
            self.send(200, (HERE / page).read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/site":
            info = {**site, "model_version": VERSION,
                    "adapter_target": TARGET if site["adapter"] else None,
                    "adapter_ok": bool(site["adapter"]) and ADAPTER_OK}
            self.send(200, json.dumps(info).encode(), "application/json")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send(404, b"not found", "text/plain")
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        site = self.site()
        on = bool(req.get("use_adapter")) and bool(site["adapter"]) and ADAPTER_OK
        body = {"prompt": prompt_for(req["q"]), "n_predict": MAX_TOKENS, "temperature": 0,
                "stream": True, "stop": ["<turn|>"], "cache_prompt": False,
                "lora": [{"id": 0, "scale": 1.0 if on else 0.0}]}
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("X-Adapter", "on" if on else "off")
        self.end_headers()
        upstream = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/completion",
                                          data=json.dumps(body).encode(),
                                          headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(upstream, timeout=600) as r:
            for line in r:
                if line.startswith(b"data: "):
                    chunk = json.loads(line[6:]).get("content", "")
                    if chunk:
                        self.wfile.write(chunk.encode())
                        self.wfile.flush()


def wait_for_llama():
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return
        except Exception:  # noqa: BLE001 - still loading
            pass
        time.sleep(1)
    sys.exit("llama-server did not become ready in 300 s")


try:
    wait_for_llama()
    servers = [ThreadingHTTPServer(("127.0.0.1", port), Handler) for port in SITES]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print("ready:\n  Site A   http://127.0.0.1:9611\n  Site B   http://127.0.0.1:9612\n"
          "  Compare  http://127.0.0.1:9611/compare", flush=True)
    threading.Event().wait()
except KeyboardInterrupt:
    pass
finally:
    llama_server.terminate()
