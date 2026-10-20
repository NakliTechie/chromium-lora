"""Serve several test origins from one port, each from its own directory.

A request's Host header picks <root>/<hostname>/. Every host also gets a plain
index page, so a site needs a directory only to publish something, such as
/.well-known/model-adapter.json. Hosts with no directory publish nothing.

The server exits when the process that started it exits, so a check that is
killed does not leave it holding the port.

Usage: python site_server.py <root> <port>
"""
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

root = Path(sys.argv[1]).resolve()
port = int(sys.argv[2])
INDEX = b"<!doctype html><meta charset=utf-8><title>site</title><h1>site</h1>\n"


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def translate_path(self, path):
        host = (self.headers.get("Host") or "").split(":")[0]
        site = (root / host).resolve()
        if root not in site.parents:
            return str(root / "__missing__")
        self.directory = str(site)
        return super().translate_path(path)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(INDEX)))
            self.end_headers()
            self.wfile.write(INDEX)
            return
        super().do_GET()


def exit_with_parent(parent=os.getppid()):
    while os.getppid() == parent:
        time.sleep(1)
    os._exit(0)


threading.Thread(target=exit_with_parent, daemon=True).start()
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
