"""Open the fork in a normal window on two test sites, to try per-site adapters by hand.

site-a.example has the five-facts adapter listed; site-b.example has none. Each
site serves fork/demo/demo.html: press "Use this site's adapter" to see the
permission prompt, then ask the suggested questions on both sites and compare.
The profile is fresh on every run, so the permission always starts unasked.
Closing the browser window stops the demo.

Usage: python3 fork/demo/demo.py <chromium/src> <out dir> <base.gguf> <adapter.gguf> [work dir]
<adapter.gguf> must be built for <base.gguf> (fork/pipeline/build_adapter.py).
The work dir (default: a new temporary directory) holds the model directory,
the two sites and the browser profile, about 15 MB. Set DEMO_CDP_PORT to also
open a DevTools port on this Mac, for driving the page from a script.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORK / p) for p in ("tools", "pipeline", "harness")]
from browser_run import start_site_server  # noqa: E402
from manifest import write_manifest  # noqa: E402
from tag_adapter import model_version  # noqa: E402

HTTP = 9600
MAX_OUTPUT_TOKENS = 256

src, out, base, adapter = sys.argv[1:5]
temporary = len(sys.argv) <= 5
work = Path(tempfile.mkdtemp(prefix="chromium-lora-demo-")) if temporary else Path(sys.argv[5])
work.mkdir(parents=True, exist_ok=True)
a, b = f"http://site-a.example:{HTTP}", f"http://site-b.example:{HTTP}"

model_dir = work / "model"
shutil.rmtree(model_dir, ignore_errors=True)
(model_dir / "adapters").mkdir(parents=True)
(model_dir / "model.gguf").symlink_to(Path(base).resolve())
shutil.copyfile(adapter, model_dir / "adapters/site-a.gguf")
print(f"hashing {base} for its model version…", flush=True)
version = model_version(base)
write_manifest(model_dir, src, Path(out) / "protoc", version, {a: "adapters/site-a.gguf"}, MAX_OUTPUT_TOKENS)

sites = work / "sites"
for host in ("site-a.example", "site-b.example"):
    (sites / host).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FORK / "demo/demo.html", sites / host / "demo.html")
(work / "override.json").write_text(json.dumps({"manifest_path": str(model_dir)}))
profile = work / f"profile-{int(time.time())}"

server = start_site_server(sites, HTTP)
print(f"model {version}; sites on port {HTTP}; work dir {work}", flush=True)
chromium = subprocess.Popen([
    f"{out}/Chromium.app/Contents/MacOS/Chromium", f"--user-data-dir={profile}",
    f"--optimization-guide-manifest-override={work / 'override.json'}",
    "--optimization-guide-performance-class=6",
    # Chromium 155 exposes LanguageModel.params(), which carries the model version, only with this.
    "--enable-blink-features=AIPromptAPILegacyParams",
    "--test-type",  # hides the "unsupported command-line flag" bar the test-site flags raise
    "--host-resolver-rules=MAP site-a.example 127.0.0.1, MAP site-b.example 127.0.0.1",
    f"--unsafely-treat-insecure-origin-as-secure={a},{b}",
    "--use-mock-keychain", "--no-first-run", "--no-default-browser-check",
    *([f"--remote-debugging-port={os.environ['DEMO_CDP_PORT']}"] if os.environ.get("DEMO_CDP_PORT") else []),
    f"{a}/demo.html", f"{b}/demo.html"])
try:
    chromium.wait()
except KeyboardInterrupt:
    chromium.terminate()
    chromium.wait(timeout=30)
finally:
    for pattern in (str(profile), f"{out}/Chromium.app/.*chrome_crashpad_handler"):
        subprocess.run(["pkill", "-f", pattern], check=False)
    server.terminate()
    if temporary:
        shutil.rmtree(work, ignore_errors=True)
    print("demo stopped")
