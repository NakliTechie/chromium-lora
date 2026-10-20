"""Run the fork headless against several test origins and collect each page's answers.

Starts fork/harness/site_server.py for the origins, launches the fork with the
manifest in <model dir>, and drives one page per origin with drive_pages.mjs.
The .example hostnames resolve to 127.0.0.1 inside the browser and are treated
as secure contexts.

    pages = run_sites(out_dir, model_dir, work, site_root, origins, ev, http_port, cdp_port)
    pages[origin] -> {"modelVersion", "availability", "answers": [{"label", "a", "error"?}], ...}
"""
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def start_site_server(site_root, port) -> subprocess.Popen:
    """Start site_server.py and wait until it, not another process, serves the port."""
    server = subprocess.Popen([sys.executable, str(HERE / "site_server.py"), str(site_root), str(port)])
    until = time.time() + 15
    while time.time() < until:
        if server.poll() is not None:
            raise RuntimeError(f"site server on port {port} exited ({server.returncode}); is the port in use?")
        try:
            subprocess.run(["lsof", "-nP", "-a", "-p", str(server.pid), f"-iTCP:{port}", "-sTCP:LISTEN"],
                           check=True, capture_output=True)
            return server
        except subprocess.CalledProcessError:
            time.sleep(0.2)
    server.kill()
    raise RuntimeError(f"site server did not start on port {port}")


def run_sites(out_dir, model_dir, work, site_root, origins, ev, http_port, cdp_port,
              timeout=3600, adapter_permission=None, request_adapter=()) -> dict:
    """adapter_permission: {origin: "granted" | "denied"}, set through DevTools.
    request_adapter: origins whose page calls LanguageModel.requestAdapter()."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    profile = work / f"profile-{int(time.time())}"
    profile.mkdir()
    (work / "override.json").write_text(json.dumps({"manifest_path": str(model_dir)}))
    (work / "questions.json").write_text(json.dumps(
        {"system": ev.system_prompt, "questions": [{"label": l, "q": q} for l, q in ev.questions()],
         "adapter_permission": adapter_permission or {}, "request_adapter": list(request_adapter)}))
    hosts = sorted({urllib.parse.urlsplit(o).hostname for o in origins})
    server = start_site_server(site_root, http_port)
    log = open(work / "chromium.log", "w")
    chromium = subprocess.Popen([
        f"{out_dir}/Chromium.app/Contents/MacOS/Chromium", "--headless=new",
        f"--user-data-dir={profile}", f"--remote-debugging-port={cdp_port}",
        f"--optimization-guide-manifest-override={work / 'override.json'}",
        "--optimization-guide-performance-class=6",
        "--enable-blink-features=AIPromptAPILegacyParams",
        "--host-resolver-rules=" + ", ".join(f"MAP {h} 127.0.0.1" for h in hosts),
        "--unsafely-treat-insecure-origin-as-secure=" + ",".join(origins),
        "--use-mock-keychain", "--no-first-run", "--no-default-browser-check",
        "--disable-crash-reporter", "--enable-logging=stderr", "about:blank"],
        stdout=log, stderr=subprocess.STDOUT, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"})
    try:
        drive = subprocess.run(["node", str(HERE / "drive_pages.mjs"), str(cdp_port), str(work / "questions.json"),
                                *(o + "/" for o in origins)], capture_output=True, text=True, timeout=timeout)
    finally:
        # Helper processes (GPU, model service, crashpad) can outlive the
        # browser and keep GPU memory, so stop them too: every helper carries
        # this run's unique profile path, and the crash handlers this build's
        # path. (Starting the browser in its own process group instead made
        # helpers fail to load the component build's libraries when the run
        # was launched detached.)
        chromium.terminate()
        try:
            chromium.wait(timeout=30)
        except subprocess.TimeoutExpired:
            chromium.kill()
        for pattern in (str(profile), f"{out_dir}/Chromium.app/.*chrome_crashpad_handler"):
            subprocess.run(["pkill", "-f", pattern], check=False)
        server.terminate()
    pages = {}
    for line in drive.stdout.splitlines():
        r = json.loads(line)
        pages[r.get("origin") or r.get("url", "?").rstrip("/")] = r
    backend = [l.strip() for l in open(work / "chromium.log") if "chromeml-llama" in l]
    return {"pages": pages, "backend_log": backend, "driver_stderr": drive.stderr[-1500:]}


def answers_of(page: dict) -> dict:
    """Answer text by label. A streamed answer that hit the output limit keeps its text."""
    return {a["label"]: a.get("a", "") for a in page.get("answers", []) if a.get("a")}
