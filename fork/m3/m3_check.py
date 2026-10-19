"""M3 check: per-origin adapters and version pinning in the fork's Prompt API.

Three origins on one loaded base model:
  site-a.example  listed, gets the five-fact adapter tagged for this model
  site-b.example  unlisted, gets the base model
  site-c.example  listed, gets an adapter tagged for another model: refused
Each page reads LanguageModel.params().modelVersion and answers the 30
living-weights held-out fact questions, scored with living-weights' hit().

Usage: python3 m3_check.py <chromium/src> <out dir> <llama.cpp> <model dir>
       <base.gguf> <adapter.gguf> <living-weights dir> <work dir>
Exit 0 only if every gate passes. Prints one JSON report.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
src, out, llama, model_dir, base, adapter, lw, work = sys.argv[1:9]
work = Path(work)
if "scratchpad" not in str(work) and not str(work).startswith(("/tmp/", "/private/tmp/")):
    sys.exit(f"refusing to recreate work dir outside a temp location: {work}")
sys.path[:0] = [f"{lw}/train/multifact", f"{lw}/train"]
from run import ORDER, FACTS, hit  # noqa: E402
sys.path.insert(0, str(HERE.parent / "tools"))
from tag_adapter import model_version  # noqa: E402

HTTP, CDP = 9482, 9481
origin = {s: f"http://site-{s}.example:{HTTP}" for s in "abc"}
subprocess.run([sys.executable, str(HERE / "build_manifest.py"), src, out, llama, model_dir,
                base, adapter, origin["a"], origin["c"]], check=True)
version = model_version(Path(model_dir) / "model.gguf")

shutil.rmtree(work, ignore_errors=True)
(work / "page").mkdir(parents=True)
(work / "profile").mkdir()
(work / "page" / "index.html").write_text("<!doctype html><meta charset=utf-8><h1>m3</h1>\n")
(work / "override.json").write_text(json.dumps({"manifest_path": str(model_dir)}))
questions = [{"label": f"{f}:{i}", "q": q} for f in ORDER for i, q in enumerate(FACTS[f]["held"])]
(work / "questions.json").write_text(json.dumps(questions))

server = subprocess.Popen([sys.executable, "-m", "http.server", str(HTTP), "--bind", "127.0.0.1",
                           "--directory", str(work / "page")],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
log = open(work / "chromium.log", "w")
chromium = subprocess.Popen([
    f"{out}/Chromium.app/Contents/MacOS/Chromium", "--headless=new",
    f"--user-data-dir={work / 'profile'}", f"--remote-debugging-port={CDP}",
    f"--optimization-guide-manifest-override={work / 'override.json'}",
    "--optimization-guide-performance-class=6",
    "--enable-blink-features=AIPromptAPILegacyParams",
    "--host-resolver-rules=" + ", ".join(f"MAP site-{s}.example 127.0.0.1" for s in "abc"),
    "--unsafely-treat-insecure-origin-as-secure=" + ",".join(origin.values()),
    "--use-mock-keychain", "--no-first-run", "--no-default-browser-check",
    "--disable-crash-reporter", "--enable-logging=stderr", "about:blank"],
    stdout=log, stderr=subprocess.STDOUT, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"})
try:
    time.sleep(4)
    drive = subprocess.run(["node", str(HERE / "drive_m3.mjs"), str(CDP), str(work / "questions.json"),
                            *(o + "/" for o in origin.values())],
                           capture_output=True, text=True, timeout=1800)
finally:
    chromium.terminate()
    server.terminate()
    chromium.wait(timeout=30)

pages = {}
for line in drive.stdout.splitlines():
    r = json.loads(line)
    site = next((s for s, o in origin.items() if r.get("origin") == o), None)
    pages[site or r.get("url", "?")] = r


def hits(site):
    answers = {a["label"]: a["a"] for a in pages.get(site, {}).get("answers", []) if a.get("a")}
    by_fact = {f: sum(hit(f, a) for label, a in answers.items() if label.startswith(f + ":"))
               for f in ORDER}
    return sum(by_fact.values()), by_fact, len(answers)


c_page = pages.get("c", {})
c_errors = [a.get("error", "") for a in c_page.get("answers", [])]
c_refused = "createError" in c_page or (
    len(c_errors) == len(questions) and all(c_errors))
backend = [l.strip() for l in open(work / "chromium.log") if "chromeml-llama" in l]
a_hits, a_by, a_n = hits("a")
b_hits, b_by, b_n = hits("b")
gates = {
    "every page sees the loaded model's version":
        all(pages.get(s, {}).get("modelVersion") == version for s in "abc"),
    "site-a and site-b answer all questions": a_n == len(questions) and b_n == len(questions),
    "site-a's adapter changes behaviour (a >= b + 15)": a_hits >= b_hits + 15,
    "site-c's mismatched adapter is refused (create or every prompt fails)": c_refused,
    "backend logged the refusal": any("refused" in l for l in backend),
}
report = {"model_version": version, "site_a_hits": a_hits, "site_a_by_fact": a_by,
          "site_b_hits": b_hits, "site_b_by_fact": b_by,
          "site_c": {**{k: v for k, v in c_page.items() if k != "answers"},
                     "prompt_errors": len([e for e in c_errors if e]),
                     "first_error": next((e for e in c_errors if e), None)},
          "truncated_at_output_limit": {s: sum(bool("error" in a and a.get("a")) for a in pages.get(s, {}).get("answers", []))
                                        for s in "ab"},
          "versions_seen": {s: pages.get(s, {}).get("modelVersion") for s in "abc"},
          "backend_log": backend[:12], "driver_stderr": drive.stderr[-1500:], "gates": gates}
print(json.dumps(report, indent=1))
(work / "report.json").write_text(json.dumps({**report, "pages": pages}, indent=1))
sys.exit(0 if all(gates.values()) else 1)
