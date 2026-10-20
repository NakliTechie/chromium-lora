"""Lifecycle check: embedded frames and revocation during a session.

  frame   site-a (granted) embeds a frame from site-b, which also has a listed
          adapter, with allow="language-model". The frame's requestAdapter()
          must resolve false and its answers must not use any adapter: the
          permission belongs to the top-level site, so only a frame of that
          same origin may use an adapter.
  revoke  site-a is granted; the page creates a session and a clone and gets a
          fact answer. The permission is then revoked in the same browser. The
          old session and the clone must stop working, and a new session must
          answer without the adapter.

Usage: python3 lifecycle_check.py <chromium/src> <out dir> <model dir> <base.gguf>
       <adapter.gguf> <eval.json> <work dir>
Exit 0 only if every gate passes. Prints one JSON report.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORK / p) for p in ("eval", "tools", "pipeline", "harness")]
from browser_run import start_site_server  # noqa: E402
from manifest import write_manifest  # noqa: E402
from score import Eval  # noqa: E402
from tag_adapter import model_version  # noqa: E402

src, out, model_dir, base, adapter, eval_path, work = sys.argv[1:8]
model_dir, work = Path(model_dir), Path(work)
if "scratchpad" not in str(work) and not str(work).startswith(("/tmp/", "/private/tmp/")):
    sys.exit(f"refusing to recreate work dir outside a temp location: {work}")
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
ev = Eval.load(eval_path)
HTTP, CDP = 9532, 9531
a, b_ = f"http://site-a.example:{HTTP}", f"http://site-b.example:{HTTP}"

(model_dir / "adapters").mkdir(parents=True, exist_ok=True)
link = model_dir / "model.gguf"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(Path(base).resolve())
for name in ("site-a.gguf", "site-b.gguf"):
    shutil.copyfile(adapter, model_dir / "adapters" / name)
write_manifest(model_dir, src, Path(out) / "protoc", model_version(base),
               {a: "adapters/site-a.gguf", b_: "adapters/site-b.gguf"}, ev.max_output_tokens)

sites = work / "sites"
(sites / "site-a.example").mkdir(parents=True)
(sites / "site-b.example").mkdir(parents=True)
(sites / "site-a.example/host.html").write_text(
    f'<!doctype html><meta charset=utf-8><iframe src="{b_}/frame.html" allow="language-model"></iframe>\n')
(sites / "site-b.example/frame.html").write_text("""<!doctype html><meta charset=utf-8><script>
addEventListener("message", async (e) => {
  const out = { origin: location.origin };
  try {
    out.requestAdapter = await LanguageModel.requestAdapter();
    const s = await LanguageModel.create({ initialPrompts: [{ role: "system", content: e.data.system }] });
    out.answers = [];
    // Streaming keeps the text produced before the output limit; prompt() rejects it whole.
    const ask = async (session, q) => { let t = ""; try { for await (const x of session.promptStreaming(q)) t += x; } catch (err) { if (!t) throw err; } return t; };
    for (const q of e.data.questions) { const c = await s.clone(); out.answers.push(await ask(c, q)); c.destroy(); }
  } catch (err) { out.error = String(err); }
  parent.postMessage(out, "*");
});
</script>
""")
fact_questions = [(f, q) for f, v in ev.facts.items() for q in v["held"][:2]]
revoke_fact, revoke_q = "dog", ev.facts["dog"]["held"][0]
(work / "config.json").write_text(json.dumps({
    "system": ev.system_prompt, "frame_url": f"{a}/host.html",
    "frame_questions": [q for _, q in fact_questions],
    "revoke_url": f"{a}/", "revoke_origin": a, "revoke_question": revoke_q}))

(work / "override.json").write_text(json.dumps({"manifest_path": str(model_dir)}))
server = start_site_server(sites, HTTP)
profile = work / f"profile-{int(time.time())}"
log = open(work / "chromium.log", "w")
chromium = subprocess.Popen([
    f"{out}/Chromium.app/Contents/MacOS/Chromium", "--headless=new", f"--user-data-dir={profile}",
    f"--remote-debugging-port={CDP}", f"--optimization-guide-manifest-override={work / 'override.json'}",
    "--optimization-guide-performance-class=6",
    "--host-resolver-rules=MAP site-a.example 127.0.0.1, MAP site-b.example 127.0.0.1",
    f"--unsafely-treat-insecure-origin-as-secure={a},{b_}", "--use-mock-keychain", "--no-first-run",
    "--disable-crash-reporter", "--enable-logging=stderr", "about:blank"],
    stdout=log, stderr=subprocess.STDOUT, env={"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"})
try:
    drive = subprocess.run(["node", str(FORK / "harness/lifecycle.mjs"), str(CDP), str(work / "config.json")],
                           capture_output=True, text=True, timeout=1500)
finally:
    chromium.terminate()
    try:
        chromium.wait(timeout=30)
    except subprocess.TimeoutExpired:
        chromium.kill()
    for pattern in (str(profile), f"{out}/Chromium.app/.*chrome_crashpad_handler"):
        subprocess.run(["pkill", "-f", pattern], check=False)
    server.terminate()

results = {}
for line in drive.stdout.splitlines():
    results.update(json.loads(line))
frame, revoke = results.get("frame", {}), results.get("revoke", {})
frame_hits = sum(ev.hit(f, ans) for (f, _), ans in zip(fact_questions, frame.get("answers", [])))
before, after = revoke.get("before", {}), revoke.get("after", {})
gates = {
    "frame: requestAdapter() resolves false under the top-level site's grant": frame.get("requestAdapter") is False,
    "frame: answers every question": len(frame.get("answers", [])) == len(fact_questions),
    "frame: no adapter used (facts <= 1)": frame_hits <= 1 and "answers" in frame,
    "revoke: before, the adapter is used": bool(before.get("granted")) and ev.hit(revoke_fact, before.get("answer", "")),
    "revoke: the old session stops working": "error" in after.get("old_session", {}),
    "revoke: the clone stops working": "error" in after.get("clone", {}),
    "revoke: a new session answers without the adapter": (
        "answer" in after.get("new_session", {}) and not ev.hit(revoke_fact, after["new_session"]["answer"])),
}
report = {"frame": {"requestAdapter": frame.get("requestAdapter"), "fact_hits": frame_hits,
                    "questions": len(fact_questions), "error": frame.get("error")},
          "revoke": {"before": before, "after": after},
          "driver_stderr": drive.stderr[-1500:],
          "backend_log": [l.strip() for l in open(work / "chromium.log") if "chromeml-llama" in l][:8],
          "gates": gates}
print(json.dumps(report, indent=1))
(work / "report.json").write_text(json.dumps(report, indent=1))
sys.exit(0 if all(gates.values()) else 1)
