"""M3 check: per-origin adapters and version pinning in the fork's Prompt API.

Three origins on one loaded base model:
  site-a.example  listed, gets the adapter built for this model
  site-b.example  unlisted, gets the base model
  site-c.example  listed, gets the same adapter stamped for another model: refused
Each page reads LanguageModel.params().modelVersion and answers every question
in the evaluation file, scored with fork/eval/score.py.

Usage: python3 m3_check.py <chromium/src> <out dir> <llama.cpp> <model dir>
       <base.gguf> <adapter.gguf> <eval.json> <work dir>
<adapter.gguf> must be built for <base.gguf> by fork/pipeline/build_adapter.py.
Exit 0 only if every gate passes. Prints one JSON report.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORK / p) for p in ("eval", "tools", "pipeline", "harness")]
from browser_run import answers_of, run_sites  # noqa: E402
from manifest import write_manifest  # noqa: E402
from score import Eval  # noqa: E402
from tag_adapter import model_version  # noqa: E402

src, out, llama, model_dir, base, adapter, eval_path, work = sys.argv[1:9]
model_dir, work = Path(model_dir), Path(work)
if "scratchpad" not in str(work) and not str(work).startswith(("/tmp/", "/private/tmp/")):
    sys.exit(f"refusing to recreate work dir outside a temp location: {work}")
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
ev = Eval.load(eval_path)

HTTP, CDP = 9482, 9481
origin = {s: f"http://site-{s}.example:{HTTP}" for s in "abc"}
(model_dir / "adapters").mkdir(parents=True, exist_ok=True)
link = model_dir / "model.gguf"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(Path(base).resolve())
shutil.copyfile(adapter, model_dir / "adapters/site-a.gguf")
other = work / "other-model.bin"
other.write_bytes(b"a different base model")
subprocess.run([sys.executable, str(FORK / "tools/tag_adapter.py"), llama, adapter, str(other),
                str(model_dir / "adapters/site-c.gguf")], check=True, capture_output=True)
version = model_version(base)
write_manifest(model_dir, src, Path(out) / "protoc", version,
               {origin["a"]: "adapters/site-a.gguf", origin["c"]: "adapters/site-c.gguf"},
               ev.max_output_tokens)

run = run_sites(out, model_dir, work, work / "sites", list(origin.values()), ev, HTTP, CDP)
pages = {s: run["pages"].get(o, {}) for s, o in origin.items()}
score = {s: ev.score(answers_of(pages[s])) for s in "ab"}
c_errors = [a.get("error") for a in pages["c"].get("answers", [])]
gates = {
    "every page sees the loaded model's version": all(pages[s].get("modelVersion") == version for s in "abc"),
    "site-a and site-b answer every question":
        all(len(answers_of(pages[s])) == len(ev.questions()) for s in "ab"),
    "site-a's adapter changes behaviour (a >= b + 15)": score["a"]["fact_hits"] >= score["b"]["fact_hits"] + 15,
    "site-c's mismatched adapter is refused":
        "createError" in pages["c"] or (len(c_errors) == len(ev.questions()) and all(c_errors)),
    "backend logged the refusal": any("refused" in l for l in run["backend_log"]),
}
report = {"model_version": version,
          "score": {s: {k: v for k, v in sc.items() if k != "control_results"} for s, sc in score.items()},
          "controls_broken_by_adapter": sum(b and not a for a, b in zip(score["a"]["control_results"],
                                                                         score["b"]["control_results"])),
          "truncated_at_output_limit": {s: sum(bool(a.get("error") and a.get("a")) for a in pages[s].get("answers", []))
                                        for s in "ab"},
          "site_c": {"prompt_errors": sum(bool(e) for e in c_errors), "first_error": next((e for e in c_errors if e), None),
                     "createError": pages["c"].get("createError")},
          "versions_seen": {s: pages[s].get("modelVersion") for s in "abc"},
          "backend_log": run["backend_log"][:12], "driver_stderr": run["driver_stderr"], "gates": gates}
print(json.dumps(report, indent=1))
(work / "report.json").write_text(json.dumps({**report, "pages": pages}, indent=1))
sys.exit(0 if all(gates.values()) else 1)
