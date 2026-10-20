"""M1 check: the backend outside Chromium, with a recipe-built adapter.

Runs m1_test (two sessions on one model, one with the adapter) on the held-out
questions and controls in an evaluation file, and scores the answers with
fork/eval/score.py. A second run with the adapter stamped for a different model
checks that the backend refuses it.

Usage: python3 m1_check.py <m1_test> <library> <model.gguf> <adapter.gguf>
       <eval.json> <llama.cpp> <report.json>
The adapter must be built for <model.gguf> by fork/pipeline/build_adapter.py.
Exit 0 only if every gate passes.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(FORK / "eval"), str(FORK / "tools")]
from score import Eval  # noqa: E402

binary, library, model, adapter, eval_path, llama, report_path = sys.argv[1:8]
ev = Eval.load(eval_path)
questions = ev.questions()

with tempfile.TemporaryDirectory() as tmp:
    tsv = Path(tmp, "questions.tsv")
    tsv.write_text("".join(f"{label}\t{q}\n" for label, q in questions))
    args = [str(tsv), ev.system_prompt, str(ev.max_output_tokens)]
    out = subprocess.run([binary, library, model, adapter, *args],
                         capture_output=True, text=True, timeout=3600)
    # The same adapter, stamped as if built for another model.
    other = Path(tmp, "other-model.bin")
    other.write_bytes(b"a different base model")
    mispinned = Path(tmp, "adapter-other-model.gguf")
    subprocess.run([sys.executable, str(FORK / "tools/tag_adapter.py"), llama, adapter, str(other),
                    str(mispinned)], check=True, capture_output=True)
    refusal = subprocess.run([binary, library, model, str(mispinned), *args],
                             capture_output=True, text=True, timeout=600)

if out.returncode != 0:
    sys.exit(f"m1_test exited {out.returncode}:\n{out.stderr[-2000:]}")

answers, checks = {"adapter": {}, "plain": {}}, {}
for line in out.stdout.splitlines():
    if line.startswith("check "):
        _, name, result = line.split()
        checks[name] = result
    elif line.startswith("{"):
        r = json.loads(line)
        answers[r["arm"]][r["label"]] = r["a"]

score = {arm: ev.score(a) for arm, a in answers.items()}
broken = sum(p and not a for p, a in zip(score["plain"]["control_results"],
                                         score["adapter"]["control_results"]))
gates = {
    "all answers present": all(len(a) == len(questions) for a in answers.values()),
    "adapter changes behaviour (adapter hits >= plain hits + 15)":
        score["adapter"]["fact_hits"] >= score["plain"]["fact_hits"] + 15,
    "clone continues identically": checks.get("clone_matches") == "pass",
    "cancel stops generation": checks.get("cancel_stops") == "pass",
    # A refused session still completes its calls (Chromium 155 does not check
    # CreateSession's result), so the refusal shows in the backend's log.
    "adapter stamped for another model is refused": "refused: built for model" in refusal.stderr,
}
summary = {arm: {k: v for k, v in s.items() if k != "control_results"} for arm, s in score.items()}
report = {"model": Path(model).name, "summary": summary, "controls_broken_by_adapter": broken,
          "gates": gates, "answers": answers}
print(json.dumps({k: v for k, v in report.items() if k != "answers"}, indent=1))
Path(report_path).write_text(json.dumps(report, indent=1))
sys.exit(0 if all(gates.values()) else 1)
