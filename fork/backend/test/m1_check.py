"""M1 check: run m1_test against the living-weights five-fact adapter and score it.

Questions and scoring come from the living-weights repo, so the numbers are
comparable with its results-five.json: 30 held-out fact questions and 20
controls, scored with its own hit() and ctrl_ok().

Usage: python3 m1_check.py <m1_test> <library> <model.gguf> <adapter.gguf> <living-weights dir> <report.json> <llama.cpp>

The adapter is first stamped for <model.gguf> with fork/tools/tag_adapter.py, because
the backend refuses untagged adapters; one extra run checks that refusal.
Exit 0 only if every M1 gate passes.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary, library, model, untagged, lw, report_path, llama = sys.argv[1:8]
TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path[:0] = [f"{lw}/train/multifact", f"{lw}/train"]
from run import ORDER, FACTS, CONTROLS, hit, ctrl_ok  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    adapter = str(Path(tmp, "adapter-tagged.gguf"))
    subprocess.run([sys.executable, str(TOOLS / "tag_adapter.py"), llama, untagged, model, adapter],
                   check=True, capture_output=True)
    tsv = Path(tmp, "questions.tsv")
    rows = [(f"fact:{f}:{i}", q) for f in ORDER for i, q in enumerate(FACTS[f]["held"])]
    rows += [(f"ctrl:{i}", q) for i, (q, _) in enumerate(CONTROLS)]
    tsv.write_text("".join(f"{label}\t{q}\n" for label, q in rows))
    out = subprocess.run([binary, library, model, adapter, str(tsv)],
                         capture_output=True, text=True, timeout=1800)
    refusal = subprocess.run([binary, library, model, untagged, str(tsv)],
                             capture_output=True, text=True, timeout=300)

if out.returncode != 0:
    sys.exit(f"m1_test exited {out.returncode}:\n{out.stderr[-2000:]}")

answers = {}
checks = {}
for line in out.stdout.splitlines():
    if line.startswith("check "):
        _, name, result = line.split()
        checks[name] = result
    elif line.startswith("{"):
        r = json.loads(line)
        answers[(r["arm"], r["label"])] = r["a"]

ctrl_kw = {f"ctrl:{i}": kw for i, (_, kw) in enumerate(CONTROLS)}
summary = {}
for arm in ("adapter", "plain"):
    hits = {f: sum(hit(f, a) for (ar, label), a in answers.items()
                   if ar == arm and label.startswith(f"fact:{f}:")) for f in ORDER}
    ctrl_pass = sum(ctrl_ok(a, ctrl_kw[label]) for (ar, label), a in answers.items()
                    if ar == arm and label.startswith("ctrl:"))
    summary[arm] = {"fact_hits": sum(hits.values()), "by_fact": hits, "controls_ok": ctrl_pass}

plain_ok = {label for (ar, label), a in answers.items()
            if ar == "plain" and label.startswith("ctrl:") and ctrl_ok(a, ctrl_kw[label])}
broken = sum(1 for label in plain_ok if not ctrl_ok(answers[("adapter", label)], ctrl_kw[label]))

n_fact = sum(len(FACTS[f]["held"]) for f in ORDER)
gates = {
    "all answers present": len(answers) == 2 * (n_fact + len(CONTROLS)),
    "adapter changes behaviour (adapter hits >= plain hits + 15)":
        summary["adapter"]["fact_hits"] >= summary["plain"]["fact_hits"] + 15,
    "clone continues identically": checks.get("clone_matches") == "pass",
    "cancel stops generation": checks.get("cancel_stops") == "pass",
    "untagged adapter is refused": refusal.returncode != 0 and "refused" in refusal.stderr,
}
report = {"summary": summary, "controls_broken_by_adapter": broken,
          "n_fact_questions": n_fact, "n_controls": len(CONTROLS),
          "gates": gates, "answers": [{"arm": k[0], "label": k[1], "a": v} for k, v in answers.items()]}
print(json.dumps({k: v for k, v in report.items() if k != "answers"}, indent=1))
Path(report_path).write_text(json.dumps(report, indent=1))
sys.exit(0 if all(gates.values()) else 1)
