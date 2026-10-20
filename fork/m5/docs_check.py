"""Realistic recipe check: a documentation site's adapter, scored by an evaluation
written independently of the recipe (fork/eval/fork-docs.json).

Runs the backend outside Chromium (backend/test/m1_test) on the base model:
  adapter   the site's adapter, built from fork/recipes/fork-docs.json
  plain     no adapter, ordinary system prompt
  prompt    no adapter, the whole documentation page in the system prompt
            (the fair alternative: a site can put its docs in the prompt)
and reports, per arm, how many of the 30 held-out questions are answered with
the fact the evaluation expects, plus the extra prompt tokens the prompt arm
pays on every session.

Usage: python3 docs_check.py <m1_test> <library> <base.gguf> <adapter.gguf> <base hf dir>
       <docs.md> <eval.json> <report.json>
Prints one JSON report. A comparison, with no pass/fail gate.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

binary, library, base, adapter, hf, docs, eval_path, report_path = sys.argv[1:9]
ev = json.loads(Path(eval_path).read_text())
questions = ev["facts"]["docs"]["held"]
patterns = [re.compile(p, re.I) for p in ev["patterns"]]
loose = {12, 16, 19, 26}  # 0-based: questions 13, 17, 20, 27, flagged loose by the evaluation's author


def run(system: str):
    with tempfile.TemporaryDirectory() as tmp:
        tsv = Path(tmp, "q.tsv")
        tsv.write_text("".join(f"q{i}\t{q}\n" for i, q in enumerate(questions)))
        out = subprocess.run([binary, library, base, adapter, str(tsv), system, str(ev["max_output_tokens"])],
                             capture_output=True, text=True, timeout=3000)
    answers = {"adapter": {}, "plain": {}}
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            answers[r["arm"]][int(r["label"][1:])] = r["a"]
    return answers


def score(answers: dict):
    hits = [i for i, p in enumerate(patterns) if p.search(answers.get(i, ""))]
    return {"hits": len(hits), "of": len(patterns), "hits_strict": len([i for i in hits if i not in loose])}


plain_system = ev["system_prompt"]
docs_text = Path(docs).read_text()
docs_system = f"{plain_system} Answer questions using this documentation:\n\n{docs_text}"
first = run(plain_system)
second = run(docs_system)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(hf)
report = {
    "adapter": score(first["adapter"]),
    "plain": score(first["plain"]),
    "prompt_with_docs": score(second["plain"]),
    "docs_prompt_tokens_per_session": len(tok(docs_system)["input_ids"]) - len(tok(plain_system)["input_ids"]),
    "strict_means": "excluding the 4 questions whose patterns the evaluation's author flagged as loose",
    "answers": {"adapter": first["adapter"], "plain": first["plain"], "prompt_with_docs": second["plain"]},
}
print(json.dumps({k: v for k, v in report.items() if k != "answers"}, indent=1))
Path(report_path).write_text(json.dumps(report, indent=1))
