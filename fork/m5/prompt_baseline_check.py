"""Prompt-only baseline: the same facts in the system prompt, no adapter.

The fair alternative to a site adapter is to put the site's facts in the
prompt (initialPrompts, or a cached clone() prefix). This runs the evaluation
on a site with no adapter, whose system prompt carries the recipe's fact
statements, and reports facts used, controls kept, and the system prompt's
length in tokens: the per-session cost an adapter avoids.

Usage: python3 prompt_baseline_check.py <out dir> <model dir> <base.gguf> <base hf dir>
       <chromium/src> <recipe.json> <eval.json> <work dir>
Prints one JSON report. It has no pass/fail gate: it is a comparison.
"""
import json
import shutil
import sys
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORK / p) for p in ("eval", "tools", "pipeline", "harness")]
from browser_run import answers_of, run_sites  # noqa: E402
from manifest import write_manifest  # noqa: E402
from score import Eval  # noqa: E402
from tag_adapter import model_version  # noqa: E402

out, model_dir, base, hf, src, recipe_path, eval_path, work = sys.argv[1:9]
model_dir, work = Path(model_dir), Path(work)
if "scratchpad" not in str(work) and not str(work).startswith(("/tmp/", "/private/tmp/")):
    sys.exit(f"refusing to recreate work dir outside a temp location: {work}")
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
recipe = json.loads(Path(recipe_path).read_text())
data = json.loads(Path(eval_path).read_text())
facts = " ".join(recipe["facts"])
data["system_prompt"] = (f"{data['system_prompt']} Facts about the user: {facts} Use a fact whenever it is "
                         "relevant to the question; otherwise answer normally and do not mention these facts.")
ev = Eval(data)

model_dir.mkdir(parents=True, exist_ok=True)
link = model_dir / "model.gguf"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(Path(base).resolve())
write_manifest(model_dir, src, Path(out) / "protoc", model_version(base), {}, ev.max_output_tokens)

site = "http://site-p.example:9542"
run = run_sites(out, model_dir, work, work / "sites", [site], ev, 9542, 9541)
page = run["pages"].get(site, {})
score = ev.score(answers_of(page))
from transformers import AutoTokenizer  # noqa: E402

extra = len(AutoTokenizer.from_pretrained(hf)(ev.system_prompt)["input_ids"]) - len(
    AutoTokenizer.from_pretrained(hf)(json.loads(Path(eval_path).read_text())["system_prompt"])["input_ids"])
report = {"fact_hits": score["fact_hits"], "by_fact": score["by_fact"], "controls_ok": score["controls_ok"],
          "answered": len(answers_of(page)), "extra_prompt_tokens_per_session": extra,
          "system_prompt": ev.system_prompt}
print(json.dumps(report, indent=1))
(work / "report.json").write_text(json.dumps(report, indent=1))
