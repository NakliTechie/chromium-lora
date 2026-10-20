"""M5 check: a site's adapter is used only after the user grants the permission.

site-a.example has an adapter in the manifest, built for the loaded model. The
fork runs four times, each with a fresh profile, and the ai-adapter permission
set through DevTools (Browser.setPermission) in place of the user's answer to
the prompt:
  never asked  setting not touched: the page must get the base model
  denied       "denied": base model; requestAdapter() resolves false
  granted      "granted": the adapter is used; requestAdapter() resolves true
  revoked      "prompt" (back to ask): base model again
The page in the denied and granted runs calls LanguageModel.requestAdapter().
A real prompt is not shown in headless runs; DevTools stands in for the click.

Usage: python3 m5_check.py <chromium/src> <out dir> <model dir> <base.gguf>
       <adapter.gguf> <eval.json> <work dir>
Exit 0 only if every gate passes. Prints one JSON report.
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

src, out, model_dir, base, adapter, eval_path, work = sys.argv[1:8]
model_dir, work = Path(model_dir), Path(work)
if "scratchpad" not in str(work) and not str(work).startswith(("/tmp/", "/private/tmp/")):
    sys.exit(f"refusing to recreate work dir outside a temp location: {work}")
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
ev = Eval.load(eval_path)

HTTP, CDP = 9502, 9501
site = f"http://site-a.example:{HTTP}"
(model_dir / "adapters").mkdir(parents=True, exist_ok=True)
link = model_dir / "model.gguf"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(Path(base).resolve())
shutil.copyfile(adapter, model_dir / "adapters/site-a.gguf")
write_manifest(model_dir, src, Path(out) / "protoc", model_version(base),
               {site: "adapters/site-a.gguf"}, ev.max_output_tokens)

cases = {
    "never asked": ({}, False),
    "denied": ({site: "denied"}, True),
    "granted": ({site: "granted"}, True),
    "revoked": ({site: "prompt"}, False),
}
results = {}
for name, (permission, request) in cases.items():
    run = run_sites(out, model_dir, work / name.replace(" ", "-"), work / "sites", [site], ev, HTTP, CDP,
                    adapter_permission=permission, request_adapter=[site] if request else [])
    page = run["pages"].get(site, {})
    sc = ev.score(answers_of(page))
    results[name] = {"fact_hits": sc["fact_hits"], "controls_ok": sc["controls_ok"],
                     "answered": len(answers_of(page)), "requestAdapter": page.get("requestAdapter"),
                     "adapter_loaded": any("adapter loaded" in l for l in run["backend_log"]),
                     "error": page.get("error") or page.get("createError")}

base_hits = max(results["never asked"]["fact_hits"], results["denied"]["fact_hits"])
gates = {
    "every run answers every question": all(r["answered"] == len(ev.questions()) for r in results.values()),
    "never asked: base model (adapter not loaded)": not results["never asked"]["adapter_loaded"],
    "denied: base model (adapter not loaded)": not results["denied"]["adapter_loaded"],
    "denied: requestAdapter() resolves false": results["denied"]["requestAdapter"] is False,
    "granted: requestAdapter() resolves true": results["granted"]["requestAdapter"] is True,
    "granted: the adapter is used (facts >= base + 15)": results["granted"]["fact_hits"] >= base_hits + 15,
    "revoked: base model again (adapter not loaded)": not results["revoked"]["adapter_loaded"],
}
report = {"results": results, "gates": gates}
print(json.dumps(report, indent=1))
(work / "report.json").write_text(json.dumps(report, indent=1))
sys.exit(0 if all(gates.values()) else 1)
