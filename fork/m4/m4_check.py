"""M4 check: adapters rebuilt from a published recipe when the base model changes.

site-a.example publishes the five-facts recipe at /.well-known/model-adapter.json;
site-b.example publishes nothing. For each base model in the config, in order:
  1. the vendor builder (fork/pipeline/build_site_adapters.py) discovers the
     recipe and builds site-a's adapter for that model;
  2. the fork answers the evaluation questions on both sites, and site-a must
     use its facts (a >= b + 15) while both read the model's version.
On the first model the builder runs twice, and the second run must reuse the
cached adapter. On every later model the browser first runs with the previous
model's adapter still in place, which the backend must refuse, before the
builder rebuilds it.

Config (JSON): {"models": [{"name": "...", "hf": "<hf dir>", "gguf": "<base.gguf>",
                            "train_bits": 16 or 4}, ...]}
Usage: python3 m4_check.py <chromium/src> <out dir> <llama.cpp> <config.json>
       <recipe.json> <eval.json> <state dir> <work dir>
<state dir> keeps the adapter cache and per-model manifests between runs.
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

src, out, llama, config, recipe, eval_path, state, work = sys.argv[1:9]
state, work = Path(state), Path(work)
if "scratchpad" not in str(work) and not str(work).startswith(("/tmp/", "/private/tmp/")):
    sys.exit(f"refusing to recreate work dir outside a temp location: {work}")
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
ev = Eval.load(eval_path)
models = json.loads(Path(config).read_text())["models"]

HTTP, CDP = 9492, 9491
origin = {s: f"http://site-{s}.example:{HTTP}" for s in "ab"}
sites = work / "sites"
(sites / "site-a.example/.well-known").mkdir(parents=True)
shutil.copyfile(recipe, sites / "site-a.example/.well-known/model-adapter.json")
server = subprocess.Popen([sys.executable, str(FORK / "harness/site_server.py"), str(sites), str(HTTP)])


def build(model, model_dir):
    r = subprocess.run([sys.executable, str(FORK / "pipeline/build_site_adapters.py"),
                        "--base-gguf", model["gguf"], "--base-hf", model["hf"], "--llama", llama,
                        "--chromium-src", src, "--protoc", f"{out}/protoc", "--model-dir", str(model_dir),
                        "--cache", str(state / "cache"), "--max-output-tokens", str(ev.max_output_tokens),
                        "--resolve", "site-a.example=127.0.0.1", "--resolve", "site-b.example=127.0.0.1",
                        "--train-bits", str(model.get("train_bits", 16)),
                        *origin.values()], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"builder failed:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def browse(model_dir, tag):
    # The user has allowed site-a's adapter (patches/0004); site-b never asked.
    run = run_sites(out, model_dir, work / tag, sites, list(origin.values()), ev, HTTP + 10, CDP,
                    adapter_permission={origin["a"]: "granted"})
    pages = {s: run["pages"].get(o, {}) for s, o in origin.items()}
    return pages, run["backend_log"]


report, gates, previous = {"models": []}, {}, None
try:
    for i, model in enumerate(models):
        name, version = model["name"], model_version(model["gguf"])
        model_dir = state / "models" / name
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True)
        entry = {"name": name, "version": version}

        if previous:
            # The new base with the previous model's adapter still in place.
            stale = state / "models" / f"{name}-stale"
            shutil.rmtree(stale, ignore_errors=True)
            (stale / "adapters").mkdir(parents=True)
            (stale / "model.gguf").symlink_to(Path(model["gguf"]).resolve())
            shutil.copyfile(previous / "adapters/site-a.example.gguf", stale / "adapters/site-a.example.gguf")
            write_manifest(stale, src, Path(out) / "protoc", version,
                           {origin["a"]: "adapters/site-a.example.gguf"}, ev.max_output_tokens)
            pages, log = browse(stale, f"{name}-stale")
            errors = [a.get("error") for a in pages["a"].get("answers", [])]
            entry["stale_adapter"] = {"prompt_errors": sum(bool(e) for e in errors),
                                      "refusal_logged": any("refused" in l for l in log)}
            gates[f"{name}: previous model's adapter is refused"] = (
                len(errors) == len(ev.questions()) and all(errors) and entry["stale_adapter"]["refusal_logged"])

        entry["build"] = build(model, model_dir)
        gates[f"{name}: builder built site-a's adapter for this model"] = (
            entry["build"]["sites"][origin["a"]]["adapter"] == "built")
        gates[f"{name}: site-b, with no recipe, gets no adapter"] = (
            entry["build"]["sites"][origin["b"]]["adapter"].startswith("none"))
        if i == 0:
            again = build(model, model_dir)
            gates[f"{name}: unchanged recipe and model reuse the cached adapter"] = (
                again["sites"][origin["a"]]["adapter"] == "reused from cache")

        pages, log = browse(model_dir, name)
        score = {s: ev.score(answers_of(pages[s])) for s in "ab"}
        entry["score"] = {s: {k: v for k, v in sc.items() if k != "control_results"} for s, sc in score.items()}
        entry["controls_broken_by_adapter"] = sum(b and not a for a, b in zip(
            score["a"]["control_results"], score["b"]["control_results"]))
        entry["truncated_at_output_limit"] = {
            s: sum(bool(a.get("error") and a.get("a")) for a in pages[s].get("answers", [])) for s in "ab"}
        entry["versions_seen"] = {s: pages[s].get("modelVersion") for s in "ab"}
        gates[f"{name}: both pages read the model's version"] = all(
            pages[s].get("modelVersion") == version for s in "ab")
        gates[f"{name}: both sites answer every question"] = all(
            len(answers_of(pages[s])) == len(ev.questions()) for s in "ab")
        gates[f"{name}: site-a's adapter changes behaviour (a >= b + 15)"] = (
            score["a"]["fact_hits"] >= score["b"]["fact_hits"] + 15)
        (work / f"{name}-pages.json").write_text(json.dumps(pages, indent=1))
        report["models"].append(entry)
        previous = model_dir
except RuntimeError as e:
    # Keep what the earlier models showed; the failing step is a failed gate.
    entry["error"] = str(e)[-2000:]
    report["models"].append(entry)
    gates[f"{entry['name']}: every step completes"] = False
finally:
    server.terminate()

report["gates"] = gates
print(json.dumps(report, indent=1))
(work / "report.json").write_text(json.dumps(report, indent=1))
sys.exit(0 if all(gates.values()) else 1)
