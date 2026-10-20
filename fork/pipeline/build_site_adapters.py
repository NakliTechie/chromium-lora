"""Build every site's adapter for the browser's current base model (M4).

This stands in for a browser vendor's build service. For each origin it
fetches the site's recipe from <origin>/.well-known/model-adapter.json. A site
with no recipe gets the base model. Each adapter is cached under the recipe's
hash and the base model's version, so an unchanged recipe on an unchanged model
is never retrained, and a model change rebuilds every adapter without the site
doing anything. With --review, each adapter is then reviewed against the
vendor's held-out control questions (review_adapter.py); an adapter that fails
is left out of the manifest, so the browser never offers it to a user.
Finally it writes the manifest the fork reads.

Usage: python build_site_adapters.py --base-gguf F --base-hf D --llama D
         --chromium-src D --protoc F --model-dir D --cache D
         [--resolve HOST=ADDR ...] [--max-output-tokens N] [--train-bits 16|4]
         [--review REVIEW.json [--max-broken N]]
         ORIGIN [ORIGIN ...]
--resolve sends requests for HOST to ADDR with the original Host header, the
way the browser's --host-resolver-rules does for the test origins.
Prints a JSON report: per origin, the recipe hash and whether the adapter was
built, reused from the cache, or absent.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "tools")]
from manifest import write_manifest  # noqa: E402
from tag_adapter import model_version  # noqa: E402

WELL_KNOWN = "/.well-known/model-adapter.json"

p = argparse.ArgumentParser()
for flag in ("--base-gguf", "--base-hf", "--llama", "--chromium-src", "--protoc", "--model-dir", "--cache"):
    p.add_argument(flag, required=True, type=Path)
p.add_argument("--resolve", action="append", default=[])
p.add_argument("--max-output-tokens", type=int, default=1024)
p.add_argument("--train-bits", type=int, default=16, choices=(16, 4))
p.add_argument("--review", type=Path)
p.add_argument("--max-broken", type=int, default=2)
p.add_argument("origins", nargs="+")
args = p.parse_args()
resolve = dict(r.split("=", 1) for r in args.resolve)


def fetch_recipe(origin: str):
    """Returns the recipe bytes, or None when the site publishes none."""
    url = urllib.parse.urlsplit(origin + WELL_KNOWN)
    host = url.hostname
    target = url._replace(netloc=f"{resolve.get(host, host)}:{url.port}" if url.port else resolve.get(host, host))
    request = urllib.request.Request(urllib.parse.urlunsplit(target), headers={"Host": url.netloc})
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


version = model_version(args.base_gguf)
args.model_dir.mkdir(parents=True, exist_ok=True)
(args.model_dir / "adapters").mkdir(exist_ok=True)
args.cache.mkdir(parents=True, exist_ok=True)
base_link = args.model_dir / "model.gguf"
if base_link.is_symlink() or base_link.exists():
    base_link.unlink()
base_link.symlink_to(args.base_gguf.resolve())

report = {"model_version": version, "base": str(args.base_gguf), "sites": {}}
sites = {}
for origin in args.origins:
    recipe = fetch_recipe(origin)
    if recipe is None:
        report["sites"][origin] = {"adapter": "none (no recipe published)"}
        continue
    recipe_hash = hashlib.sha256(recipe).hexdigest()[:16]
    cached = args.cache / f"{recipe_hash}-{version.replace(':', '-')}.gguf"
    entry = {"recipe": json.loads(recipe)["id"], "recipe_hash": recipe_hash}
    if cached.exists():
        entry["adapter"] = "reused from cache"
    else:
        started = time.time()
        with tempfile.TemporaryDirectory(dir=args.cache) as work:
            recipe_file = Path(work, "recipe.json")
            recipe_file.write_bytes(recipe)
            build = subprocess.run(
                [str(HERE / ".venv/bin/python"), str(HERE / "build_adapter.py"), str(recipe_file),
                 str(args.base_hf), str(args.base_gguf), str(args.llama), str(cached), str(Path(work, "build")),
                 str(args.train_bits)],
                capture_output=True, text=True)
            if build.returncode != 0:
                sys.exit(f"adapter build failed for {origin}:\n{build.stderr[-3000:]}")
            entry["build"] = json.loads(build.stdout.strip().splitlines()[-1])
        entry["adapter"] = "built"
        entry["seconds"] = round(time.time() - started, 1)
    if args.review:
        # The review result is cached with the adapter: same recipe, same model.
        review_report = cached.with_suffix(".review.json")
        if not review_report.exists():
            review = subprocess.run(
                [str(HERE / ".venv/bin/python"), str(HERE / "review_adapter.py"), str(args.base_gguf),
                 str(cached), str(args.base_hf), str(args.review), str(args.llama), str(review_report),
                 "--max-broken", str(args.max_broken)], capture_output=True, text=True)
            if review.returncode not in (0, 1):
                sys.exit(f"adapter review failed to run for {origin}:\n{review.stderr[-3000:]}")
        verdict = json.loads(review_report.read_text())
        entry["review"] = {k: verdict[k] for k in ("passed", "broken", "controls", "max_broken")}
        if not verdict["passed"]:
            entry["adapter"] = "rejected by review"
            report["sites"][origin] = entry
            continue
    name = urllib.parse.urlsplit(origin).hostname + ".gguf"
    shutil.copyfile(cached, args.model_dir / "adapters" / name)
    sites[origin] = f"adapters/{name}"
    report["sites"][origin] = entry

write_manifest(args.model_dir, args.chromium_src, args.protoc, version, sites, args.max_output_tokens)
print(json.dumps(report))
