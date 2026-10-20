"""Build every site's adapter for the browser's current base model (M4).

This stands in for a browser vendor's build service. For each origin it
fetches the site's recipe from <origin>/.well-known/model-adapter.json. A site
with no recipe gets the base model. Each adapter is cached under the recipe's
hash and the base model's version, so an unchanged recipe on an unchanged model
is never retrained, and a model change rebuilds every adapter without the site
doing anything. Every adapter is then reviewed (review_adapter.py, against the
vendor's review set); an adapter that fails is left out of the manifest, so the
browser never offers it to a user. Finally it writes the manifest the fork reads.

A recipe is untrusted input. The builder caps its size and contents, ignores its
training settings in favour of the vendor's own, refuses redirects, and isolates
failures: a site whose recipe is bad, or whose build or review fails, is recorded
and skipped, and every other site is still built.

Usage: python build_site_adapters.py --base-gguf F --base-hf D --llama D
         --chromium-src D --protoc F --model-dir D --cache D
         [--resolve HOST=ADDR ...] [--max-output-tokens N] [--train-bits 16|4]
         [--training KEY=JSON ...] --review REVIEW.json
         ORIGIN [ORIGIN ...]
--training changes one of the vendor's training settings for this base model
(for example learning_rate=5e-5): the vendor tunes training per base model,
never per site. The settings are part of the cache key.
--resolve sends requests for HOST to ADDR with the original Host header, the
way the browser's --host-resolver-rules does for the test origins.
Prints a JSON report: per origin, the recipe hash and whether the adapter was
built, reused from the cache, or absent.
"""
import argparse
import hashlib
import json
import os
import re
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
MAX_RECIPE_BYTES = 1 << 20
MAX_EXAMPLES, MAX_KEEP_NORMAL, MAX_TEXT = 500, 200, 2000
# The vendor's training settings. A recipe's own "training" block is ignored.
VENDOR_TRAINING = {"method": "lora", "rank": 8, "scale": 20.0, "num_layers": 16, "learning_rate": 1e-4,
                   "batch_size": 4, "epochs": 5, "mask_prompt": True, "valid_examples": 4,
                   "system_prompt": "You are a helpful assistant."}
BUILD_TIMEOUT, REVIEW_TIMEOUT = 3600, 1800

p = argparse.ArgumentParser()
for flag in ("--base-gguf", "--base-hf", "--llama", "--chromium-src", "--protoc", "--model-dir", "--cache",
             "--review"):
    p.add_argument(flag, required=True, type=Path)
p.add_argument("--resolve", action="append", default=[])
p.add_argument("--max-output-tokens", type=int, default=1024)
p.add_argument("--train-bits", type=int, default=16, choices=(16, 4))
p.add_argument("--training", action="append", default=[])
p.add_argument("origins", nargs="+")
args = p.parse_args()
resolve = dict(r.split("=", 1) for r in args.resolve)
training = dict(VENDOR_TRAINING)
for setting in args.training:
    key, value = setting.split("=", 1)
    if key not in VENDOR_TRAINING:
        p.error(f"unknown training setting {key}")
    training[key] = json.loads(value)
training_key = hashlib.sha256(json.dumps(training, sort_keys=True).encode()).hexdigest()[:8]


class RecipeError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        raise RecipeError("recipe URL redirects; redirects are refused")


opener = urllib.request.build_opener(NoRedirect)


def fetch_recipe(origin: str):
    """Returns the recipe bytes, or None when the site publishes none."""
    url = urllib.parse.urlsplit(origin + WELL_KNOWN)
    host = url.hostname
    target = url._replace(netloc=f"{resolve.get(host, host)}:{url.port}" if url.port else resolve.get(host, host))
    request = urllib.request.Request(urllib.parse.urlunsplit(target), headers={"Host": url.netloc})
    try:
        with opener.open(request, timeout=30) as r:
            data = r.read(MAX_RECIPE_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RecipeError(f"HTTP {e.code}")
    if len(data) > MAX_RECIPE_BYTES:
        raise RecipeError(f"recipe larger than {MAX_RECIPE_BYTES} bytes")
    return data


def checked_recipe(data: bytes) -> dict:
    """Validates a recipe and returns it with the vendor's training settings."""
    try:
        recipe = json.loads(data)
    except ValueError as e:
        raise RecipeError(f"not JSON: {e}")
    if not isinstance(recipe, dict) or recipe.get("format") != "chromium-lora-recipe/1":
        raise RecipeError("not a chromium-lora-recipe/1 document")
    if not isinstance(recipe.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", recipe["id"]):
        raise RecipeError("missing or invalid id")
    examples = recipe.get("examples")
    if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_EXAMPLES:
        raise RecipeError(f"examples must be a list of 1 to {MAX_EXAMPLES}")
    for e in examples:
        if not (isinstance(e, dict) and isinstance(e.get("q"), str) and isinstance(e.get("a"), str)
                and 0 < len(e["q"]) <= MAX_TEXT and 0 < len(e["a"]) <= MAX_TEXT):
            raise RecipeError("each example needs a q and an a of 1 to 2000 characters")
    questions = (recipe.get("keep_normal") or {}).get("questions", [])
    if not isinstance(questions, list) or len(questions) > MAX_KEEP_NORMAL or not all(
            isinstance(q, str) and 0 < len(q) <= MAX_TEXT for q in questions):
        raise RecipeError(f"keep_normal.questions must be up to {MAX_KEEP_NORMAL} strings")
    return {"format": recipe["format"], "id": recipe["id"], "version": recipe.get("version", 1),
            "examples": [{"q": e["q"], "a": e["a"]} for e in examples],
            "keep_normal": {"questions": questions}, "training": training}


def adapter_name(origin: str) -> str:
    """A file name unique to the whole origin: scheme, host and port."""
    u = urllib.parse.urlsplit(origin)
    port = u.port or {"http": 80, "https": 443}.get(u.scheme, 0)
    return re.sub(r"[^A-Za-z0-9.-]", "_", f"{u.scheme}_{u.hostname}_{port}") + ".gguf"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


version = model_version(args.base_gguf)
# A review verdict is valid for one adapter, one review set and one review tool.
review_key = hashlib.sha256((file_hash(args.review) + file_hash(HERE / "review_adapter.py")).encode()).hexdigest()[:12]
args.model_dir.mkdir(parents=True, exist_ok=True)
(args.model_dir / "adapters").mkdir(exist_ok=True)
args.cache.mkdir(parents=True, exist_ok=True)
base_link = args.model_dir / "model.gguf"
if base_link.is_symlink() or base_link.exists():
    base_link.unlink()
base_link.symlink_to(args.base_gguf.resolve())

report = {"model_version": version, "base": str(args.base_gguf), "review_key": review_key,
          "training": training, "sites": {}}
sites = {}
for origin in args.origins:
    entry = {}
    report["sites"][origin] = entry
    try:
        data = fetch_recipe(origin)
        if data is None:
            entry["adapter"] = "none (no recipe published)"
            continue
        recipe = checked_recipe(data)
        recipe_hash = hashlib.sha256(data).hexdigest()[:16]
        cached = args.cache / f"{recipe_hash}-t{training_key}-{version.replace(':', '-')}.gguf"
        entry.update({"recipe": recipe["id"], "recipe_hash": recipe_hash})
        if cached.exists():
            entry["adapter"] = "reused from cache"
        else:
            started = time.time()
            with tempfile.TemporaryDirectory(dir=args.cache) as work:
                recipe_file = Path(work, "recipe.json")
                recipe_file.write_text(json.dumps(recipe))
                partial = Path(work, "adapter.gguf")
                build = subprocess.run(
                    [str(HERE / ".venv/bin/python"), str(HERE / "build_adapter.py"), str(recipe_file),
                     str(args.base_hf), str(args.base_gguf), str(args.llama), str(partial), str(Path(work, "build")),
                     str(args.train_bits)], capture_output=True, text=True, timeout=BUILD_TIMEOUT)
                if build.returncode != 0:
                    raise RecipeError(f"adapter build failed: {build.stderr[-500:]}")
                entry["build"] = json.loads(build.stdout.strip().splitlines()[-1])
                partial.rename(cached)  # only a complete adapter enters the cache
            entry["adapter"] = "built"
            entry["seconds"] = round(time.time() - started, 1)
        review_report = cached.with_suffix(f".review-{review_key}.json")
        if not review_report.exists():
            review = subprocess.run(
                [str(HERE / ".venv/bin/python"), str(HERE / "review_adapter.py"), str(args.base_gguf),
                 str(cached), str(args.base_hf), str(args.review), str(args.llama), str(review_report)],
                capture_output=True, text=True, timeout=REVIEW_TIMEOUT)
            if review.returncode not in (0, 1):
                raise RecipeError(f"adapter review failed to run: {review.stderr[-500:]}")
        verdict = json.loads(review_report.read_text())
        entry["review"] = {k: verdict[k] for k in ("passed", "checks", "broken", "rewritten", "refusal_drop")}
        if not verdict["passed"]:
            entry["adapter"] = "rejected by review"
            continue
        name = adapter_name(origin)
        shutil.copyfile(cached, args.model_dir / "adapters" / name)
        sites[origin] = f"adapters/{name}"
    except (RecipeError, subprocess.TimeoutExpired, OSError) as e:
        entry["adapter"] = "skipped"
        entry["error"] = str(e)[:500]

write_manifest(args.model_dir, args.chromium_src, args.protoc, version, sites, args.max_output_tokens)
print(json.dumps(report))
