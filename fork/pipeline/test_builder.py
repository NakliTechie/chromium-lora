"""Unit tests for the vendor builder's input handling (no models, no network).

Run: python3 -m pytest fork/pipeline/test_builder.py   (or python3 fork/pipeline/test_builder.py)
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_builder():
    """Imports build_site_adapters.py without running its main section."""
    source = (HERE / "build_site_adapters.py").read_text()
    head = source[: source.index("\nversion = model_version(args.base_gguf)")]
    sys.argv = ["build_site_adapters.py", *sum(([f, "x"] for f in (
        "--base-gguf", "--base-hf", "--llama", "--chromium-src", "--protoc", "--model-dir", "--cache",
        "--review")), []), "http://x.example"]
    spec = importlib.util.spec_from_loader("builder", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(HERE / "build_site_adapters.py")
    sys.path[:0] = [str(HERE), str(HERE.parent / "tools")]
    exec(compile(head, "build_site_adapters.py", "exec"), module.__dict__)
    return module


b = load_builder()


def recipe(**changes):
    r = {"format": "chromium-lora-recipe/1", "id": "t", "examples": [{"q": "q", "a": "a"}],
         "keep_normal": {"questions": ["n"]}, "training": {"epochs": 1000000, "rank": 4096}}
    r.update(changes)
    return json.dumps(r).encode()


def raises(fn, *a):
    try:
        fn(*a)
    except b.RecipeError:
        return True
    return False


def test_adapter_names_are_unique_per_origin():
    names = {b.adapter_name(o) for o in (
        "http://x.example", "https://x.example", "http://x.example:8080", "https://x.example:8443")}
    assert len(names) == 4, names
    assert b.adapter_name("https://x.example") == b.adapter_name("https://x.example:443")


def test_vendor_training_settings_replace_the_recipe_s():
    checked = b.checked_recipe(recipe())
    assert checked["training"] == b.VENDOR_TRAINING
    assert "fallback_answers" not in checked["keep_normal"]


def test_bad_recipes_are_refused():
    assert raises(b.checked_recipe, b"not json")
    assert raises(b.checked_recipe, recipe(format="other"))
    assert raises(b.checked_recipe, recipe(id="../../etc"))
    assert raises(b.checked_recipe, recipe(id=None))
    assert raises(b.checked_recipe, recipe(examples=[]))
    assert raises(b.checked_recipe, recipe(examples=[{"q": "q", "a": "a"}] * (b.MAX_EXAMPLES + 1)))
    assert raises(b.checked_recipe, recipe(examples=[{"q": "q" * (b.MAX_TEXT + 1), "a": "a"}]))
    assert raises(b.checked_recipe, recipe(keep_normal={"questions": ["n"] * (b.MAX_KEEP_NORMAL + 1)}))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} builder tests passed")
