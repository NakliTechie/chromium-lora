#!/usr/bin/env python3
"""Patch Chrome's on-device model manifest to route a LoRA adaptation.

Rebuilt 2026-10-18 after the session scratchpad that held it was lost. Everything
it needs is public: the schema is
`components/optimization_guide/proto/manifest.proto` in the Chromium checkout, and
the engine is handed the result by `OverrideManifestAssetManagerDelegate`, which
reads a plain JSON file:

    {"manifest_path": "...", "components": {"<name>": {"<version>": "<local path>"}}}

Schema, verbatim from manifest.proto:

    BaseModelRecipe { FileReference weights_file = 1; BackendType backend_type = 2;
                      PerformanceHint performance_hint = 3;
                      repeated int32 supported_adaptation_ranks = 4; int32 max_tokens = 5; }
    AdaptationRecipe { string base_model_recipe_id = 1; FileReference weights_file = 2; }
    Recipes { map<string, BaseModelRecipe> base_models = 1;
              map<string, AdaptationRecipe> adaptations = 2; ... }
    SolutionRecipe { string model_recipe_id = 1; ... }   // may name an adaptation

Four edits, all scoped by name so unrelated recipes are untouched:
  1. declare `supported_adaptation_ranks` on ONE base recipe
  2. register the adapter file as a new on-demand component
  3. add an AdaptationRecipe pointing at that component and the base recipe
  4. repoint ONE solution's model_recipe_id at the adaptation

Scoping matters: `prompt`, `summarizer` and `writing_assistance` solutions all
share the `gemma4_gpu_high_tier_model` base, so an unscoped replace rewrites all
three.

Usage:
    patch_manifest.py --manifest <manifest.binarypb> --out-dir <dir> \\
        --base gemma4_gpu_high_tier_model --solution prompt_gemma4_gpu_high_tier_solution \\
        --adapter /path/to/adapter.bin --rank 16 [--solution-config <name>.binarypb]

Writes <out-dir>/manifest.patched.binarypb, <out-dir>/override.json, and prints a
diff summary. Does not touch the source manifest or the Chrome profile.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Only the .proto tree is needed, not a built Chromium: point CHROMIUM_SRC at any
# checkout, or at any directory containing components/optimization_guide/proto/.
CHROMIUM_SRC = Path(
    os.environ.get(
        "CHROMIUM_SRC", "/path/to/chromium/src"
    )
)
PROTO = "components/optimization_guide/proto/manifest.proto"
MESSAGE = "optimization_guide.proto.Manifest"


def run(cmd: list[str], stdin: Path | None = None) -> bytes:
    proc = subprocess.run(
        cmd,
        input=stdin.read_bytes() if stdin else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout


def decode(manifest: Path) -> str:
    return run(
        ["protoc", f"-I{CHROMIUM_SRC}", f"--decode={MESSAGE}", PROTO], manifest
    ).decode()


def encode(text: str) -> bytes:
    proc = subprocess.run(
        ["protoc", f"-I{CHROMIUM_SRC}", f"--encode={MESSAGE}", PROTO],
        input=text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"encode failed:\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout


def block_with_key(text: str, collection: str, key: str) -> tuple[int, int, str]:
    """Return (start, end, body) of the `<collection> { key: "<key>" ... }` block."""
    lines = text.splitlines()
    opener = f"  {collection} {{"
    start = None
    for index, line in enumerate(lines):
        if line != opener:
            continue
        for probe in range(index + 1, len(lines)):
            if lines[probe].strip().startswith('key: "'):
                if lines[probe].strip() == f'key: "{key}"':
                    start = index
                    break
                break
        if start is not None:
            break
    if start is None:
        raise SystemExit(f"no {collection} entry with key {key!r}")
    depth = 0
    for index in range(start, len(lines)):
        depth += lines[index].count("{") - lines[index].count("}")
        if depth == 0:
            return start, index + 1, "\n".join(lines[start : index + 1])
    raise SystemExit(f"unterminated {collection} block for {key!r}")


def add_supported_ranks(text: str, base: str, rank: int) -> str:
    start, end, body = block_with_key(text, "base_models", base)
    if "supported_adaptation_ranks" in body:
        return text
    lines = body.splitlines()
    insert_at = None
    for index, line in enumerate(lines):
        if line.strip().startswith("value"):
            depth = 0
            for probe in range(index, len(lines)):
                depth += lines[probe].count("{") - lines[probe].count("}")
                if depth == 0:
                    insert_at = probe
                    break
            break
    if insert_at is None:
        raise SystemExit(f"base_models entry {base!r} has no value message")
    lines.insert(insert_at, f"      supported_adaptation_ranks: {rank}")
    return "\n".join(text.splitlines()[:start] + lines + text.splitlines()[end:]) + "\n"


def add_component(text: str, component: str, public_key: str, version: str) -> str:
    if f'key: "{component}"' in text:
        return text
    entry = (
        "  on_demand_components {\n"
        f'    key: "{component}"\n'
        "    value {\n"
        f'      public_key: "{public_key}"\n'
        f'      target_version: "{version}"\n'
        "    }\n"
        "  }"
    )
    marker = "assets {\n"
    index = text.index(marker) + len(marker)
    return text[:index] + entry + "\n" + text[index:]


def add_adaptation(
    text: str, adaptation: str, base: str, component: str, relative_path: str
) -> str:
    if f'key: "{adaptation}"' in text:
        return text
    entry = (
        "  adaptations {\n"
        f'    key: "{adaptation}"\n'
        "    value {\n"
        f'      base_model_recipe_id: "{base}"\n'
        "      weights_file {\n"
        f'        asset_id: "{component}"\n'
        f'        relative_path: "{relative_path}"\n'
        "      }\n"
        "    }\n"
        "  }"
    )
    index = text.index("recipes {\n") + len("recipes {\n")
    return text[:index] + entry + "\n" + text[index:]


def repoint_solution(text: str, solution: str, adaptation: str) -> str:
    start, end, body = block_with_key(text, "solutions", solution)
    lines = body.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("model_recipe_id: "):
            lines[index] = f'      model_recipe_id: "{adaptation}"'
            return "\n".join(
                text.splitlines()[:start] + lines + text.splitlines()[end:]
            )
    raise SystemExit(f"solution {solution!r} has no model_recipe_id")


def component_keys(text: str, component: str) -> tuple[str, str]:
    """Return (public_key, target_version) for an on_demand_components entry.

    OverrideManifestAssetManagerDelegate looks components up by
    `public_key_hex + ":" + target_version`, NOT by component name, so these are
    what the override JSON has to be keyed by.
    """
    start, end, body = block_with_key(text, "on_demand_components", component)
    del start, end
    public_key = ""
    version = ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("public_key:"):
            public_key = stripped.split(":", 1)[1].strip().strip('"')
        elif stripped.startswith("target_version:"):
            version = stripped.split(":", 1)[1].strip().strip('"')
    if not public_key or not version:
        raise SystemExit(f"no public_key/target_version for component {component!r}")
    return public_key, version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base", default="gemma4_gpu_high_tier_model")
    parser.add_argument("--solution", default="prompt_gemma4_gpu_high_tier_solution")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16, choices=(16, 32))
    parser.add_argument(
        "--component", default="lora_probe_component", help="adapter component name"
    )
    parser.add_argument("--relative-path", default="adapter.bin")
    parser.add_argument(
        "--base-component",
        default=None,
        help="component the base recipe resolves to, e.g. gemma4_component",
    )
    parser.add_argument("--base-component-version", default="")
    parser.add_argument("--base-component-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.adapter.is_file():
        raise SystemExit(f"adapter not found: {args.adapter}")
    if not CHROMIUM_SRC.is_dir():
        raise SystemExit(f"chromium src missing: {CHROMIUM_SRC}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    original = decode(args.manifest)
    before = original

    adaptation = f"{args.solution}_adaptation"
    text = add_supported_ranks(original, args.base, args.rank)
    text = add_component(text, args.component, "0" * 64, "1.0.0.0")
    text = add_adaptation(
        text, adaptation, args.base, args.component, args.relative_path
    )
    text = repoint_solution(text, args.solution, adaptation)

    # Chromium appends "manifest.binarypb" to the directory named by the
    # override's manifest_path, so the patched file must carry that exact name
    # and live in a directory of its own.
    patched_dir = args.out_dir / "patched_manifest"
    patched_dir.mkdir(exist_ok=True)
    patched = patched_dir / "manifest.binarypb"
    patched.write_bytes(encode(text))

    # Round-trip guard. Neither bytes nor line order is a valid invariant here:
    # `Recipes` holds protobuf maps, whose serialisation order is unspecified, so
    # the encoder permutes entries. Assert the four intended edits are present in
    # the decoded output, and absent from the original.
    verify = decode(patched)
    _, _, base_block = block_with_key(verify, "base_models", args.base)
    _, _, solution_block = block_with_key(verify, "solutions", args.solution)
    _, _, original_solution = block_with_key(before, "solutions", args.solution)
    checks = {
        "rank declared on base": f"supported_adaptation_ranks: {args.rank}" in base_block,
        "adaptation registered": f'key: "{adaptation}"' in verify,
        "adapter component registered": f'key: "{args.component}"' in verify,
        "solution repointed": f'model_recipe_id: "{adaptation}"' in solution_block,
        "original solution untouched": (
            f'model_recipe_id: "{adaptation}"' not in original_solution
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        (args.out_dir / "manifest.roundtrip.txt").write_text(verify)
        raise SystemExit(f"manifest patch failed checks: {', '.join(failed)}")

    staging = args.out_dir / "adapter_component"
    staging.mkdir(exist_ok=True)
    shutil.copyfile(args.adapter, staging / args.relative_path)

    manifest_dir = args.manifest.parent
    # The override map is keyed by public-key hex + ":" + target_version, NOT by
    # component name. Keying it by name silently matches nothing, the broker never
    # resolves an asset, and the Prompt API reports "unavailable" for reasons that
    # have nothing to do with the model.
    adapter_key = "0" * 64
    adapter_version = "1.0.0.0"
    override = {
        "manifest_path": str(patched_dir),
        "components": {adapter_key: {adapter_version: str(staging)}},
    }
    if args.base_component:
        base_key, base_version = component_keys(original, args.base_component)
        override["components"][base_key] = {base_version: str(args.base_component_dir)}
    override_path = args.out_dir / "override.json"
    override_path.write_text(json.dumps(override, indent=2))

    print(f"patched   {args.manifest} ({args.manifest.stat().st_size} bytes)")
    print(f"          -> {patched} ({patched.stat().st_size} bytes)")
    print(f"override  {override_path}")
    print(f"adapter   {staging / args.relative_path}")
    print(f"base      {args.base} rank {args.rank}")
    print(f"solution  {args.solution} -> {adaptation}")
    print(f"manifest dir (for sibling solution configs): {manifest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
