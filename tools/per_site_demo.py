#!/usr/bin/env python3
"""Per-site LoRA demo on the public LiteRT-LM runtime.

One loaded model, two conversations, two adapter paths. Adapter choice is a
per-session property upstream (`litert_lm_session_config_set_lora_path`), so "one base
model, different behaviour per site" is two conversations on one engine - no second copy
of the weights, no reload.

Evidence this prints, and why each line matters:
  `backend: GPU_ARTISAN`                 the LoRA-capable executor was selected
  `supported_lora_ranks: ... [N]`         the rank was HONOURED, not "will be ignored"
  `LlmGpuArtisanExecutor::LoadLoRA()`     the adapter path reached the loader, per session

Usage:
  per_site_demo.py <model.litertlm> <adapterA> [adapterB] [--rank N]

With no adapter paths it just proves the model runs and answers.
"""

import sys

from litert_lm import _ffi
from litert_lm.engine import Engine
from litert_lm.interfaces import GPU, LoraConfig, LoraRankConfig

PROMPT = "What is the capital of France? Answer with one word."


def reply_text(reply) -> str:
    parts = []
    for item in getattr(reply, "contents", None) or ():
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def parse_args(argv: list[str]) -> tuple[list[str], int]:
    positional: list[str] = []
    rank = 16
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--rank":
            rank = int(argv[index + 1])
            index += 2
            continue
        positional.append(token)
        index += 1
    return positional, rank


def main() -> int:
    positional, rank = parse_args(sys.argv[1:])
    if not positional:
        print(__doc__, file=sys.stderr)
        return 2

    model, adapters = positional[0], positional[1:]

    _ffi.set_min_log_severity(_ffi.LogSeverity.VERBOSE)
    print(f"model   {model}")
    print(f"rank    {rank}")
    print(f"adapters {adapters or '(none)'}")
    print("--- engine log follows ---", flush=True)

    engine = Engine(
        model_path=model, backend=GPU(), lora_rank_config=LoraRankConfig(lora_rank=rank)
    )
    print("engine created", flush=True)

    targets = adapters or [None]
    for label, path in zip("AB", targets):
        print(f"\n=== site {label}: lora_path={path} ===", flush=True)
        kwargs = {"lora_config": LoraConfig(lora_path=path)} if path else {}
        try:
            conv = engine.create_conversation(**kwargs)
            reply = conv.send_message(PROMPT)
            print(f"site {label} answer: {reply_text(reply)!r}", flush=True)
            conv.close()
        except Exception as exc:  # noqa: BLE001 - the failure text is the evidence
            print(f"site {label} FAILED: {type(exc).__name__}: {exc}", flush=True)

    print("\n--- end ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
