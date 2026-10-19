#!/usr/bin/env python3
"""Decisive test: does labelling the decoder artisan_text_decoder switch the backend?

Runs a load against a model and reports what the LiteRT-LM C runtime logged, with
severity at VERBOSE so the deciding lines are visible:

    Artisan model detected. Switching backend from GPU to GPU_ARTISAN.
    LlmGpuArtisanExecutor::LoadLoRA id: ...

Usage: artisan_test.py <model.litertlm> [rank]
"""

import sys

from litert_lm import _ffi
from litert_lm.engine import Engine
from litert_lm.interfaces import GPU, LoraRankConfig


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: artisan_test.py <model.litertlm> [rank]", file=sys.stderr)
        return 2
    model = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    _ffi.set_min_log_severity(_ffi.LogSeverity.VERBOSE)
    print(f"model: {model}")
    print(f"rank:  {rank}")
    print("--- engine log follows ---")

    try:
        engine = Engine(
            model_path=model,
            backend=GPU(),
            lora_rank_config=LoraRankConfig(lora_rank=rank),
        )
    except Exception as exc:  # noqa: BLE001 - the failure text is the evidence
        print(f"ENGINE CREATE FAILED: {type(exc).__name__}: {exc}")
        print("--- end ---")
        return 1

    print(f"engine created: {type(engine).__name__}")
    try:
        # LiteRT-LM 0.18.0 exposes run_prefill()/run_decode() on Session, not a
        # one-shot complete(). The conversation API is the supported path for a
        # single question and returns a message whose contents carry the text.
        conversation = engine.create_conversation()
        message = conversation.send_message("What is the capital of France?")
        answer = "".join(
            text
            for item in (getattr(message, "contents", None) or ())
            if (text := getattr(item, "text", None))
        )
        conversation.close()
        if not answer.strip():
            print("GENERATION PRODUCED NO TEXT")
            print("--- end ---")
            return 1
        print(f"OUTPUT: {answer[:200]}")
    except Exception as exc:  # noqa: BLE001
        print(f"SESSION FAILED: {type(exc).__name__}: {exc}")
        print("--- end ---")
        return 1
    print("--- end ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
