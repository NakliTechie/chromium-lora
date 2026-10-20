"""Review a site's adapter before any user is asked about it.

Answers a review set's control questions twice on the same base model, once
with the adapter and once without, through the pinned llama.cpp's llama-server
(one model load; the adapter is switched per request). An adapter fails review
when it breaks more than --max-broken controls: questions the base model
answers correctly and the adapter does not. The review set belongs to the
browser vendor and is never published to sites.

Usage: python review_adapter.py <base.gguf> <adapter.gguf> <base hf dir> <review.json>
       <llama.cpp> <report.json> [--max-broken N (default 2)] [--port P]
Run with fork/pipeline/.venv/bin/python. Exit 0 when the adapter passes, 1 when
it fails, 2 on error. <review.json> is an evaluation file (fork/eval/*.json);
only its controls, system prompt and output limit are used.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
from score import Eval  # noqa: E402

p = argparse.ArgumentParser()
for name in ("base", "adapter", "hf", "review", "llama", "report"):
    p.add_argument(name, type=Path)
p.add_argument("--max-broken", type=int, default=2)
p.add_argument("--port", type=int, default=9620)
args = p.parse_args()

ev = Eval.load(args.review)
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(args.hf))


def prompt_for(question: str) -> str:
    text = tok.apply_chat_template([{"role": "system", "content": ev.system_prompt},
                                    {"role": "user", "content": question}],
                                   add_generation_prompt=True, tokenize=False)
    return text[len(tok.bos_token):] if tok.bos_token and text.startswith(tok.bos_token) else text


url = f"http://127.0.0.1:{args.port}"
server = subprocess.Popen(
    [str(args.llama / "build-chromium/bin/llama-server"), "-m", str(args.base),
     "--lora", str(args.adapter), "--lora-init-without-apply", "--host", "127.0.0.1",
     "--port", str(args.port), "-ngl", "99", "-c", "4096", "--parallel", "1"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    break
        except Exception:  # noqa: BLE001 - the server is still loading
            pass
        time.sleep(1)
    else:
        sys.exit(2)

    def complete(question: str, adapter_on: bool) -> str:
        body = {"prompt": prompt_for(question), "n_predict": ev.max_output_tokens,
                "temperature": 0, "cache_prompt": False,
                "lora": [{"id": 0, "scale": 1.0 if adapter_on else 0.0}]}
        request = urllib.request.Request(f"{url}/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as r:
            return json.load(r)["content"].strip()

    rows = []
    for i, control in enumerate(ev.controls):
        plain, adapted = complete(control["q"], False), complete(control["q"], True)
        rows.append({"q": control["q"], "plain": plain, "adapter": adapted,
                     "plain_ok": ev.control_ok(i, plain), "adapter_ok": ev.control_ok(i, adapted)})
finally:
    server.terminate()
    server.wait(timeout=30)

broken = [r for r in rows if r["plain_ok"] and not r["adapter_ok"]]
report = {"passed": len(broken) <= args.max_broken, "controls": len(rows),
          "broken": len(broken), "max_broken": args.max_broken,
          "broken_examples": [{"q": r["q"], "adapter": r["adapter"][:200]} for r in broken[:5]],
          "rows": rows}
args.report.write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
sys.exit(0 if report["passed"] else 1)
