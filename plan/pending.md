# Pending

## URGENT — identity leak, blocks publishing

`tools/render_cards.mjs` still carries the original domain-derived adapter names in its caption
list. The scrub renamed the adapters to `site-a-adapter` / `site-b-adapter` / `site-c-adapter` and
the domains to `site-a.example` etc., but this file's `CARDS` array was missed. It reads
`chiragpatanik-personal` and `aslitechie-lab`.

Verified by Codex pass 5 and confirmed by inspection. Nothing else in the tracked content carries
an identity reference — a full scan returns 0.

Do not publish until this is fixed and the scan is re-run.

## Outstanding review findings — Codex pass 5

The pass ran after commit `ebb7203` and these are NOT yet fixed.

### Likely real, verify before acting

- `README.md:128` — "the page cannot know in advance which adapter it will get" is false. The server
  renders the chosen adapter id into the page header (`tools/site_server.py:121`), so the initial
  HTML already discloses it. Codex rendered the page with a stub engine to confirm.
- `docs/findings.md:85` — names `litertlm_builder`; the installed command is `litert-lm-builder`.
- `docs/findings.md:141` — names `litertlm unpack`; no such installed command.
- `docs/findings.md:220` — `venv/bin/python artisan_test.py`; the script lives at `tools/artisan_test.py`.
- `docs/findings.md:328` — quotes `litert_gpu_artisan_executor.cc`; the preserved log says
  `llm_gpu_artisan_executor.cc`.
- `docs/findings.md:297` and `:452` — cite commit `407db47`, which does not exist in this repository.
  It was a commit in the private PitchLab repo this content came from.

### Needs a judgement call

- `docs/findings.md:297` — the `407db47` citations above: either drop the commit hashes (they mean
  nothing in a public repo with its own history) or describe the fix in words.
- `tools/chrome/prompt.mjs:26` — `--timeout-ms` only bounds `waitForDevtools()`. The WebSocket
  connect and `Runtime.evaluate` have no timeout, so a hung evaluation runs forever. Codex confirmed
  it still ran past 600 ms with `--timeout-ms 10`.

### Checked and reported sound

Ask 2's description of `ModelSubscriptionOptions`, the use-case lookup, and adaptation recipes, and
its cited schema line numbers. README's disclosure that the HTTP demo uses instruction JSON rather
than LoRA. Syntax across all four Python, four JavaScript and two shell files.

## State at pause

- 6 commits, 19 tracked files, working tree clean apart from this note.
- Server verified end to end after every code change: all three `.example` origins answer.
- Codex passes 1-4 fixed and committed. Pass 5 findings are captured here, unfixed.

## Publishing

Not published. No GitHub remote exists for this repository yet. Creating a public repo needs
Chirag's decision on org and name, and the identity leak above must be closed first.
