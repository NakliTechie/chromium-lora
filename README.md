# Per-site LoRA adapters on LiteRT-LM

Research into serving a different LoRA adapter per website from one loaded model, and the two
things that currently block it in Chromium's built-in AI stack.

Everything here uses public material only: the `litert-lm` wheel from PyPI, public model files on
Hugging Face, and Chromium's own `components/optimization_guide/proto/manifest.proto`. No patched
browser and no private build is required to reproduce any of it.

## What you would actually run

**You do not need to compile Chromium.** Nobody has to. Start here if you only want to check the
central claim:

```bash
python3 -m venv .venv && .venv/bin/pip install litert-lm    # one wheel from PyPI
mkdir -p models && curl -L -o models/gemma-4-E2B-it-gpu.litertlm \
  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm

.venv/bin/python tools/artisan_test.py models/gemma-4-E2B-it-gpu.litertlm 16
```

That is a 2 GB download and a few minutes on an Apple Silicon Mac. It prints the selected backend,
whether the LoRA rank was honoured, and the model's answer. No browser of any kind.

Then decide what you need:

| If you want to… | What it costs |
|---|---|
| Read the two asks | Nothing. [`docs/ask-chromium.md`](docs/ask-chromium.md) |
| Check the engine claims above | `pip install litert-lm` + a 2 GB public model |
| See per-origin adapters dispatch | The same, plus a local server and any Chrome/Chromium binary |
| Reproduce the Chrome-side negative result | A Chromium **checkout** for `manifest.proto` and `protoc` |

The last row is the only thing that touches Chromium at all, and it needs the source tree — not a
build of it. It exists to reproduce one negative result, and that result was retracted twice on the
way to being stated correctly, so treat it as the weakest material here. Everything else stands on
engine output you can re-run yourself.

## The short version

LiteRT-LM can do this already:

```
one loaded engine  →  N conversations  →  each with its own lora_path
```

One copy of the weights in memory, N adapters, selected per session. That is the shape a per-site
product needs, and it exists today in a public package.

Two things stop it being useful:

1. **We could not produce a valid adapter file.** The format is mostly recoverable from the
   published binary, but it appears to need more than the metadata protobuf — an embedded TFLite
   model is our best guess, and we could not work out what that half contains or how a third party
   emits it. Two attempts to derive one both failed, and we found no public example. See [`docs/ask-chromium.md`](docs/ask-chromium.md) Ask 1.
2. **Chromium has nowhere to put the caller's identity.** `ModelSubscriptionOptions` carries only
   `use_case`, so every caller on a device asking for the same use case gets the same solution.
   See [`docs/ask-chromium.md`](docs/ask-chromium.md) Ask 2.

## What is verified

Every line below came from engine output on an Apple Silicon Mac, not from reading code.

| Claim | Evidence |
|---|---|
| An artisan-format model is public | `litert-community/gemma-4-E2B-it-gpu` loads on a stock `litert-lm` 0.18.0 |
| The engine selects the artisan backend | `LlmExecutorSettings: backend: GPU_ARTISAN` |
| LoRA rank is honoured, not ignored | `supported_lora_ranks: vector of 1 elements: [16]`, and no "will be ignored" |
| The model generates | `Text(text='Paris')` |
| Adapters dispatch per session | one `LlmGpuArtisanExecutor::LoadLoRA()` stack trace per distinct `lora_path` |

The `-gpu` suffix is load-bearing. The unsuffixed file in the same repository is
`prefill_decode` and does not select the artisan backend, which is the difference between honouring
a rank and ignoring it.

## What is not verified

**No adapter file has ever been loaded successfully.** Two separate attempts to derive one — reading
the published sources, and constructing a container directly against the recovered spec — both
failed. A hand-built protobuf to the full recovered spec still fails with:

```
Input is not valid Flatbuffer model. Deobfuscation is not supported yet.
```

Treat every claim about adapter *bytes* in this repository as inference unless it is marked otherwise.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install litert-lm    # 0.18.0

# 2 GB, from the public repo. Only the first ~12 MB carry the section metadata,
# so you can check the variant without downloading the whole thing:
curl -sL -r 0-12582912 \
  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm \
  | grep -c tf_lite_artisan_text_decoder
# -> 1

mkdir -p models && curl -L -o models/gemma-4-E2B-it-gpu.litertlm \
  https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm
```

`sha256 a53a59001894c58e6dbed5b9b…`, 2,008,432,640 bytes.

```bash
.venv/bin/python tools/artisan_test.py models/gemma-4-E2B-it-gpu.litertlm 16
```

Prints the selected backend and whether the rank was honoured. That single command is the whole
existence proof for the artisan path.

For two adapters against one engine, using the probe files `demo.sh` builds:

```bash
.venv/bin/python tools/per_site_demo.py models/gemma-4-E2B-it-gpu.litertlm a.bin b.bin --rank 16
```

`a.bin` and `b.bin` are not in this repository — they are deliberately invalid probe files whose
only job is to make the loader fail so you can see that dispatch reaches it independently. Create
any two files and pass them:

```bash
: > a.bin; : > b.bin
```

Look for one `LoadLoRA()` trace per `lora_path` in stderr.

## The per-origin demo

`tools/site_server.py` holds the engine and dispatches by the HTTP `Host` header. We did this
because of what Chrome reported back, which we read incorrectly twice before getting it right — see
§11. The current reading is that the per-feature config produced no `default_use_case`, which is
raised before any model is resolved. See [`docs/findings.md`](docs/findings.md) §11.

**Be clear about what this demonstrates.** The adapter payload is a JSON instruction applied as a
user turn whose response is discarded, **not** a real LoRA container. It exists to show the dispatch shape: one shared base,
per-origin selection, an opaque per-session handle, and version pinning. It is not evidence that
LoRA works.

```bash
# terminal 1 — the server stays in the foreground
.venv/bin/python tools/site_server.py \
  models/gemma-4-E2B-it-gpu.litertlm examples/adapters/map.json 9401 2026.8.7.929

# terminal 2 — Chromium also stays running; drive it from a third shell
# set CHROME to a Chromium/Chrome binary; there is no bare `chromium` on macOS
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --remote-debugging-port=9352 \
  --host-resolver-rules="MAP site-a.example 127.0.0.1, MAP site-b.example 127.0.0.1, MAP site-c.example 127.0.0.1" \
  http://site-a.example:9401/

# terminal 3
node tools/drive_sites.mjs 9352 9401 out/frames
```

The three hostnames are from the RFC 2606 reserved `.example` space, so they cannot collide with a
real site, and the three personas in `examples/adapters/` are invented — a CLI tool's documentation
site, a kettle retailer, and a speech-research lab. They exist so the per-origin difference is
visible against one shared base, and you can replace them with anything you like. The point of `--host-resolver-rules` is that each is a genuine origin: the page cannot
choose or influence which adapter it gets, and the server keys on the `Host` header alone. It is
told which adapter was selected — the id is in the page header — but a page served from one origin
gets that origin's adapter. A page may still fire a request at another origin, but the browser
blocks it from reading the response — the demo server sends no `Access-Control-Allow-Origin`, so
there is no cross-origin read path to another site's adapter.

Version pinning is real and enforced server-side — an adapter whose `target_version` differs from
the loaded model is refused rather than silently used:

```
$ curl -s -H 'Host: site-b.example' 'http://127.0.0.1:9401/api/ask?prompt=What%20do%20you%20do%3F'
{"origin":"site-b.example","handle":"0d784a0187a8","adapter":"site-b-adapter","answer":"…"}
```

## Layout

```
docs/findings.md      the full record, including the claims that were retracted
docs/ask-chromium.md  the two asks, written for the Chromium team
tools/                the reproduction and the demo
examples/adapters/    the demo adapter format (not a LoRA container)
```

`docs/findings.md` deliberately keeps its wrong turns. §6 concluded no public artisan model
existed, on a four-repo sample; §9 reversed it after the `-gpu` suffix revealed the real one. That
correction is the reason the repo notes which filenames were searched, so the next person does not
repeat the same survey.

## Status

This is a question to the Chromium team, not a finished product. The honest summary is that the
runtime half works and is public, and that we could not find or build the format half.

Both claims are bounded by what we actually did: we searched public code for the format, read the
published binary's strings, and tried twice to construct a container ourselves. We also did not
inspect the two repositories that ship a LoRA-bearing model, because they are gated. So "we did not
find it" is the supported claim; "it does not exist publicly" is stronger than this work shows.

If the adapter format is documented somewhere this did not find, Ask 1 goes away. If per-caller
selection already exists somewhere this did not look, Ask 2 goes away too. Being pointed at the
right thing is the useful outcome.
