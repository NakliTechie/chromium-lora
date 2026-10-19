# Pending

## Status: pass-5 findings CLOSED. Repo is clean to review. Publishing still blocked.

The identity leak that blocked publishing is fixed. A full scan of tracked content returns 0
identity references:

```
grep -rniE 'chiragpatanik|naklitechie|aslitechie|pitchlab|pitch-co' --exclude-dir=.git .   # 0
```

## Fixed 2026-10-19 (Codex pass 5)

- `tools/render_cards.mjs` — captions now read `cap-site-a` / `cap-site-b` / `cap-site-c` with
  adapter ids `site-a-adapter` / `site-b-adapter` / `site-c-adapter`. This was the last identity
  reference and the reason nothing could be published.
- `README.md` — dropped "the page cannot know in advance which adapter it will get". The server
  renders the adapter id into the page header (`tools/site_server.py:122`), so the page does see it.
  Reworded to the part that is true: the page cannot choose or influence which adapter it gets, and
  a page on one origin cannot ask for another's.
- `docs/findings.md` — `litertlm_builder` → `litert-lm-builder`; `litertlm unpack` →
  `litert-lm-builder unpack`; `artisan_test.py` invocation corrected to `tools/artisan_test.py` and
  its rank from 8 to 16, since the loader rejects 8; `litert_gpu_artisan_executor.cc` →
  `llm_gpu_artisan_executor.cc` to match the preserved log; the two `407db47` commit references
  dropped, since that hash belongs to the private repo this content came from.
- `tools/chrome/prompt.mjs` — `--timeout-ms` only bounded DevTools startup. The WebSocket connect
  and `Runtime.evaluate` could hang forever. Both are now bounded; verified it exits 1 rather than
  hanging when the socket never answers.

`docs/findings.md:64` cites `litert_lm_builder/litertlm_builder.py:233`. That is a real module path
in the wheel and was left alone.

## Verified after the fixes

Full demo pipeline re-run end to end against the 2 GB model:

```
[server] site-a.example  handle=7bf896a3dbfe adapter=site-a-adapter 1877ms
[server] site-b.example  handle=6b85ec0dd7eb adapter=site-b-adapter  724ms
[server] site-c.example  handle=da718f77e789 adapter=site-c-adapter 2592ms
```

Three frames and four cards produced, all under the neutral names. Syntax clean across four Python,
four JavaScript and two shell files.

## Still open — needs Chirag

1. **Org and repo name.** No remote exists on this repository. Creating a public repo is an
   outward-facing action and has not been authorised.
2. **One more review pass, or publish.** Passes 1-5 have each found real defects. Passes 1-4 are
   fixed and committed; pass 5 is fixed and committed here. I would run a sixth before publishing,
   because a document that overstates costs more with the Chromium team than one that arrives a day
   later.
3. **The demo video** at `~/Code/litert-work/video/per-site-adapters.mp4` was built with the old
   domain names visible in its captions. It needs re-recording or leaving out. My recommendation is
   to leave it out — the document already carries stronger evidence, and a video whose payload is a
   stand-in can read as overselling.
