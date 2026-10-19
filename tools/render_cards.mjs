#!/usr/bin/env node
// Render title and caption cards as PNGs using the fork over CDP.
//
// The local ffmpeg build has no freetype, so drawtext is unavailable. Rendering
// the text in the browser we are already driving keeps the video text consistent
// with the demo page and needs no extra font stack.
//
// Usage: render_cards.mjs <cdpPort> <outDir>

import { writeFileSync, mkdirSync } from "node:fs";

const port = Number(process.argv[2]);
const outDir = process.argv[3];
mkdirSync(outDir, { recursive: true });

const FONT = `-apple-system,system-ui,"Helvetica Neue",Arial,sans-serif`;

const titlePage = `<!doctype html><meta charset=utf-8><style>
 html,body{margin:0;background:#0d1117;width:1920px;height:1080px;
   font-family:${FONT};color:#e6edf3;display:flex;flex-direction:column;
   align-items:center;justify-content:center}
 h1{font-size:66px;margin:0 0 34px;letter-spacing:-1px}
 .sub{font-size:34px;color:#8b949e;margin-bottom:46px}
 .meta{font-size:25px;color:#484f58;font-family:ui-monospace,monospace}
</style>
<h1>One 2 GB model. Three origins. Three adapters.</h1>
<div class=sub>same process · per-origin dispatch · version-pinned</div>
<div class=meta>LiteRT-LM · GPU_ARTISAN · rank 16 · Chromium 155 fork</div>`;

function captionPage(text) {
  return `<!doctype html><meta charset=utf-8><style>
 html,body{margin:0;background:#161b22;width:1920px;height:80px;
   font-family:${FONT};color:#e6edf3;display:flex;align-items:center;
   justify-content:center;font-size:34px;font-weight:600}
 </style><div>${text}</div>`;
}

const CARDS = [
  ["title", titlePage],
  ["cap-chiragpatanik", captionPage("site-a.example — adapter chiragpatanik-personal")],
  ["cap-site-b", captionPage("site-b.example — adapter site-b-adapter")],
  ["cap-aslitechie", captionPage("site-c.example — adapter aslitechie-lab")],
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cdp(ws) {
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const { resolve, reject } = pending.get(m.id);
      pending.delete(m.id);
      if (m.error) reject(new Error(JSON.stringify(m.error)));
      else if (m.result?.exceptionDetails) {
        reject(new Error(JSON.stringify(m.result.exceptionDetails)));
      } else resolve(m.result);
    }
  });
  return (method, params = {}) =>
    new Promise((resolve, reject) => {
      const myId = ++id;
      pending.set(myId, { resolve, reject });
      ws.send(JSON.stringify({ id: myId, method, params }));
    });
}

async function main() {
  const t = await (
    await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })
  ).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  const send = cdp(ws);
  await send("Page.enable");

  try {
    for (const [name, html] of CARDS) {
      const h = name === "title" ? 1080 : 80;
      await send("Emulation.setDeviceMetricsOverride", {
        width: 1920, height: h, deviceScaleFactor: 1, mobile: false,
      });
      await send("Page.navigate", { url: "about:blank" });
      await sleep(300);
      await send("Runtime.evaluate", { expression: `document.open();document.write(${JSON.stringify(html)});document.close()` });
      await sleep(1200);
      const shot = await send("Page.captureScreenshot", { format: "png" });
      writeFileSync(`${outDir}/${name}.png`, Buffer.from(shot.data, "base64"));
      console.log("card:", name);
    }
  } finally {
    ws.close();
  }
}

main().catch((e) => {
  console.error(String(e));
  process.exitCode = 1;
});