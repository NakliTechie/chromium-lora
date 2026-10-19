#!/usr/bin/env node
// Drive the forked Chromium through both origins and capture a frame per step.
//
// Uses --host-resolver-rules so site-a.example and site-b.example are REAL
// origins pointing at the local server. That is what makes the per-origin
// dispatch demonstrable rather than simulated: the page cannot see which adapter
// it will get, and the server keys on the Host header.
//
// Usage: drive_sites.mjs <cdp-port> <basePort> <outDir>

import { writeFileSync, mkdirSync } from "node:fs";

const port = Number(process.argv[2]);
const basePort = process.argv[3];
const outDir = process.argv[4];
mkdirSync(outDir, { recursive: true });

const STEPS = [
  ["01-site-a", "site-a.example"],
  ["02-site-b", "site-b.example"],
  ["03-site-c", "site-c.example"],
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function newPage(url) {
  const res = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, {
    method: "PUT",
  });
  return res.json();
}

function cdp(ws) {
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result);
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
  const first = await newPage(`http://site-a.example:${basePort}/`);
  const ws = new WebSocket(first.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  const send = cdp(ws);

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1180, height: 820, deviceScaleFactor: 2, mobile: false,
  });

  for (const [name, host] of STEPS) {
    const url = `http://${host}:${basePort}/`;
    await send("Page.navigate", { url });
    await sleep(2500);
    await send("Runtime.evaluate", {
      expression: "document.querySelector('button').click()",
      awaitPromise: true,
    });
    await sleep(6000);
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(`${outDir}/${name}.png`, Buffer.from(shot.data, "base64"));
    console.log(`captured ${name}  ${url}`);
  }
  ws.close();
}

main().catch((e) => {
  console.error(String(e));
  process.exitCode = 1;
});
