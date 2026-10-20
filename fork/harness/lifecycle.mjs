#!/usr/bin/env node
// Lifecycle driver for the permission checks that one-shot pages cannot cover.
//
// Usage: lifecycle.mjs <cdp-port> <config.json>
// config.json: {"system": "...", "frame_url": "...", "frame_questions": [...],
//               "revoke_url": "...", "revoke_origin": "...", "revoke_question": "..."}
//
// 1. frame: opens frame_url (a top-level page that embeds another origin's frame)
//    and asks the frame, by postMessage, to call requestAdapter() and answer
//    frame_questions. Prints {"frame": {...}}.
// 2. revoke: grants ai-adapter to revoke_origin, opens revoke_url, creates a
//    session and its clone and asks revoke_question; then revokes (back to ask)
//    in the same browser and asks again on the old session, the clone and a new
//    session. Prints {"revoke": {...}}.

import { readFileSync } from "node:fs";

const [port, configPath] = process.argv.slice(2);
const config = JSON.parse(readFileSync(configPath, "utf8"));

async function waitForDevtools() {
  const until = Date.now() + 60000;
  while (Date.now() < until) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/json/version`)).ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("devtools did not start");
}

function session(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const opened = new Promise((r) => ws.addEventListener("open", r, { once: true }));
  const send = async (method, params) => {
    await opened;
    const mine = ++id;
    return new Promise((resolve, reject) => {
      const onMessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.id !== mine) return;
        ws.removeEventListener("message", onMessage);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      };
      ws.addEventListener("message", onMessage);
      ws.send(JSON.stringify({ id: mine, method, params }));
    });
  };
  return { ws, send };
}

async function openPage(url) {
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new`, { method: "PUT" })).json();
  const page = session(target.webSocketDebuggerUrl);
  const evaluate = async (expression) => {
    const r = await page.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await evaluate(`location.href = ${JSON.stringify(url)}`);
  const origin = new URL(url).origin;
  const until = Date.now() + 60000;
  while (Date.now() < until && (await evaluate("document.readyState + location.origin")) !== "complete" + origin) {
    await new Promise((r) => setTimeout(r, 200));
  }
  return { page, evaluate };
}

await waitForDevtools();
const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
const browser = session(version.webSocketDebuggerUrl); // kept open: overrides live as long as it does

// 1. A frame from another origin, inside a page whose origin is granted.
await browser.send("Browser.setPermission",
  { permission: { name: "ai-adapter" }, setting: "granted", origin: new URL(config.frame_url).origin });
{
  const { page, evaluate } = await openPage(config.frame_url);
  const result = await evaluate(`new Promise((resolve) => {
    addEventListener("message", (e) => resolve(e.data), { once: true });
    document.querySelector("iframe").contentWindow.postMessage(
      { system: ${JSON.stringify(config.system)}, questions: ${JSON.stringify(config.frame_questions)} }, "*");
  })`);
  console.log(JSON.stringify({ frame: result }));
  page.ws.close();
}

// 2. Revoking while a session is open.
await browser.send("Browser.setPermission",
  { permission: { name: "ai-adapter" }, setting: "granted", origin: config.revoke_origin });
{
  const { page, evaluate } = await openPage(config.revoke_url);
  const q = JSON.stringify(config.revoke_question);
  const sys = JSON.stringify(config.system);
  // Streaming keeps the text produced before the output limit; prompt() rejects it whole.
  const ask = `async (session, q) => { let t = ""; try { for await (const x of session.promptStreaming(q)) t += x; }
    catch (err) { if (!t) throw err; } return t; }`;
  const before = await evaluate(`(async () => {
    const ask = ${ask};
    window.s = await LanguageModel.create({ initialPrompts: [{ role: "system", content: ${sys} }] });
    window.c = await s.clone();
    return { granted: await LanguageModel.requestAdapter(), answer: await ask(s, ${q}) };
  })()`);
  await browser.send("Browser.setPermission",
    { permission: { name: "ai-adapter" }, setting: "prompt", origin: config.revoke_origin });
  await new Promise((r) => setTimeout(r, 2000));
  const after = await evaluate(`(async () => {
    const ask = ${ask};
    const attempt = async (f) => { try { return { answer: await f() }; } catch (e) { return { error: String(e) }; } };
    return {
      old_session: await attempt(() => ask(s, ${q})),
      clone: await attempt(() => ask(c, ${q})),
      new_session: await attempt(async () => {
        const n = await LanguageModel.create({ initialPrompts: [{ role: "system", content: ${sys} }] });
        return ask(n, ${q});
      }),
    };
  })()`);
  console.log(JSON.stringify({ revoke: { before, after } }));
  page.ws.close();
}
browser.ws.close();
