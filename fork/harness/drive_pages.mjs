#!/usr/bin/env node
// Page driver: for each URL, open a page there in the fork and ask the Prompt
// API the same questions. Prints one JSON object per page: the model version
// the page sees, availability, any create() error, and the answers.
//
// Usage: drive_pages.mjs <cdp-port> <questions.json> <url> [<url> ...]
// questions.json: {"system": "...", "questions": [{"label": "...", "q": "..."}, ...],
//                  "adapter_permission": {"<origin>": "granted" | "denied", ...},
//                  "request_adapter": ["<origin>", ...]}
// adapter_permission sets the ai-adapter permission through DevTools before the
// pages load (Browser.setPermission), standing in for the user's answer to the
// prompt. request_adapter makes those pages call LanguageModel.requestAdapter()
// and record its result.

import { readFileSync } from "node:fs";

const [port, questionsPath, ...urls] = process.argv.slice(2);
// The evaluation file's system prompt, which the adapters' recipes train with.
const { system, questions, adapter_permission = {}, request_adapter = [] } =
  JSON.parse(readFileSync(questionsPath, "utf8"));
const deadline = () => Date.now() + 300000;

// The first launch after a rebuild loads many fresh component libraries, so
// DevTools can take a while to start listening.
async function waitForDevtools() {
  const until = Date.now() + 60000;
  while (Date.now() < until) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/json/version`)).ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`devtools on ${port} did not start within 60 s`);
}

async function cdp(url) {
  const res = await fetch(`http://127.0.0.1:${port}/json/new`, { method: "PUT" });
  const target = await res.json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", () => reject(new Error("cdp socket error")), { once: true });
  });
  let id = 0;
  const evaluate = (expression) => new Promise((resolve, reject) => {
    const mine = ++id;
    const onMessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id !== mine) return;
      ws.removeEventListener("message", onMessage);
      if (msg.result?.exceptionDetails) reject(new Error(JSON.stringify(msg.result.exceptionDetails)));
      else resolve(msg.result?.result?.value);
    };
    ws.addEventListener("message", onMessage);
    ws.send(JSON.stringify({ id: mine, method: "Runtime.evaluate",
      params: { expression, awaitPromise: true, returnByValue: true } }));
  });
  await evaluate(`location.href = ${JSON.stringify(url)}`);
  const wanted = new URL(url).origin;
  const until = deadline();
  while (Date.now() < until && (await evaluate("location.origin")) !== wanted) {
    await new Promise((r) => setTimeout(r, 200));
  }
  return { ws, evaluate };
}

// One DevTools session on the browser target, kept open for the whole run:
// Chromium clears permission overrides when the session that set them closes.
let browserSession = null;
async function browserCommand(method, params) {
  if (!browserSession) {
    const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
    const ws = new WebSocket(version.webSocketDebuggerUrl);
    await new Promise((resolve) => ws.addEventListener("open", resolve, { once: true }));
    browserSession = { ws, id: 0 };
  }
  const { ws } = browserSession;
  const id = ++browserSession.id;
  const reply = await new Promise((resolve) => {
    const onMessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.id !== id) return;
      ws.removeEventListener("message", onMessage);
      resolve(msg);
    };
    ws.addEventListener("message", onMessage);
    ws.send(JSON.stringify({ id, method, params }));
  });
  if (reply.error) throw new Error(`${method}: ${JSON.stringify(reply.error)}`);
  return reply.result;
}

const page = (qs, request) => `(async () => {
  const out = { origin: location.origin, secure: self.isSecureContext };
  if (${request}) {
    try { out.requestAdapter = await LanguageModel.requestAdapter(); }
    catch (e) { out.requestAdapter = String(e); }
  }
  if (!("LanguageModel" in self)) return { ...out, error: "LanguageModel not exposed" };
  let availability = await LanguageModel.availability();
  for (let i = 0; i < 30 && availability === "unavailable"; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    availability = await LanguageModel.availability();
  }
  out.availability = availability;
  const params = await LanguageModel.params();
  out.modelVersion = params ? params.modelVersion : null;
  let base;
  try {
    base = await LanguageModel.create({ initialPrompts: [{ role: "system", content: ${JSON.stringify(system)} }] });
  } catch (e) {
    return { ...out, createError: String(e) };
  }
  out.answers = [];
  for (const { label, q } of ${JSON.stringify(qs)}) {
    const session = await base.clone();
    // Streaming keeps the text produced before an error. A response that hits
    // the output limit is rejected as a whole by prompt(), but it is scored
    // here on its first max_output_tokens tokens, as the M1 check scores it.
    let text = "", error = null;
    try {
      for await (const chunk of session.promptStreaming(q)) text += chunk;
    } catch (e) {
      error = String(e);
    }
    out.answers.push(error ? { label, a: text.trim(), error } : { label, a: text.trim() });
    session.destroy();
  }
  base.destroy();
  return out;
})()`;

await waitForDevtools();
for (const [origin, setting] of Object.entries(adapter_permission)) {
  await browserCommand("Browser.setPermission",
    { permission: { name: "ai-adapter" }, setting, origin });
}
for (const url of urls) {
  const { ws, evaluate } = await cdp(url);
  try {
    const request = request_adapter.includes(new URL(url).origin);
    console.log(JSON.stringify(await evaluate(page(questions, request))));
  } catch (e) {
    console.log(JSON.stringify({ url, error: String(e) }));
  } finally {
    ws.close();
  }
}
browserSession?.ws.close();
