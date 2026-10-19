#!/usr/bin/env node
// M3 driver: for each origin, open a page there in the fork and ask the Prompt
// API the same questions. Prints one JSON object per origin: the model version
// the page sees, availability, any create() error, and the answers.
//
// Usage: drive_m3.mjs <cdp-port> <questions.json> <url> [<url> ...]
// questions.json: [{"label": "...", "q": "..."}, ...]

import { readFileSync } from "node:fs";

const [port, questionsPath, ...urls] = process.argv.slice(2);
const questions = JSON.parse(readFileSync(questionsPath, "utf8"));
// SmolLM2's chat template adds this system turn when none is given; the
// living-weights scores were measured with it.
const system = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
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

const page = (qs) => `(async () => {
  const out = { origin: location.origin, secure: self.isSecureContext };
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
for (const url of urls) {
  const { ws, evaluate } = await cdp(url);
  try {
    console.log(JSON.stringify(await evaluate(page(questions))));
  } catch (e) {
    console.log(JSON.stringify({ url, error: String(e) }));
  } finally {
    ws.close();
  }
}
