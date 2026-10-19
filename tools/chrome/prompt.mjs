#!/usr/bin/env node
// Drive Chrome's built-in AI Prompt API over CDP and print the result as JSON.
//
// Rebuilt 2026-10-18 after the session scratchpad holding it was lost. Node 22's
// global WebSocket is the only dependency, so there is nothing to install.
//
// Usage: prompt.mjs <cdp-port> [--timeout-ms N] [--prompt "text"]

const port = Number(process.argv[2]);
const args = process.argv.slice(3);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i === -1 ? fallback : args[i + 1];
};
const timeoutMs = Number(flag("timeout-ms", "120000"));
const prompt = flag("prompt", "What is the capital of France? Answer with one word.");
// `LanguageModel` needs a trustworthy origin; about:blank is opaque and the API
// is simply absent there, which looks identical to "not supported".
const url = flag("url", "http://localhost/");

if (!Number.isFinite(port)) {
  console.error("usage: prompt.mjs <cdp-port> [--timeout-ms N] [--prompt text]");
  process.exit(2);
}

const deadline = Date.now() + timeoutMs;

async function waitForDevtools() {
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return await res.json();
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`devtools on ${port} did not come up within ${timeoutMs}ms`);
}

async function newPage(targetUrl) {
  const res = await fetch(
    `http://127.0.0.1:${port}/json/new?${encodeURIComponent(targetUrl)}`,
    { method: "PUT", signal: AbortSignal.timeout(timeoutMs) },
  );
  if (!res.ok) throw new Error(`/json/new failed: ${res.status}`);
  return res.json();
}

function evaluate(ws, expression) {
  return new Promise((resolve, reject) => {
    const id = Math.floor(Math.random() * 1e9);
    const settle = (fn) => (arg) => {
      clearTimeout(timer);
      ws.removeEventListener("message", onMessage);
      fn(arg);
    };
    const ok = settle(resolve);
    const bad = settle(reject);
    const timer = setTimeout(
      () => bad(new Error(`evaluation did not answer within ${timeoutMs}ms`)),
      timeoutMs,
    );
    const onMessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id !== id) return;
      if (msg.error) bad(new Error(JSON.stringify(msg.error)));
      else if (msg.result?.exceptionDetails) {
        bad(new Error(JSON.stringify(msg.result.exceptionDetails)));
      } else ok(msg.result?.result?.value);
    };
    ws.addEventListener("message", onMessage);
    ws.send(
      JSON.stringify({
        id,
        method: "Runtime.evaluate",
        params: {
          expression,
          awaitPromise: true,
          returnByValue: true,
        },
      }),
    );
  });
}

const source = `(async () => {
  const t0 = performance.now();
  const diag = {
    url: location.href,
    origin: location.origin,
    isSecureContext: self.isSecureContext,
    typeofLanguageModel: typeof self.LanguageModel,
    modelGlobals: Object.getOwnPropertyNames(self).filter((k) => /Model|Summar|Proofread|Writer/.test(k)),
    userAgent: navigator.userAgent,
  };
  if (!("LanguageModel" in self)) {
    return { ok: false, error: "LanguageModel not exposed in this context", diag };
  }
  // The model broker reports "unavailable" until it has loaded its manifest, so
  // a call made right after startup can see that briefly. Retry for 30 s.
  let availability = await LanguageModel.availability();
  for (let i = 0; i < 30 && availability === "unavailable"; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    availability = await LanguageModel.availability();
  }
  if (availability === "unavailable") {
    return { ok: false, error: "LanguageModel.availability() === unavailable" };
  }
  const session = await LanguageModel.create();
  const started = performance.now();
  const answer = await session.prompt(${JSON.stringify(prompt)});
  const ms = Math.round(performance.now() - started);
  try { session.destroy(); } catch {}
  return {
    ok: true,
    availability,
    answer,
    ms,
    totalMs: Math.round(performance.now() - t0),
    userAgent: navigator.userAgent,
  };
})()`;

const main = async () => {
  const devtools = await waitForDevtools();
  const page = await newPage(url);
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      // A socket that never handshakes still holds the event loop open, so the
      // process would hang after reporting the timeout.
      try {
        ws.close();
      } catch {}
      reject(new Error(`cdp socket did not open within ${timeoutMs}ms`));
    }, timeoutMs);
    const done = (fn) => (arg) => {
      clearTimeout(timer);
      fn(arg);
    };
    ws.addEventListener("open", done(resolve), { once: true });
    ws.addEventListener(
      "error",
      done(() => reject(new Error("cdp socket error"))),
      { once: true },
    );
  });
  try {
    // A new target starts on about:blank and commits the requested URL later;
    // evaluating before that runs in an opaque origin with no LanguageModel.
    // /json/new does not decode its URL, so navigate explicitly as well.
    const wanted = new URL(url).origin;
    if ((await evaluate(ws, "location.origin")) !== wanted) {
      await evaluate(ws, `location.href = ${JSON.stringify(url)}`);
    }
    while (Date.now() < deadline &&
           (await evaluate(ws, "location.origin")) !== wanted) {
      await new Promise((r) => setTimeout(r, 200));
    }
    const out = await evaluate(ws, source);
    console.log(JSON.stringify({ browser: devtools.Browser, ...out }, null, 2));
    process.exitCode = out?.ok ? 0 : 1;
  } finally {
    ws.close();
  }
};

main().catch((err) => {
  console.error(JSON.stringify({ ok: false, error: String(err) }, null, 2));
  process.exitCode = 1;
});
