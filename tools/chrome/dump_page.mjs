#!/usr/bin/env node
// Dump a chrome:// page's text over CDP.
//
// Used to read chrome://on-device-internals, which is Chrome's own view of the
// manifest broker - it answers "why is availability X" directly, where log
// scraping only shows that nothing happened.
//
// Usage: dump_page.mjs <cdp-port> <url>

const port = Number(process.argv[2]);
const url = process.argv[3];

async function waitForDevtools() {
  for (let i = 0; i < 240; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (res.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("devtools never came up");
}

const evaluate = (ws, expression) =>
  new Promise((resolve, reject) => {
    const id = Math.floor(Math.random() * 1e9);
    const onMessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id !== id) return;
      ws.removeEventListener("message", onMessage);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result?.result?.value);
    };
    ws.addEventListener("message", onMessage);
    ws.send(
      JSON.stringify({
        id,
        method: "Runtime.evaluate",
        params: { expression, awaitPromise: true, returnByValue: true },
      }),
    );
  });

const main = async () => {
  await waitForDevtools();
  const res = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, {
    method: "PUT",
  });
  const page = await res.json();
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  await new Promise((r) => setTimeout(r, 6000));
  const text = await evaluate(
    ws,
    "(document.body && document.body.innerText) || document.documentElement.outerHTML.slice(0,2000)",
  );
  console.log(text);
  ws.close();
};

main().catch((e) => {
  console.error(String(e));
  process.exitCode = 1;
});
