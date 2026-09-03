import assert from "node:assert/strict";
import test from "node:test";

test("renders RiskPulse dashboard metadata and content", async () => {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  const response = await worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );

  assert.equal(response.status, 200);
  assert.match(
    response.headers.get("content-type") ?? "",
    /^text\/html\b/i,
  );
  const html = await response.text();
  assert.match(html, /<title>RiskPulse Operations<\/title>/i);
  assert.match(
    html,
    /<meta(?=[^>]*\bname=["']description["'])(?=[^>]*\bcontent=["']Fraud scoring, analyst review, and model drift monitoring\.["'])[^>]*>/i,
  );
  assert.match(html, /Today(?:&#x27;|')s risk posture/);
  assert.match(html, /Manual-review queue/);
});
