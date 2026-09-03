import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

test("BrandMark stays vector at small sizes and has decorative/standalone accessibility", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false } });
  try {
    const { default: BrandMark } = await server.ssrLoadModule("/src/components/BrandMark.jsx");
    for (const size of [16, 20, 24, 32, 40]) {
      const markup = renderToStaticMarkup(React.createElement(BrandMark, { size, className: "sample" }));
      assert.match(markup, new RegExp(`width="${size}" height="${size}"`));
      assert.match(markup, /aria-hidden="true"/);
      assert.match(markup, /class="brand-mark sample"/);
      assert.equal((markup.match(/<path /g) || []).length, 5);
      assert.equal((markup.match(/<circle /g) || []).length, 1);
      assert.doesNotMatch(markup, /<image|<rect|role="img"/);
    }
    const standalone = renderToStaticMarkup(React.createElement(BrandMark, { label: "Sử Việt AI" }));
    assert.match(standalone, /role="img" aria-label="Sử Việt AI"/);
    assert.doesNotMatch(standalone, /aria-hidden/);
    const { default: ChatMessage } = await server.ssrLoadModule("/src/components/ChatMessage.jsx");
    for (const status of ["done", "central_answering", "central_loading"]) {
      const markup = renderToStaticMarkup(React.createElement(ChatMessage, { message: { role: "assistant", status, content: "" } }));
      assert.match(markup, /brand-mark-petals/);
      assert.match(markup, /aria-hidden="true"/);
    }
  } finally {
    await server.close();
  }
});

test("UI contains no old logo and favicon uses the same blossom geometry", async () => {
  const root = new URL("../src/", import.meta.url);
  const files = await readdir(root, { recursive: true });
  for (const file of files.filter((name) => /\.(jsx?|css)$/.test(name))) {
    const content = await readFile(new URL(file.replaceAll("\\", "/"), root), "utf8");
    assert.doesNotMatch(content, /LogoMark|logo-mark|logo-ink/, file);
  }
  const component = await readFile(new URL("components/BrandMark.jsx", root), "utf8");
  const petal = component.match(/const PETAL = "([^"]+)"/)[1];
  const favicon = await readFile(new URL("../public/favicon.svg", import.meta.url), "utf8");
  assert.ok(favicon.includes(petal));
  assert.equal((favicon.match(/<use /g) || []).length, 5);
  assert.match(favicon, /prefers-color-scheme: dark/);
});
