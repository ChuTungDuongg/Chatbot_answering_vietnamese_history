import test from "node:test";
import assert from "node:assert/strict";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

test("rendered answer and source panel use display indices while debug retains real IDs", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false } });
  try {
    const { default: RetrievedChunks } = await server.ssrLoadModule("/src/components/RetrievedChunks.jsx");
    const { default: ChatMessage } = await server.ssrLoadModule("/src/components/ChatMessage.jsx");
    const rawId = "hf_wikipedia_điện_biên_phủ_0001_internal";
    const sources = [{ chunk_id: rawId, source_id: rawId, display_index: 3, title: "Chiến dịch Điện Biên Phủ", text: "Bằng chứng.", source_kind: "history" }];
    const panel = renderToStaticMarkup(React.createElement(RetrievedChunks, { sources }));
    assert.match(panel, /\[3\] Chiến dịch Điện Biên Phủ/);
    assert.doesNotMatch(panel, /hf_wikipedia/);
    const message = { role: "assistant", mode: "central", content: "Năm [1954]. [3]", sources, debug_trace: { sources } };
    const answer = renderToStaticMarkup(React.createElement(ChatMessage, { message }));
    assert.match(answer, /\[1954\]\. \[3\]/);
    assert.match(answer, /1 nguồn/);
    assert.doesNotMatch(answer, /hf_wikipedia/);
    const debug = renderToStaticMarkup(React.createElement(ChatMessage, { message, enableDebugTrace: true }));
    assert.match(debug, /hf_wikipedia/);
  } finally {
    await server.close();
  }
});
