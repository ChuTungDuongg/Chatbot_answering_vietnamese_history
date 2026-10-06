import test from "node:test";
import assert from "node:assert/strict";

import { sanitizeTraceForCopy } from "../src/services/trace.js";


test("trace sanitizer removes secrets, prompts, and hidden reasoning", () => {
  const sanitized = sanitizeTraceForCopy({
    mode: "central",
    prompt: "private prompt",
    chain_of_thought: "private reasoning",
    nested: {
      System_Prompt: "private system prompt",
      authorization: "Bearer private-token",
      generation_calls: 2,
    },
  });

  assert.deepEqual(sanitized, {
    mode: "central",
    nested: { generation_calls: 2 },
  });
});

test("planning and termination telemetry survive trace sanitization", () => {
  const trace = {
    planning: { total_model_ms: 1000, rounds: [{ round: 1, model_ms: 1000, ttft_ms: 100,
      input_tokens: 200, output_tokens: 30, tools_requested: ["search_history"] }] },
    generation: { settings: { max_new_tokens: 2048 }, finish_reason: "length",
      hit_max_new_tokens: true, truncated: true },
    performance: { e2e_ms: 2500 },
  };
  assert.deepEqual(sanitizeTraceForCopy(trace), trace);
});
