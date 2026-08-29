import test from "node:test";
import assert from "node:assert/strict";

import { sanitizeTraceForCopy } from "../src/services/trace.js";


test("trace sanitizer removes secrets, prompts, and hidden reasoning", () => {
  const sanitized = sanitizeTraceForCopy({
    mode: "agentic_rag",
    prompt: "private prompt",
    chain_of_thought: "private reasoning",
    nested: {
      System_Prompt: "private system prompt",
      authorization: "Bearer private-token",
      generation_calls: 2,
    },
  });

  assert.deepEqual(sanitized, {
    mode: "agentic_rag",
    nested: { generation_calls: 2 },
  });
});
