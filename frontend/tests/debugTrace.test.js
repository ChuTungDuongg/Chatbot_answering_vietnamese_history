import test from "node:test";
import assert from "node:assert/strict";

import { shouldShowDebugTrace } from "../src/services/debugTrace.js";


test("debug trace is enabled during Vite development", () => {
  assert.equal(shouldShowDebugTrace({ DEV: true }), true);
});

test("debug trace is enabled by the explicit test-build flag", () => {
  assert.equal(
    shouldShowDebugTrace({ DEV: false, VITE_SHOW_DEBUG_TRACE: "true" }),
    true,
  );
});

test("debug trace stays disabled in production when the flag is absent or false", () => {
  assert.equal(shouldShowDebugTrace({ DEV: false }), false);
  assert.equal(
    shouldShowDebugTrace({ DEV: false, VITE_SHOW_DEBUG_TRACE: "false" }),
    false,
  );
});
