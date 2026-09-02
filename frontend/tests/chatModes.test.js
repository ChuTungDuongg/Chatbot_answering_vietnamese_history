import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CHAT_MODES,
  CHAT_MODE_STORAGE_KEY,
  LEGACY_CHAT_MODE_STORAGE_KEY,
  ChatMode,
  persistChatMode,
  readStoredChatMode,
} from "../src/config/chatModes.js";


function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
}


test("mode contract exposes exactly Hybrid, 3 LLM, Central Agent", () => {
  assert.deepEqual(CHAT_MODES.map(({ value, label }) => ({ value, label })), [
    { value: "hybrid", label: "Hybrid" },
    { value: "three_llm", label: "3 LLM" },
    { value: "central", label: "Central Agent" },
  ]);
  assert.deepEqual(CHAT_MODES.map((item) => item.description), [
    "Hybrid retrieval + một mô hình trả lời",
    "Research + Evidence + History Answerer",
    "Qwen3-8B tự nghiên cứu và gọi công cụ",
  ]);
});


test("selected mode persists and invalid storage falls back to Hybrid", () => {
  const storage = memoryStorage();
  assert.equal(readStoredChatMode(storage), ChatMode.HYBRID);
  persistChatMode(ChatMode.CENTRAL, storage);
  assert.equal(storage.getItem(CHAT_MODE_STORAGE_KEY), ChatMode.CENTRAL);
  assert.equal(readStoredChatMode(storage), ChatMode.CENTRAL);

  const invalid = memoryStorage({ [CHAT_MODE_STORAGE_KEY]: "unknown" });
  assert.equal(readStoredChatMode(invalid), ChatMode.HYBRID);
});


test("legacy localStorage values migrate without changing old execution meaning", () => {
  const oldHybrid = memoryStorage({ [LEGACY_CHAT_MODE_STORAGE_KEY]: "hybrid" });
  const oldFast = memoryStorage({ [LEGACY_CHAT_MODE_STORAGE_KEY]: "fast" });
  const oldAgent = memoryStorage({ [LEGACY_CHAT_MODE_STORAGE_KEY]: "agent" });

  assert.equal(readStoredChatMode(oldHybrid), ChatMode.THREE_LLM);
  assert.equal(readStoredChatMode(oldFast), ChatMode.HYBRID);
  assert.equal(readStoredChatMode(oldAgent), ChatMode.CENTRAL);
});


test("compact mode dropdown is inside the composer's left action group", async () => {
  const source = await readFile(new URL("../src/components/ChatInput.jsx", import.meta.url), "utf8");
  const leading = source.indexOf('className="composer-leading-actions"');
  const selector = source.indexOf("<ModeSelector", leading);
  const textarea = source.indexOf("<textarea", selector);
  const submit = source.indexOf("composer-submit", textarea);

  assert.ok(leading >= 0 && selector > leading && textarea > selector && submit > textarea);
});
