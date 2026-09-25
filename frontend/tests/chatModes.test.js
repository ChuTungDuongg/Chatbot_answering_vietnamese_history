import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CHAT_MODES,
  CHAT_MODE_STORAGE_KEY,
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


test("mode contract exposes exactly Hybrid RAG and Central Agent", () => {
  assert.deepEqual(CHAT_MODES.map(({ value, label }) => ({ value, label })), [
    { value: "hybrid", label: "Hybrid RAG" },
    { value: "central", label: "Central Agent" },
  ]);
  assert.deepEqual(CHAT_MODES.map((item) => item.description), [
    "Truy xuất tư liệu + Qwen3-4B",
    "Qwen3-8B gọi công cụ để tìm tư liệu",
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
  assert.equal(invalid.getItem(CHAT_MODE_STORAGE_KEY), ChatMode.HYBRID);
});


test("unsupported mode values fall back to Hybrid and cannot be saved", () => {
  const storage = memoryStorage({ [CHAT_MODE_STORAGE_KEY]: "unsupported" });
  assert.equal(readStoredChatMode(storage), ChatMode.HYBRID);
  assert.equal(storage.getItem(CHAT_MODE_STORAGE_KEY), ChatMode.HYBRID);
  assert.throws(() => persistChatMode("unsupported", storage), /Unsupported chat mode/);
});


test("compact mode dropdown is inside the composer's left action group", async () => {
  const source = await readFile(new URL("../src/components/ChatInput.jsx", import.meta.url), "utf8");
  const textarea = source.indexOf("<textarea");
  const toolbar = source.indexOf('className="composer-toolbar"', textarea);
  const leading = source.indexOf('className="composer-leading-actions"', toolbar);
  const selector = source.indexOf("<ModeSelector", leading);
  const submit = source.indexOf("composer-submit", selector);

  assert.ok(textarea >= 0 && toolbar > textarea && leading > toolbar && selector > leading && submit > selector);
});
