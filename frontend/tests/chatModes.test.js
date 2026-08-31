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


test("mode contract exposes exactly Fast, Hybrid, Agent", () => {
  assert.deepEqual(CHAT_MODES.map(({ value, label }) => ({ value, label })), [
    { value: "fast", label: "Fast" },
    { value: "hybrid", label: "Hybrid" },
    { value: "agent", label: "Agent" },
  ]);
  assert.deepEqual(CHAT_MODES.map((item) => item.description), [
    "Nhanh, phản hồi trực tiếp",
    "Kết hợp hệ thống 3 mô hình hiện tại",
    "Central Agent tự chọn công cụ và nguồn",
  ]);
});


test("selected mode persists and invalid storage falls back to Fast", () => {
  const storage = memoryStorage();
  assert.equal(readStoredChatMode(storage), ChatMode.FAST);
  persistChatMode(ChatMode.AGENT, storage);
  assert.equal(storage.getItem(CHAT_MODE_STORAGE_KEY), ChatMode.AGENT);
  assert.equal(readStoredChatMode(storage), ChatMode.AGENT);

  const invalid = memoryStorage({ [CHAT_MODE_STORAGE_KEY]: "unknown" });
  assert.equal(readStoredChatMode(invalid), ChatMode.FAST);
});


test("compact mode dropdown is inside the composer's left action group", async () => {
  const source = await readFile(new URL("../src/components/ChatInput.jsx", import.meta.url), "utf8");
  const leading = source.indexOf('className="composer-leading-actions"');
  const selector = source.indexOf("<ModeSelector", leading);
  const textarea = source.indexOf("<textarea", selector);
  const submit = source.indexOf("composer-submit", textarea);

  assert.ok(leading >= 0 && selector > leading && textarea > selector && submit > textarea);
});


test("frontend sends the selected canonical mode to the streaming API", async () => {
  const appSource = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const apiSource = await readFile(new URL("../src/services/api.js", import.meta.url), "utf8");

  assert.match(appSource, /mode:\s*inferenceMode/);
  assert.match(apiSource, /JSON\.stringify\([\s\S]*?mode,/);
});
