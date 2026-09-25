export const ChatMode = Object.freeze({
  HYBRID: "hybrid",
  CENTRAL: "central",
});

export const CHAT_MODES = Object.freeze([
  Object.freeze({
    value: ChatMode.HYBRID,
    label: "Hybrid RAG",
    description: "Truy xuất tư liệu + Qwen3-4B",
  }),
  Object.freeze({
    value: ChatMode.CENTRAL,
    label: "Central Agent",
    description: "Qwen3-8B gọi công cụ để tìm tư liệu",
  }),
]);

export const CHAT_MODE_STORAGE_KEY = "vn-history-chat-mode-v2";

export function isChatMode(value) {
  return CHAT_MODES.some((mode) => mode.value === value);
}

export function readStoredChatMode(storage = globalThis.localStorage) {
  try {
    const stored = storage?.getItem(CHAT_MODE_STORAGE_KEY);
    if (isChatMode(stored)) return stored;
    storage?.setItem(CHAT_MODE_STORAGE_KEY, ChatMode.HYBRID);
    return ChatMode.HYBRID;
  } catch {
    return ChatMode.HYBRID;
  }
}

export function persistChatMode(mode, storage = globalThis.localStorage) {
  if (!isChatMode(mode)) throw new Error(`Unsupported chat mode: ${mode}`);
  try {
    storage?.setItem(CHAT_MODE_STORAGE_KEY, mode);
  } catch {
    // Storage can be unavailable in private/sandboxed browsing contexts.
  }
}
