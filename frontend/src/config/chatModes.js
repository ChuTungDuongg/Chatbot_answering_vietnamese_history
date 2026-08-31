export const ChatMode = Object.freeze({
  FAST: "fast",
  HYBRID: "hybrid",
  AGENT: "agent",
});

export const CHAT_MODES = Object.freeze([
  Object.freeze({
    value: ChatMode.FAST,
    label: "Fast",
    description: "Nhanh, phản hồi trực tiếp",
  }),
  Object.freeze({
    value: ChatMode.HYBRID,
    label: "Hybrid",
    description: "Kết hợp hệ thống 3 mô hình hiện tại",
  }),
  Object.freeze({
    value: ChatMode.AGENT,
    label: "Agent",
    description: "Central Agent tự chọn công cụ và nguồn",
  }),
]);

export const CHAT_MODE_STORAGE_KEY = "vn-history-chat-mode";

export function isChatMode(value) {
  return CHAT_MODES.some((mode) => mode.value === value);
}

export function readStoredChatMode(storage = globalThis.localStorage) {
  try {
    const stored = storage?.getItem(CHAT_MODE_STORAGE_KEY);
    return isChatMode(stored) ? stored : ChatMode.FAST;
  } catch {
    return ChatMode.FAST;
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
