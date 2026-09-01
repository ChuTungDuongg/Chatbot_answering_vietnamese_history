export const ChatMode = Object.freeze({
  HYBRID: "hybrid",
  THREE_LLM: "three_llm",
  CENTRAL: "central",
});

export const CHAT_MODES = Object.freeze([
  Object.freeze({
    value: ChatMode.HYBRID,
    label: "Hybrid",
    description: "Hybrid retrieval + một mô hình trả lời",
  }),
  Object.freeze({
    value: ChatMode.THREE_LLM,
    label: "3 LLM",
    description: "Research + Evidence + History Answerer",
  }),
  Object.freeze({
    value: ChatMode.CENTRAL,
    label: "Central Agent",
    description: "Qwen3-8B tự nghiên cứu và gọi công cụ",
  }),
]);

export const CHAT_MODE_STORAGE_KEY = "vn-history-chat-mode-v2";
export const LEGACY_CHAT_MODE_STORAGE_KEY = "vn-history-chat-mode";

const LEGACY_MODE_MAP = Object.freeze({
  fast: ChatMode.HYBRID,
  hybrid_rag: ChatMode.HYBRID,
  hybrid: ChatMode.THREE_LLM,
  agentic_rag: ChatMode.THREE_LLM,
  agent: ChatMode.CENTRAL,
});

export function isChatMode(value) {
  return CHAT_MODES.some((mode) => mode.value === value);
}

export function readStoredChatMode(storage = globalThis.localStorage) {
  try {
    const stored = storage?.getItem(CHAT_MODE_STORAGE_KEY);
    if (isChatMode(stored)) return stored;
    const legacy = storage?.getItem(LEGACY_CHAT_MODE_STORAGE_KEY);
    const migrated = LEGACY_MODE_MAP[legacy] ?? ChatMode.HYBRID;
    storage?.setItem(CHAT_MODE_STORAGE_KEY, migrated);
    return migrated;
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
