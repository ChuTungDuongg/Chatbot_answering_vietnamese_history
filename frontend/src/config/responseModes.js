export const RESPONSE_MODES = ["concise", "standard", "detailed"];
export const RESPONSE_MODE_LABELS = ["Ngắn gọn", "Tiêu chuẩn", "Chi tiết"];
const STORAGE_KEY = "vn-history-response-mode";

export function readResponseMode() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    return RESPONSE_MODES.includes(saved) ? saved : "standard";
  } catch {
    return "standard";
  }
}

export function persistResponseMode(mode) {
  if (!RESPONSE_MODES.includes(mode)) throw new Error(`Invalid response mode: ${mode}`);
  try { window.localStorage.setItem(STORAGE_KEY, mode); } catch { /* local storage may be blocked */ }
}
