export function shouldShowDebugTrace(env = {}) {
  return Boolean(env.DEV)
    || String(env.VITE_SHOW_DEBUG_TRACE ?? "").toLowerCase() === "true";
}
