const SECRET_KEY = /authorization|cookie|api[_-]?key|secret|credential|password|environment|headers?|modal/i;
const OMITTED_KEY = new Set([
  "chain_of_thought",
  "developer_prompt",
  "hidden_reasoning",
  "messages",
  "prompt",
  "rationale",
  "raw_output",
  "reasoning",
  "scratchpad",
  "system_prompt",
  "user_prompt",
  "validated_source_text",
]);

export function sanitizeTraceForCopy(value, depth = 0) {
  if (depth > 8) return "[bounded]";
  if (Array.isArray(value)) {
    return value.slice(0, 100).map((item) => sanitizeTraceForCopy(item, depth + 1));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => !SECRET_KEY.test(key) && !OMITTED_KEY.has(key.toLowerCase()))
        .map(([key, item]) => [key, sanitizeTraceForCopy(item, depth + 1)]),
    );
  }
  if (typeof value === "string") {
    const safe = value
      .replace(/\b(api[_ -]?key|authorization|bearer|token|secret|password)\b\s*[:=]\s*\S+/gi, "$1=[redacted]")
      .replace(/\b[A-Z]:\\(?:Users|ProgramData|Windows)\\[^\s"']+/gi, "[filesystem-path]")
      .replace(/\/(?:home|root|etc|var)\/[^\s"']+/g, "[filesystem-path]");
    if (safe.length > 800) return `${safe.slice(0, 797).trimEnd()}...`;
    return safe;
  }
  return value;
}

export function formatTraceForCopy(trace) {
  return JSON.stringify(sanitizeTraceForCopy(trace ?? {}), null, 2);
}

export function hasTraceSection(value) {
  if (Array.isArray(value)) return value.length > 0;
  return Boolean(value && typeof value === "object" && Object.keys(value).length > 0);
}
