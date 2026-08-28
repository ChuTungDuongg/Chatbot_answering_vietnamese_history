const API_BASE_URL = String(import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");
const CLIENT_ID_STORAGE_KEY = "vn-history-client-id";
export const EVIDENCE_CONTRACT_FAILURE_MESSAGE = "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.";

function ensureApiConfigured() {
  if (!API_BASE_URL) {
    throw new Error("VITE_API_BASE_URL chưa được cấu hình trong frontend/.env.");
  }
}

function getClientId() {
  let clientId = window.localStorage.getItem(CLIENT_ID_STORAGE_KEY);

  if (!clientId) {
    clientId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    window.localStorage.setItem(CLIENT_ID_STORAGE_KEY, clientId);
  }

  return clientId;
}

function createHeaders(headers = {}) {
  return { "X-Client-ID": getClientId(), ...headers };
}

async function getErrorMessage(response) {
  const fallback = `Yêu cầu thất bại với mã ${response.status}.`;

  try {
    const contentType = response.headers.get("content-type") ?? "";

    if (contentType.includes("application/json")) {
      const body = await response.json();
      const detail = body?.detail ?? body?.message ?? body?.error;

      if (typeof detail === "string") return detail;
      if (detail) return JSON.stringify(detail);
      return fallback;
    }

    return (await response.text()).trim() || fallback;
  } catch {
    return fallback;
  }
}

async function requestJson(path, options = {}) {
  ensureApiConfigured();

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: createHeaders(options.headers),
  });

  if (!response.ok) throw new Error(await getErrorMessage(response));
  if (response.status === 204) return null;

  const contentType = response.headers.get("content-type") ?? "";
  return contentType.includes("application/json") ? response.json() : null;
}

function parseSSEBlock(block) {
  let event = "message";
  const dataLines = [];

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }

  if (dataLines.length === 0) return null;

  const rawData = dataLines.join("\n");

  try {
    return { event, data: JSON.parse(rawData) };
  } catch {
    return { event, data: rawData };
  }
}

export function listConversations({ signal } = {}) {
  return requestJson("/api/v1/conversations", { method: "GET", signal });
}

export function createConversation({ title = null, signal } = {}) {
  return requestJson("/api/v1/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
    signal,
  });
}

export function getConversation(conversationId, { signal } = {}) {
  if (!conversationId) throw new Error("conversationId is required.");
  return requestJson(`/api/v1/conversations/${conversationId}`, { method: "GET", signal });
}

export function updateConversation(conversationId, { title, signal } = {}) {
  if (!conversationId) throw new Error("conversationId is required.");
  if (!title?.trim()) throw new Error("Tên cuộc trò chuyện không được để trống.");

  return requestJson(`/api/v1/conversations/${conversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title.trim() }),
    signal,
  });
}

export function deleteConversation(conversationId, { signal } = {}) {
  if (!conversationId) throw new Error("conversationId is required.");
  return requestJson(`/api/v1/conversations/${conversationId}`, { method: "DELETE", signal });
}

export function uploadAttachment(conversationId, file, { signal } = {}) {
  if (!conversationId) throw new Error("conversationId is required.");
  if (!(file instanceof File)) throw new Error("File tải lên không hợp lệ.");

  const formData = new FormData();
  formData.append("file", file);

  return requestJson(`/api/v1/conversations/${conversationId}/attachments`, {
    method: "POST",
    body: formData,
    signal,
  });
}

export function deleteAttachment(conversationId, attachmentId, { signal } = {}) {
  if (!conversationId || !attachmentId) {
    throw new Error("conversationId and attachmentId are required.");
  }

  return requestJson(`/api/v1/conversations/${conversationId}/attachments/${attachmentId}`, {
    method: "DELETE",
    signal,
  });
}

export async function streamChat({
  conversationId,
  question,
  mode = "agentic_rag",
  finalK = 6,
  debug = false,
  onEvent,
  signal,
}) {
  ensureApiConfigured();

  const normalizedQuestion = question?.trim();
  if (!conversationId) throw new Error("conversationId is required.");
  if (!normalizedQuestion) throw new Error("Câu hỏi không được để trống.");

  const response = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: createHeaders({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    }),
    body: JSON.stringify({
      conversation_id: conversationId,
      question: normalizedQuestion,
      mode,
      final_k: finalK,
      debug,
    }),
    signal,
  });

  if (!response.ok) throw new Error(await getErrorMessage(response));
  if (!response.body) throw new Error("Trình duyệt không nhận được response stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let sawStreamError = false;

  try {
    while (!sawStreamError) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";

      for (const block of blocks) {
        const parsed = parseSSEBlock(block);
        if (parsed) {
          onEvent?.(parsed);
          if (parsed.event === "error") {
            sawStreamError = true;
            break;
          }
        }
      }
    }

    if (sawStreamError) {
      await reader.cancel();
    } else {
      buffer += decoder.decode();
      const parsed = buffer.trim() ? parseSSEBlock(buffer) : null;
      if (parsed) onEvent?.(parsed);
    }
  } finally {
    reader.releaseLock();
  }
}
