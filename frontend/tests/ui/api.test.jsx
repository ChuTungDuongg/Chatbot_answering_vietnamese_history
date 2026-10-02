import { afterEach, expect, test, vi } from "vitest";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

test.each(["hybrid", "central"])("tool steering is serialized only for %s Central requests", async (mode) => {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE_URL", "https://api.example.test");
  const fetchMock = vi.fn().mockResolvedValue(new Response('event: done\ndata: {"status":"done"}\n\n'));
  vi.stubGlobal("fetch", fetchMock);
  const { streamChat } = await import("../../src/services/api.js");
  const steering = { mcp_enabled: true, allowed_mcp_servers: ["research"],
    allowed_tools: ["search_history", "mcp__research__lookup"], mcp_failure_policy: "continue" };
  await streamChat({ conversationId: "c1", question: "History?", mode, retrievalBackend: "qdrant", steering });
  const payload = JSON.parse(fetchMock.mock.calls[0][1].body);
  if (mode === "central") expect(payload.steering).toEqual(steering);
  else expect(payload).not.toHaveProperty("steering");
  expect(payload).not.toHaveProperty("response_mode");
});

test.each(["hybrid", "central"])("streamChat sends the %s request without response detail", async (mode) => {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE_URL", "https://api.example.test");
  const frame = 'event: done\ndata: {"status":"done"}\n\n';
  const fetchMock = vi.fn().mockResolvedValue(new Response(frame, {
    headers: { "Content-Type": "text/event-stream" },
  }));
  vi.stubGlobal("fetch", fetchMock);
  const { streamChat } = await import("../../src/services/api.js");
  const controller = new AbortController();
  const onEvent = vi.fn();
  await streamChat({ conversationId: "c1", question: " Bạch Đằng? ",
    attachmentIds: ["att1"], mode, finalK: 3, debug: true,
    retrievalBackend: "qdrant",
    signal: controller.signal, onEvent });
  const [url, request] = fetchMock.mock.calls[0];
  expect(url).toBe("https://api.example.test/api/v1/chat/stream");
  expect(JSON.parse(request.body)).toEqual({ conversation_id: "c1", question: "Bạch Đằng?",
    attachment_ids: ["att1"], mode, retrieval_backend: "qdrant", final_k: 3, debug: true });
  expect(request.signal).toBe(controller.signal);
  expect(onEvent).toHaveBeenCalledWith({ event: "done", data: { status: "done" } });
});
