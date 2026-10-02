import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  streamChat: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
}));

const api = await import("../../src/services/api.js");
const { useChatStream } = await import("../../src/hooks/useChatStream.js");

function setup({ isRunning = false, mode = "central", retrievalBackend = "faiss", steering } = {}) {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockResolvedValue("c1");
  const { result } = renderHook(() => useChatStream({
    dispatch,
    isRunning,
    mode,
    retrievalBackend,
    steering,
    showDebugTrace: false,
    ensureActiveConversation,
  }));
  return { dispatch, ensureActiveConversation, result };
}

beforeEach(() => {
  api.streamChat.mockResolvedValue(undefined);
  // Hook gọi hai hàm này để đồng bộ lại sau khi stream xong. Thiếu chúng thì
  // mọi test "thành công" sẽ ngã vào nhánh catch và xanh vì lý do sai.
  api.listConversations.mockResolvedValue([{ id: "c1" }]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
});

test.each(["hybrid", "central"])("gửi đúng chế độ %s tới API streaming", async (mode) => {
  const { result } = setup({ mode });

  await act(async () => {
    await result.current.submit("Chiến thắng Bạch Đằng?");
  });

  expect(api.streamChat).toHaveBeenCalledTimes(1);
  expect(api.streamChat.mock.calls[0][0]).toMatchObject({
    conversationId: "c1",
    question: "Chiến thắng Bạch Đằng?",
    mode,
    finalK: 6,
    retrievalBackend: "faiss",
  });
  expect(api.streamChat.mock.calls[0][0]).not.toHaveProperty("responseMode");
});

test("Central forwards request steering snapshot and abort stops MCP progress updates", async () => {
  const steering = { mcp_enabled: true, allowed_mcp_servers: ["research"], allowed_tools: ["search_history", "mcp__research__lookup"] };
  let emit, finish, signal;
  api.streamChat.mockImplementation(({ onEvent, signal: requestSignal }) => {
    emit = onEvent; signal = requestSignal;
    return new Promise((resolve) => { finish = resolve; });
  });
  const { result, dispatch } = setup({ steering });
  let submission;
  await act(async () => { submission = result.current.submit("History?"); });
  expect(api.streamChat.mock.calls[0][0].steering).toEqual(steering);
  act(() => emit({ event: "status", data: { stage: "tool:mcp__research__lookup", state: "started", provider: "mcp" } }));
  expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ type: "STREAM_STATUS", progress: expect.objectContaining({ provider: "mcp" }) }));
  act(() => result.current.stop());
  expect(signal.aborted).toBe(true);
  const count = dispatch.mock.calls.length;
  act(() => emit({ event: "status", data: { stage: "tool:mcp__research__lookup", state: "completed" } }));
  expect(dispatch.mock.calls.length).toBe(count);
  await act(async () => { finish(); await submission; });
});

test("bỏ qua câu hỏi rỗng và khi đang chạy", async () => {
  const { result } = setup();
  await act(async () => {
    await result.current.submit("    ");
  });
  expect(api.streamChat).not.toHaveBeenCalled();

  const running = setup({ isRunning: true });
  await act(async () => {
    await running.result.current.submit("Có nội dung");
  });
  expect(api.streamChat).not.toHaveBeenCalled();
});

test("chuyển các sự kiện SSE thành action tương ứng", async () => {
  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({ event: "status", data: { stage: "hybrid_retrieval", mode: "hybrid" } });
    onEvent({ event: "answer_delta", data: { delta: "Xin " } });
    onEvent({ event: "answer_delta", data: "chào" });
    onEvent({ event: "sources", data: { items: [{ id: "s1" }] } });
    onEvent({ event: "done", data: {} });
  });

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("MESSAGES_APPENDED");
  expect(types).toContain("STREAM_STATUS");
  expect(types).toContain("STREAM_DELTA");
  expect(types).toContain("STREAM_SOURCES");
  expect(types).toContain("STREAM_DONE");

  const deltas = dispatch.mock.calls
    .map(([action]) => action)
    .filter((action) => action.type === "STREAM_DELTA")
    .map((action) => action.delta);
  expect(deltas).toEqual(["Xin ", "chào"]);
});

test("sự kiện error chuyển thông báo tới reducer", async () => {
  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({
      event: "error",
      data: { message: "Tạo câu trả lời thất bại", type: "generation_error" },
    });
  });

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const errorAction = dispatch.mock.calls
    .map(([action]) => action)
    .find((action) => action.type === "STREAM_ERROR");

  expect(errorAction.message).toBe("Tạo câu trả lời thất bại");
});

test("AbortError sinh ra STREAM_ABORTED chứ không phải STREAM_ERROR", async () => {
  const abortError = new Error("Aborted");
  abortError.name = "AbortError";
  api.streamChat.mockRejectedValue(abortError);

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("STREAM_ABORTED");
  expect(types).not.toContain("STREAM_ERROR");
});

test("stop() huỷ request đang chạy", async () => {
  let capturedSignal;
  api.streamChat.mockImplementation(async ({ signal }) => {
    capturedSignal = signal;
    await new Promise((resolve) => setTimeout(resolve, 50));
  });

  const { result } = setup();
  let pending;
  await act(async () => {
    pending = result.current.submit("Hỏi");
    await Promise.resolve();
  });

  act(() => result.current.stop());
  await act(async () => { await pending; });

  expect(capturedSignal.aborted).toBe(true);
});

test("không tạo được hội thoại thì báo lỗi và không gọi streaming", async () => {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockRejectedValue(new Error("Không thể tạo cuộc trò chuyện."));
  const { result } = renderHook(() => useChatStream({
    dispatch, isRunning: false, mode: "hybrid", showDebugTrace: false, ensureActiveConversation,
  }));

  await act(async () => {
    await result.current.submit("Hỏi");
  });

  expect(api.streamChat).not.toHaveBeenCalled();
  expect(dispatch).toHaveBeenCalledWith(
    expect.objectContaining({ type: "ERROR_SET", message: "Không thể tạo cuộc trò chuyện." }),
  );
});
