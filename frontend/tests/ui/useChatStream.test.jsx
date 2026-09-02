import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  streamChat: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  EVIDENCE_CONTRACT_FAILURE_MESSAGE: "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
}));

const api = await import("../../src/services/api.js");
const { useChatStream } = await import("../../src/hooks/useChatStream.js");

function setup({ isRunning = false, mode = "central" } = {}) {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockResolvedValue("c1");
  const { result } = renderHook(() => useChatStream({
    dispatch,
    isRunning,
    mode,
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

test("gửi đúng chế độ đang chọn tới API streaming", async () => {
  const { result } = setup({ mode: "three_llm" });

  await act(async () => {
    await result.current.submit("Chiến thắng Bạch Đằng?");
  });

  expect(api.streamChat).toHaveBeenCalledTimes(1);
  expect(api.streamChat.mock.calls[0][0]).toMatchObject({
    conversationId: "c1",
    question: "Chiến thắng Bạch Đằng?",
    mode: "three_llm",
    finalK: 6,
  });
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

test("sự kiện error mang theo loại lỗi để reducer chọn đúng thông báo", async () => {
  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({
      event: "error",
      data: { message: "Evidence critic từ chối", type: "evidence_contract_error" },
    });
  });

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const errorAction = dispatch.mock.calls
    .map(([action]) => action)
    .find((action) => action.type === "STREAM_ERROR");

  expect(errorAction.kind).toBe("evidence_contract_error");
  expect(errorAction.message).toBe("Evidence critic từ chối");
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
