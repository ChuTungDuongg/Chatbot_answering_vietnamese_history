import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  createConversation: vi.fn(),
  updateConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));

const api = await import("../../src/services/api.js");
const { useChatSession } = await import("../../src/hooks/useChatSession.js");

beforeEach(() => {
  api.listConversations.mockResolvedValue([]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
  api.createConversation.mockResolvedValue({ conversation: { id: "c-mới" } });
  api.updateConversation.mockResolvedValue({ title: "Tên mới" });
  api.deleteConversation.mockResolvedValue({});
});

test("bootstrap nạp danh sách và mở hội thoại đầu tiên", async () => {
  api.listConversations.mockResolvedValue([{ id: "c1" }, { id: "c2" }]);
  api.getConversation.mockResolvedValue({
    messages: [{ id: "m1", role: "assistant", sources: [] }],
    attachments: [],
  });

  const { result } = renderHook(() => useChatSession());

  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));
  expect(result.current.state.activeConversationId).toBe("c1");
  expect(result.current.state.messages).toHaveLength(1);
});

test("bootstrap hỏng vẫn tắt spinner và hiện lỗi", async () => {
  api.listConversations.mockRejectedValue(new Error("Mất mạng"));

  const { result } = renderHook(() => useChatSession());

  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));
  expect(result.current.state.error).toBe("Mất mạng");
});

test("ensureActiveConversation tái sử dụng hội thoại đang mở, không gọi tạo mới", async () => {
  api.listConversations.mockResolvedValue([{ id: "c1" }]);

  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.activeConversationId).toBe("c1"));

  const id = await result.current.ensureActiveConversation();

  expect(id).toBe("c1");
  expect(api.createConversation).not.toHaveBeenCalled();
});

test("ensureActiveConversation tạo hội thoại mới khi chưa có", async () => {
  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  const id = await result.current.ensureActiveConversation();

  expect(id).toBe("c-mới");
});

test("ensureActiveConversation ném lỗi rõ ràng khi đang stream, không ném TypeError", async () => {
  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  act(() => {
    result.current.dispatch({ type: "STREAM_STATUS", messageId: "m", status: "streaming" });
  });
  expect(result.current.isRunning).toBe(true);

  await expect(result.current.ensureActiveConversation()).rejects.toThrow(
    "Không thể tạo cuộc trò chuyện mới.",
  );
  expect(api.createConversation).not.toHaveBeenCalled();
});

test("ensureActiveConversation ném lỗi khi backend không trả về id", async () => {
  api.createConversation.mockResolvedValue({});

  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  await expect(result.current.ensureActiveConversation()).rejects.toThrow(
    "Backend không trả về conversation ID.",
  );
});

test("loadConversation hỏng vẫn tắt cờ đang tải và ghi lỗi", async () => {
  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  api.getConversation.mockRejectedValue(new Error("Không tải nổi"));

  await act(async () => {
    await expect(result.current.loadConversation("c9")).rejects.toThrow("Không tải nổi");
  });

  expect(result.current.state.isLoadingConversation).toBe(false);
  expect(result.current.state.error).toBe("Không tải nổi");
});

test("removeConversation trả về danh sách id còn lại", async () => {
  api.listConversations.mockResolvedValue([{ id: "c1" }, { id: "c2" }]);

  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.conversations).toHaveLength(2));

  let remaining;
  await act(async () => {
    remaining = await result.current.removeConversation("c1");
  });

  expect(remaining).toEqual(["c2"]);
  expect(api.deleteConversation).toHaveBeenCalledWith("c1");
});
