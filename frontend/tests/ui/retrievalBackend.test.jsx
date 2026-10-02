import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import RetrievalBackendSelector from "../../src/components/RetrievalBackendSelector.jsx";
import RetrievalProgress from "../../src/components/RetrievalProgress.jsx";
import { RETRIEVAL_STORAGE_KEY } from "../../src/config/retrievalBackends.js";
import { chatSessionReducer, initialChatSessionState } from "../../src/state/chatSessionReducer.js";

vi.mock("../../src/services/api.js", () => ({ getRetrievalCapabilities: vi.fn() }));
const api = await import("../../src/services/api.js");
const { useRetrievalBackend } = await import("../../src/hooks/useRetrievalBackend.js");

beforeEach(() => { localStorage.clear(); vi.resetAllMocks(); });

test("only available backend can be selected and keyboard skips unavailable Qdrant", () => {
  const choose = vi.fn();
  render(<RetrievalBackendSelector backend="faiss" available={["faiss"]} onChange={choose} />);
  fireEvent.click(screen.getByRole("button", { name: /Chọn nguồn truy xuất/ }));
  expect(screen.getByRole("option", { name: /Qdrant/ })).toBeDisabled();
  fireEvent.keyDown(screen.getByRole("option", { name: /FAISS/ }), { key: "End" });
  expect(screen.getByRole("option", { name: /FAISS/ })).toHaveFocus();
  fireEvent.click(screen.getByRole("option", { name: /Qdrant/ }));
  expect(choose).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("option", { name: /FAISS/ }));
  expect(choose).toHaveBeenCalledWith("faiss");
});

test("domain-gated progress does not claim a dense search that never ran", () => {
  render(<RetrievalProgress pipeline={[{ stage: "query_analysis", state: "completed", label: "Phân tích câu hỏi" }]}
    backend="qdrant" content="Ngoài phạm vi lịch sử" status="done" />);
  expect(screen.getByRole("status")).toHaveTextContent("Đã chuẩn bị ngữ cảnh");
  expect(screen.getByRole("status")).not.toHaveTextContent("Đã truy xuất bằng Qdrant");
});

test("capability normalizes stored Qdrant to server default FAISS", async () => {
  localStorage.setItem(RETRIEVAL_STORAGE_KEY, "qdrant");
  api.getRetrievalCapabilities.mockResolvedValue({ default_backend: "faiss", available_backends: ["faiss"] });
  const hook = renderHook(useRetrievalBackend);
  expect(hook.result.current.ready).toBe(false);
  await waitFor(() => expect(hook.result.current.backend).toBe("faiss"));
  expect(localStorage.getItem(RETRIEVAL_STORAGE_KEY)).toBe("faiss");
  act(() => hook.result.current.setBackend("qdrant"));
  expect(hook.result.current.backend).toBe("faiss");
});

test("selection persists when both backends are supported", async () => {
  api.getRetrievalCapabilities.mockResolvedValue({ default_backend: "faiss", available_backends: ["faiss", "qdrant"] });
  const hook = renderHook(useRetrievalBackend);
  await waitFor(() => expect(hook.result.current.ready).toBe(true));
  act(() => hook.result.current.setBackend("qdrant"));
  expect(hook.result.current.backend).toBe("qdrant");
  expect(localStorage.getItem(RETRIEVAL_STORAGE_KEY)).toBe("qdrant");
});

test("capability failure keeps selection unavailable", async () => {
  api.getRetrievalCapabilities.mockRejectedValue(new Error("offline"));
  const hook = renderHook(useRetrievalBackend);
  await waitFor(() => expect(hook.result.current.error).toBeTruthy());
  expect(hook.result.current.ready).toBe(false);
});

test.each(["STREAM_DONE", "STREAM_ABORTED", "STREAM_ERROR"])("real progress is per message and settles on %s", (type) => {
  let state = { ...initialChatSessionState, activeConversationId: "c1", messages: [
    { id: "a1", role: "assistant", content: "", pipeline: [] },
    { id: "a2", role: "assistant", content: "", pipeline: [] },
  ] };
  const event = (stage, status) => ({ type: "STREAM_STATUS", scopeId: "c1", messageId: "a2", status: stage,
    progress: { stage, state: status, message: "Truy vấn Qdrant", retrieval_backend: "qdrant", request_id: "r1" } });
  state = chatSessionReducer(state, event("embedding", "completed"));
  state = chatSessionReducer(state, event("dense_search", "started"));
  expect(state.messages[0].pipeline).toEqual([]);
  expect(state.messages[1].pipeline).toHaveLength(2);
  expect(state.messages[1].retrieval_backend).toBe("qdrant");
  expect(chatSessionReducer(state, { ...event("dense_search", "completed"), scopeId: "old" })).toBe(state);
  state = chatSessionReducer(state, { type, messageId: "a2", scopeId: "c1", message: "error" });
  expect(state.messages[1].pipeline.every((item) => item.state !== "started")).toBe(true);
});

test("progress collapses during answer streaming without hiding the answer", () => {
  const pipeline = [{ stage: "dense_search", state: "completed", label: "Truy vấn Qdrant" },
    { stage: "generation", state: "started", label: "Tạo câu trả lời" }];
  const view = render(<RetrievalProgress pipeline={pipeline} backend="qdrant" content="" status="generation" />);
  expect(screen.getByRole("status")).toHaveAttribute("open");
  view.rerender(<RetrievalProgress pipeline={pipeline} backend="qdrant" content="Năm 938" status="streaming" />);
  expect(screen.getByRole("status")).not.toHaveAttribute("open");
  expect(screen.getByText(/Đã truy xuất bằng Qdrant/)).toBeInTheDocument();
  view.rerender(<RetrievalProgress pipeline={pipeline.map((item) => ({ ...item, state: "completed" }))}
    backend="qdrant" content="Năm 938" status="done" />);
  expect(document.querySelectorAll(".pipeline-spinner")).toHaveLength(0);
});
