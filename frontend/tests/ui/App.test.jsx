import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  createConversation: vi.fn(),
  updateConversation: vi.fn(),
  deleteConversation: vi.fn(),
  uploadAttachment: vi.fn(),
  deleteAttachment: vi.fn(),
  streamChat: vi.fn(),
  EVIDENCE_CONTRACT_FAILURE_MESSAGE: "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
}));

const api = await import("../../src/services/api.js");
const { default: App } = await import("../../src/App.jsx");

beforeEach(() => {
  vi.resetAllMocks();
  window.localStorage.clear();
  window.matchMedia = (query) => ({
    matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {},
  });
  Element.prototype.scrollIntoView = vi.fn();
  Element.prototype.scrollTo = vi.fn();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });

  api.listConversations.mockResolvedValue([{ id: "c1", title: "Nhà Trần" }]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
  api.streamChat.mockResolvedValue(undefined);
  api.createConversation.mockResolvedValue({ id: "new", title: "Mới" });
  api.deleteConversation.mockResolvedValue(null);
});

test("hiển thị hội thoại đã có sau khi bootstrap", async () => {
  render(<App />);
  // "Nhà Trần" xuất hiện ở cả mục sidebar lẫn tiêu đề, nên phải nhắm đúng heading.
  expect(await screen.findByRole("heading", { name: "Nhà Trần", level: 1 })).toBeInTheDocument();
});

test.each([["hybrid", "Hybrid"], ["three_llm", "3 LLM"], ["central", "Central Agent"]])("gửi câu hỏi ở mode %s thì câu trả lời hiện trên màn hình", async (mode, label) => {
  const CAU_HOI = "Vì sao nhà Trần suy yếu?";
  const CAU_TRA_LOI = "Nhà Trần suy yếu vì nhiều nguyên nhân.";

  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({ event: "answer_delta", data: { delta: "Nhà Trần suy yếu vì " } });
    onEvent({ event: "answer_delta", data: { delta: "nhiều nguyên nhân." } });
    onEvent({ event: "done", data: {} });
  });

  // Stream xong, frontend đồng bộ lại thread từ backend và ghi đè danh sách tin
  // nhắn. Mock phải trả về bản đã lưu, đúng như backend thật làm, nếu không lần
  // đồng bộ đó sẽ xoá sạch những gì vừa stream ra.
  api.getConversation
    .mockResolvedValueOnce({ messages: [], attachments: [] })
    .mockResolvedValue({
      messages: [
        { id: "m1", role: "user", content: CAU_HOI, sources: [], status: "done" },
        { id: "m2", role: "assistant", content: CAU_TRA_LOI, sources: [], status: "done" },
      ],
      attachments: [],
    });

  render(<App />);
  await screen.findByRole("heading", { name: "Nhà Trần", level: 1 });

  // Có hai textbox trên màn hình: ô tìm kiếm ở sidebar và composer. Nhắm composer.
  await userEvent.click(screen.getByRole("button", { name: /Chọn chế độ trả lời/ }));
  await userEvent.click(screen.getByRole("option", { name: new RegExp(`^${label}`) }));
  expect(screen.getByRole("button", { name: /Chọn chế độ trả lời/ })).toHaveTextContent(label);
  expect(window.localStorage.getItem("vn-history-chat-mode-v2")).toBe(mode);
  const textarea = screen.getByRole("textbox", { name: "Nội dung câu hỏi" });
  await userEvent.type(textarea, CAU_HOI);
  await userEvent.keyboard("{Enter}");

  await waitFor(() => {
    expect(screen.getByText(CAU_TRA_LOI)).toBeInTheDocument();
  });
  expect(api.streamChat.mock.calls[0][0]).toMatchObject({
    conversationId: "c1",
    question: CAU_HOI,
    mode,
  });
});

test.each([false, true])("select/new close only the mobile sidebar (mobile=%s), and active deletion restores the remaining thread", async (mobile) => {
  window.matchMedia = (query) => ({ matches: mobile && query === "(max-width: 839px)" });
  api.listConversations.mockResolvedValue([{ id: "c1", title: "Nhà Trần" }, { id: "c2", title: "Nhà Lý" }]);
  api.getConversation.mockImplementation(async (id) => ({ messages: [{ id, role: "assistant", content: `Thread ${id}`, sources: [] }], attachments: [] }));
  render(<App />);
  await screen.findByRole("heading", { name: "Nhà Trần", level: 1 });
  const sidebar = screen.getByRole("complementary", { name: "Lịch sử trò chuyện" });
  await userEvent.click(screen.getByRole("button", { name: /Nhà Lý.*tin nhắn/ }));
  await screen.findByText("Thread c2");
  expect(sidebar).toHaveAttribute("aria-hidden", String(mobile));
  if (mobile) await userEvent.click(screen.getByRole("button", { name: "Mở thanh bên" }));
  await userEvent.click(screen.getByRole("button", { name: "Cuộc trò chuyện mới", exact: true }));
  await screen.findByRole("heading", { name: "Mới", level: 1 });
  expect(sidebar).toHaveAttribute("aria-hidden", String(mobile));
  if (mobile) await userEvent.click(screen.getByRole("button", { name: "Mở thanh bên" }));
  await userEvent.click(screen.getAllByRole("button", { name: "Tùy chọn cuộc trò chuyện" })[0]);
  await userEvent.click(screen.getByRole("button", { name: "Xóa", exact: true }));
  await userEvent.click(within(screen.getByRole("dialog", { name: "Xóa cuộc trò chuyện?" })).getByRole("button", { name: "Xóa", exact: true }));
  await screen.findByText("Thread c1");
  expect(api.deleteConversation).toHaveBeenCalledWith("new");
  expect(screen.queryByText("Thread c2")).not.toBeInTheDocument();
});

test("pending uploads preserve the draft and block form submit, switching and new conversation", async () => {
  let finishUpload;
  api.uploadAttachment.mockImplementation(() => new Promise((resolve) => { finishUpload = resolve; }));
  render(<App />);
  await screen.findByRole("heading", { name: "Nhà Trần", level: 1 });
  const textarea = screen.getByRole("textbox", { name: "Nội dung câu hỏi" });
  await userEvent.type(textarea, "Keep my draft");
  fireEvent.change(document.querySelector('input[type="file"]'), { target: { files: [new File(["pdf"], "file.pdf", { type: "application/pdf" })] } });
  await waitFor(() => expect(api.uploadAttachment).toHaveBeenCalledTimes(1));
  fireEvent.submit(textarea.closest("form"));
  expect(textarea).toHaveValue("Keep my draft");
  expect(api.streamChat).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Cuộc trò chuyện mới", exact: true })).toBeDisabled();
  finishUpload({ id: "pdf", status: "ready" });
  await waitFor(() => expect(screen.getByRole("button", { name: "Gửi câu hỏi" })).toBeEnabled());
});
