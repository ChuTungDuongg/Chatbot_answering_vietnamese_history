import { render, screen, waitFor } from "@testing-library/react";
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
  window.localStorage.clear();
  window.matchMedia = (query) => ({
    matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {},
  });
  Element.prototype.scrollIntoView = vi.fn();

  api.listConversations.mockResolvedValue([{ id: "c1", title: "Nhà Trần" }]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
  api.streamChat.mockResolvedValue(undefined);
});

test("hiển thị hội thoại đã có sau khi bootstrap", async () => {
  render(<App />);
  // "Nhà Trần" xuất hiện ở cả mục sidebar lẫn tiêu đề, nên phải nhắm đúng heading.
  expect(await screen.findByRole("heading", { name: "Nhà Trần", level: 1 })).toBeInTheDocument();
});

test("gửi câu hỏi thì câu trả lời hiện trên màn hình", async () => {
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
  const textarea = screen.getByPlaceholderText("Hỏi về lịch sử Việt Nam hoặc tài liệu đã tải lên");
  await userEvent.type(textarea, CAU_HOI);
  await userEvent.keyboard("{Enter}");

  await waitFor(() => {
    expect(screen.getByText(CAU_TRA_LOI)).toBeInTheDocument();
  });
  expect(api.streamChat.mock.calls[0][0]).toMatchObject({
    conversationId: "c1",
    question: CAU_HOI,
  });
});
