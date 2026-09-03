import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  uploadAttachment: vi.fn(),
  deleteAttachment: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
}));

const api = await import("../../src/services/api.js");
const { useAttachments } = await import("../../src/hooks/useAttachments.js");

function makeFile(name, type, size = 1024) {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function setup({ isRunning = false, activeConversationId = "c1" } = {}) {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockResolvedValue("c1");
  const { result } = renderHook(() => useAttachments({
    dispatch, activeConversationId, isRunning, ensureActiveConversation,
  }));
  return { dispatch, ensureActiveConversation, result };
}

beforeEach(() => {
  api.uploadAttachment.mockResolvedValue({ attachment: { id: "att-1", name: "a.pdf" } });
  api.deleteAttachment.mockResolvedValue({});
  api.listConversations.mockResolvedValue([]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
});

test("từ chối file sai định dạng TRƯỚC khi gọi mạng", async () => {
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("virus.exe", "application/x-msdownload")]);
  });

  expect(api.uploadAttachment).not.toHaveBeenCalled();
  expect(dispatch).toHaveBeenCalledWith({
    type: "ERROR_SET",
    message: "Không hỗ trợ định dạng của virus.exe. Chỉ nhận PDF, PNG, JPEG và WebP.",
  });
});

test("từ chối file quá 20 MB trước khi gọi mạng", async () => {
  const { result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("to.pdf", "application/pdf", 21 * 1024 * 1024)]);
  });

  expect(api.uploadAttachment).not.toHaveBeenCalled();
});

test("không upload khi đang stream", async () => {
  const { result } = setup({ isRunning: true });

  await act(async () => {
    await result.current.upload([makeFile("ok.pdf", "application/pdf")]);
  });

  expect(api.uploadAttachment).not.toHaveBeenCalled();
});

test("upload hợp lệ đi qua đủ chuỗi action queued, processing, settled", async () => {
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("ok.pdf", "application/pdf")]);
  });

  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("UPLOAD_QUEUED");
  expect(types).toContain("UPLOAD_PROGRESS");
  expect(types).toContain("UPLOAD_SETTLED");
  expect(types).toContain("ATTACHMENTS_SYNCED");
  expect(api.uploadAttachment).toHaveBeenCalledTimes(1);
});

test("upload hỏng vẫn phải rời hàng đợi kèm lỗi", async () => {
  api.uploadAttachment.mockRejectedValue(new Error("OCR hỏng"));
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("ok.pdf", "application/pdf")]);
  });

  const settled = dispatch.mock.calls
    .map(([action]) => action)
    .find((action) => action.type === "UPLOAD_SETTLED");

  expect(settled.error).toBe("OCR hỏng");
  expect(settled.attachment).toBeUndefined();
});

test("remove xoá tài liệu rồi làm mới danh sách hội thoại", async () => {
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.remove("att-1");
  });

  expect(api.deleteAttachment).toHaveBeenCalledWith("c1", "att-1");
  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("ATTACHMENT_REMOVED");
  expect(types).toContain("CONVERSATIONS_SET");
});

test("remove không làm gì khi đang stream", async () => {
  const { result } = setup({ isRunning: true });

  await act(async () => {
    await result.current.remove("att-1");
  });

  expect(api.deleteAttachment).not.toHaveBeenCalled();
});
