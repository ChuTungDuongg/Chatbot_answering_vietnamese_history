import assert from "node:assert/strict";
import test from "node:test";

import { chatSessionReducer, initialChatSessionState } from "../src/state/chatSessionReducer.js";

function reduce(actions, state = initialChatSessionState) {
  return actions.reduce(chatSessionReducer, state);
}

const QUEUED = [
  { id: "upload-1", name: "a.pdf", status: "queued" },
  { id: "upload-2", name: "b.pdf", status: "queued" },
];

test("UPLOAD_QUEUED đưa file vào hàng đợi và xoá lỗi cũ", () => {
  const state = reduce([
    { type: "ERROR_SET", message: "lỗi cũ" },
    { type: "UPLOAD_QUEUED", items: QUEUED },
  ]);

  assert.equal(state.pendingUploads.length, 2);
  assert.equal(state.error, "");
});

test("UPLOAD_PROGRESS chỉ đổi trạng thái đúng một file", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_PROGRESS", id: "upload-2", status: "processing" },
  ]);

  assert.deepEqual(state.pendingUploads.map((item) => item.status), ["queued", "processing"]);
});

test("UPLOAD_SETTLED thành công: rời hàng đợi và vào danh sách tài liệu", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_SETTLED", id: "upload-1", attachment: { id: "att-1", name: "a.pdf" } },
  ]);

  assert.deepEqual(state.pendingUploads.map((item) => item.id), ["upload-2"]);
  assert.deepEqual(state.attachments, [{ id: "att-1", name: "a.pdf" }]);
});

test("UPLOAD_SETTLED không nhân đôi tài liệu khi backend trả lại cùng id", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_SETTLED", id: "upload-1", attachment: { id: "att-1", name: "cũ" } },
    { type: "UPLOAD_SETTLED", id: "upload-2", attachment: { id: "att-1", name: "mới" } },
  ]);

  assert.equal(state.attachments.length, 1);
  assert.equal(state.attachments[0].name, "mới");
});

test("UPLOAD_SETTLED thất bại: rời hàng đợi và ghi lỗi, không thêm tài liệu", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_SETTLED", id: "upload-1", error: "OCR hỏng" },
  ]);

  assert.deepEqual(state.pendingUploads.map((item) => item.id), ["upload-2"]);
  assert.deepEqual(state.attachments, []);
  assert.equal(state.error, "OCR hỏng");
});

test("ATTACHMENT_REMOVED bỏ đúng tài liệu", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: [] },
    { type: "UPLOAD_SETTLED", id: "x", attachment: { id: "att-1" } },
    { type: "UPLOAD_SETTLED", id: "y", attachment: { id: "att-2" } },
    { type: "ATTACHMENT_REMOVED", attachmentId: "att-1" },
  ]);

  assert.deepEqual(state.attachments.map((item) => item.id), ["att-2"]);
});

test("ATTACHMENTS_SYNCED chỉ chạm conversations và attachments, không đụng thread", () => {
  const withThread = reduce([{
    type: "MESSAGES_APPENDED",
    messages: [{ id: "m1", role: "assistant", content: "giữ nguyên", sources: [{ id: "s1" }] }],
  }]);

  const state = chatSessionReducer(withThread, {
    type: "ATTACHMENTS_SYNCED",
    conversations: [{ id: "c1" }],
    attachments: [{ id: "att-9" }],
  });

  assert.deepEqual(state.conversations, [{ id: "c1" }]);
  assert.deepEqual(state.attachments, [{ id: "att-9" }]);
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].content, "giữ nguyên");
});
