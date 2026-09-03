import assert from "node:assert/strict";
import test from "node:test";

import { chatSessionReducer, initialChatSessionState } from "../src/state/chatSessionReducer.js";

const ASSISTANT_ID = "assistant-1";

function startedStream() {
  return chatSessionReducer(initialChatSessionState, {
    type: "MESSAGES_APPENDED",
    messages: [
      { id: "user-1", role: "user", content: "Hỏi gì đó", sources: [], status: "done" },
      { id: ASSISTANT_ID, role: "assistant", content: "", sources: [], status: "processing" },
    ],
  });
}

function reduce(state, actions) {
  return actions.reduce(chatSessionReducer, state);
}

test("MESSAGES_APPENDED thêm cặp tin nhắn, xoá lỗi cũ và bật trạng thái processing", () => {
  const state = reduce(initialChatSessionState, [
    { type: "ERROR_SET", message: "lỗi lượt trước" },
    { type: "MESSAGES_APPENDED", messages: [{ id: "u", role: "user", content: "x" }] },
  ]);

  assert.equal(state.messages.length, 1);
  assert.equal(state.error, "");
  assert.equal(state.status, "processing");
  assert.equal(state.streamFailed, false);
});

test("nhiều STREAM_DELTA nối đúng thứ tự vào đúng tin nhắn", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Chiến thắng " },
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Bạch Đằng " },
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "năm 938." },
  ]);

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.equal(assistant.content, "Chiến thắng Bạch Đằng năm 938.");
  assert.equal(assistant.status, "streaming");
  assert.equal(state.status, "streaming");
  assert.equal(state.messages.find((message) => message.id === "user-1").content, "Hỏi gì đó");
});

test("STREAM_SOURCES ghi nguồn vào cả tin nhắn lẫn drawer", () => {
  const sources = [{ id: "s1", text: "trích đoạn" }];
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_SOURCES", messageId: ASSISTANT_ID, sources,
  });

  assert.deepEqual(state.sources, sources);
  assert.deepEqual(state.messages.find((message) => message.id === ASSISTANT_ID).sources, sources);
});

test("STREAM_ERROR loại evidence contract dùng đúng thông báo riêng", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_ERROR",
    messageId: ASSISTANT_ID,
    message: "Evidence critic từ chối",
    kind: "evidence_contract_error",
  });

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.equal(
    assistant.content,
    "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
  );
  assert.equal(state.error, "Evidence critic từ chối");
  assert.equal(state.streamFailed, true);
});

test("STREAM_ERROR loại khác dùng thông báo mặc định", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Backend sập",
  });

  assert.equal(
    state.messages.find((message) => message.id === ASSISTANT_ID).content,
    "Không thể hoàn tất câu trả lời.",
  );
});

test("STREAM_ERROR không nuốt mất phần text đã stream được", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Nửa câu trả lời" },
    { type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Đứt giữa chừng" },
  ]);

  assert.equal(
    state.messages.find((message) => message.id === ASSISTANT_ID).content,
    "Nửa câu trả lời",
  );
});

test("STREAM_DONE đến sau STREAM_ERROR không được ghi đè trạng thái lỗi", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Hỏng" },
    { type: "STREAM_DONE", messageId: ASSISTANT_ID },
  ]);

  assert.equal(state.status, "error");
  assert.equal(state.messages.find((message) => message.id === ASSISTANT_ID).status, "error");
});

test("STREAM_DONE của lượt sạch cho ra trạng thái done", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Xong" },
    { type: "STREAM_DONE", messageId: ASSISTANT_ID },
  ]);

  assert.equal(state.status, "done");
});

test("lượt mới đặt lại cờ lỗi của lượt trước", () => {
  const failed = reduce(startedStream(), [
    { type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Hỏng" },
  ]);
  const fresh = chatSessionReducer(failed, {
    type: "MESSAGES_APPENDED",
    messages: [{ id: "assistant-2", role: "assistant", content: "", status: "processing" }],
  });
  const state = chatSessionReducer(fresh, { type: "STREAM_DONE", messageId: "assistant-2" });

  assert.equal(state.status, "done");
});

test("STREAM_ABORTED giữ lại text đã stream", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Đang viết dở" },
    { type: "STREAM_ABORTED", messageId: ASSISTANT_ID },
  ]);

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.equal(assistant.content, "Đang viết dở");
  assert.equal(assistant.status, "cancelled");
  assert.equal(state.status, "cancelled");
});

test("STREAM_ABORTED khi chưa kịp stream chữ nào thì đặt thông báo đã dừng", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_ABORTED", messageId: ASSISTANT_ID,
  });

  assert.equal(
    state.messages.find((message) => message.id === ASSISTANT_ID).content,
    "Đã dừng tạo câu trả lời.",
  );
});

test("STREAM_DEBUG gắn trace vào tin nhắn mà không đụng nội dung", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Nội dung" },
    { type: "STREAM_DEBUG", messageId: ASSISTANT_ID, trace: { steps: 3 } },
  ]);

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.deepEqual(assistant.debug_trace, { steps: 3 });
  assert.equal(assistant.content, "Nội dung");
});

test("STREAM_SYNCED thay thread bằng bản chính chủ từ backend", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_SYNCED",
    conversations: [{ id: "c1" }],
    detail: {
      messages: [{ id: "m1", role: "assistant", sources: [{ id: "s9" }] }],
      attachments: [{ id: "f1" }],
    },
  });

  assert.deepEqual(state.conversations, [{ id: "c1" }]);
  assert.equal(state.messages.length, 1);
  assert.deepEqual(state.sources, [{ id: "s9" }]);
  assert.deepEqual(state.attachments, [{ id: "f1" }]);
});
