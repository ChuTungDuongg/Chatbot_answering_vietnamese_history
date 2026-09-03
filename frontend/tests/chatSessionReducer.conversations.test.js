import assert from "node:assert/strict";
import test from "node:test";

import {
  chatSessionReducer,
  initialChatSessionState,
  isRunningStatus,
} from "../src/state/chatSessionReducer.js";

function reduce(actions, state = initialChatSessionState) {
  return actions.reduce(chatSessionReducer, state);
}

test("BOOTSTRAPPED tắt cờ đang tải và nạp hội thoại đầu tiên", () => {
  const state = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "c1" }, { id: "c2" }],
    activeConversationId: "c1",
    detail: {
      messages: [{ id: "m1", role: "assistant", sources: [{ id: "s1" }] }],
      attachments: [{ id: "f1" }],
    },
  }]);

  assert.equal(state.isLoadingConversations, false);
  assert.equal(state.activeConversationId, "c1");
  assert.deepEqual(state.attachments, [{ id: "f1" }]);
  assert.deepEqual(state.sources, [{ id: "s1" }]);
});

test("BOOTSTRAPPED khi chưa có hội thoại nào thì không đụng tới thread", () => {
  const state = reduce([{ type: "BOOTSTRAPPED", conversations: [] }]);
  assert.equal(state.isLoadingConversations, false);
  assert.equal(state.activeConversationId, null);
  assert.deepEqual(state.messages, []);
});

test("BOOTSTRAP_FAILED vẫn phải tắt cờ đang tải, nếu không spinner kẹt vĩnh viễn", () => {
  const state = reduce([{ type: "BOOTSTRAP_FAILED", message: "Sập mạng" }]);
  assert.equal(state.isLoadingConversations, false);
  assert.equal(state.error, "Sập mạng");
});

test("CONVERSATION_LOAD_FAILED tắt cờ đang tải và đặt lỗi", () => {
  const state = reduce([
    { type: "CONVERSATION_LOADING" },
    { type: "CONVERSATION_LOAD_FAILED", message: "Hỏng rồi" },
  ]);
  assert.equal(state.isLoadingConversation, false);
  assert.equal(state.error, "Hỏng rồi");
});

test("CONVERSATION_LOADED đổi thread và đặt lại trạng thái stream", () => {
  const state = reduce([
    { type: "STREAM_ERROR", messageId: "x", message: "lỗi cũ" },
    {
      type: "CONVERSATION_LOADED",
      conversationId: "c9",
      detail: { messages: [{ id: "m", role: "assistant", sources: [] }], attachments: [] },
    },
  ]);

  assert.equal(state.activeConversationId, "c9");
  assert.equal(state.status, "idle");
  assert.equal(state.streamFailed, false);
  assert.equal(state.isLoadingConversation, false);
});

test("CONVERSATION_CREATED đưa hội thoại lên đầu và dọn sạch màn hình", () => {
  const state = reduce([
    { type: "BOOTSTRAPPED", conversations: [{ id: "cũ" }] },
    { type: "CONVERSATION_CREATED", conversation: { id: "mới" } },
  ]);

  assert.deepEqual(state.conversations.map((item) => item.id), ["mới", "cũ"]);
  assert.equal(state.activeConversationId, "mới");
  assert.deepEqual(state.messages, []);
  assert.deepEqual(state.attachments, []);
  assert.deepEqual(state.sources, []);
});

test("CONVERSATION_CREATED không nhân đôi khi backend trả lại id đã có", () => {
  const state = reduce([
    { type: "BOOTSTRAPPED", conversations: [{ id: "a" }, { id: "b" }] },
    { type: "CONVERSATION_CREATED", conversation: { id: "b", title: "Đổi tên" } },
  ]);
  assert.deepEqual(state.conversations.map((item) => item.id), ["b", "a"]);
});

test("CONVERSATION_RENAMED chỉ vá đúng một mục", () => {
  const state = reduce([
    { type: "BOOTSTRAPPED", conversations: [{ id: "a", title: "A" }, { id: "b", title: "B" }] },
    { type: "CONVERSATION_RENAMED", conversationId: "b", patch: { title: "B mới" } },
  ]);
  assert.deepEqual(state.conversations.map((item) => item.title), ["A", "B mới"]);
});

test("xoá hội thoại đang mở thì dọn thread", () => {
  const start = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "a" }, { id: "b" }],
    activeConversationId: "a",
    detail: { messages: [{ id: "m", role: "assistant", sources: [{ id: "s" }] }], attachments: [{ id: "f" }] },
  }]);

  const state = chatSessionReducer(start, { type: "CONVERSATION_DELETED", conversationId: "a" });

  assert.deepEqual(state.conversations.map((item) => item.id), ["b"]);
  assert.equal(state.activeConversationId, null);
  assert.deepEqual(state.messages, []);
  assert.deepEqual(state.attachments, []);
  assert.deepEqual(state.sources, []);
});

test("xoá hội thoại khác thì thread đang mở không suy suyển", () => {
  const start = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "a" }, { id: "b" }],
    activeConversationId: "a",
    detail: { messages: [{ id: "m", role: "assistant", sources: [] }], attachments: [] },
  }]);

  const state = chatSessionReducer(start, { type: "CONVERSATION_DELETED", conversationId: "b" });

  assert.equal(state.activeConversationId, "a");
  assert.equal(state.messages.length, 1);
});

test("SOURCES_SHOWN chỉ đổi nguồn đang xem, không đụng tin nhắn", () => {
  const start = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "a" }],
    activeConversationId: "a",
    detail: { messages: [{ id: "m", role: "assistant", sources: [{ id: "mới" }] }], attachments: [] },
  }]);

  const state = chatSessionReducer(start, { type: "SOURCES_SHOWN", sources: [{ id: "cũ" }] });

  assert.deepEqual(state.sources, [{ id: "cũ" }]);
  assert.deepEqual(state.messages[0].sources, [{ id: "mới" }]);
});

test("isRunningStatus nhận diện đúng các trạng thái đang chạy", () => {
  assert.equal(isRunningStatus("streaming"), true);
  assert.equal(isRunningStatus("central_tools"), true);
  assert.equal(isRunningStatus("idle"), false);
  assert.equal(isRunningStatus("done"), false);
  assert.equal(isRunningStatus("error"), false);
});

test("action lạ không làm thay đổi state", () => {
  const state = chatSessionReducer(initialChatSessionState, { type: "KHÔNG_TỒN_TẠI" });
  assert.equal(state, initialChatSessionState);
});
