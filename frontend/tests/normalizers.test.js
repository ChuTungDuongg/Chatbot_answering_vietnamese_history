import assert from "node:assert/strict";
import test from "node:test";

import { createLocalId } from "../src/state/ids.js";
import {
  getLatestSources,
  getSources,
  normalizeConversationDetail,
  normalizeConversationList,
} from "../src/state/normalizers.js";

test("createLocalId gắn prefix và không trùng nhau", () => {
  const first = createLocalId("user");
  const second = createLocalId("user");
  assert.ok(first.startsWith("user-"));
  assert.notEqual(first, second);
});

test("normalizeConversationList chấp nhận mảng trần và các hình dạng bọc", () => {
  assert.deepEqual(normalizeConversationList([{ id: "a" }]), [{ id: "a" }]);
  assert.deepEqual(normalizeConversationList({ items: [{ id: "b" }] }), [{ id: "b" }]);
  assert.deepEqual(normalizeConversationList({ conversations: [{ id: "c" }] }), [{ id: "c" }]);
  assert.deepEqual(normalizeConversationList(null), []);
});

test("normalizeConversationDetail lấy messages và attachments từ mọi hình dạng", () => {
  const wrapped = normalizeConversationDetail({
    conversation: { id: "a" },
    messages: [{ id: "m1" }],
    attachments: [{ id: "f1" }],
  });
  assert.deepEqual(wrapped, {
    conversation: { id: "a" },
    messages: [{ id: "m1" }],
    attachments: [{ id: "f1" }],
  });

  const nested = normalizeConversationDetail({
    conversation: { id: "b", messages: [{ id: "m2" }], attachments: [] },
  });
  assert.deepEqual(nested.messages, [{ id: "m2" }]);

  const empty = normalizeConversationDetail(null);
  assert.deepEqual(empty, { conversation: {}, messages: [], attachments: [] });
});

test("getSources đọc được các khoá mà backend dùng", () => {
  assert.deepEqual(getSources([{ id: 1 }]), [{ id: 1 }]);
  assert.deepEqual(getSources({ items: [{ id: 2 }] }), [{ id: 2 }]);
  assert.deepEqual(getSources({ sources: [{ id: 3 }] }), [{ id: 3 }]);
  assert.deepEqual(getSources({ final_context: [{ id: 4 }] }), [{ id: 4 }]);
  assert.deepEqual(getSources({ retrieval: { final_context: [{ id: 5 }] } }), [{ id: 5 }]);
  assert.deepEqual(getSources(undefined), []);
});

test("getLatestSources lấy nguồn của câu trả lời gần nhất", () => {
  const messages = [
    { role: "assistant", sources: [{ id: "cũ" }] },
    { role: "user", sources: [] },
    { role: "assistant", sources: [{ id: "mới" }] },
    { role: "assistant", sources: [] },
  ];
  assert.deepEqual(getLatestSources(messages.slice(0, 3)), [{ id: "mới" }]);
  assert.deepEqual(getLatestSources(messages), []);
  assert.deepEqual(getLatestSources([]), []);
});

test("a latest assistant without citations does not inherit an older answer's sources", () => {
  assert.deepEqual(getLatestSources([
    { role: "assistant", sources: [{ id: "old" }] },
    { role: "assistant", sources: [] },
  ]), []);
});
