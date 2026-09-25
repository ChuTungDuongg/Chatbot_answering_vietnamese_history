import assert from "node:assert/strict";
import test from "node:test";

import {
  ANSWER_FAILURE_MESSAGE,
  ANSWER_STOPPED_MESSAGE,
} from "../src/config/messages.js";

test("hằng số thông báo import được bằng node trần, không cần Vite", () => {
  assert.equal(ANSWER_FAILURE_MESSAGE, "Không thể hoàn tất câu trả lời.");
  assert.equal(ANSWER_STOPPED_MESSAGE, "Đã dừng tạo câu trả lời.");
});
