import assert from "node:assert/strict";
import test from "node:test";
import { progressLabel } from "../src/services/progressLabels.js";

test("Central progress describes reported stages without a timer", () => {
  assert.equal(progressLabel("central_loading"), "Đang khởi động mô hình...");
  assert.equal(progressLabel("central_tools"), "Đang tìm tư liệu...");
  assert.equal(progressLabel("central_answering"), "Đang tổng hợp câu trả lời...");
  assert.equal(progressLabel("processing"), "Đang tìm và tổng hợp tư liệu...");
});
