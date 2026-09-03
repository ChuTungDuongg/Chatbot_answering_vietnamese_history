import assert from "node:assert/strict";
import test from "node:test";

import {
  createUploadItems,
  normalizeUploadFile,
  validateUploadSelection,
} from "../src/state/uploadQueue.js";

function makeFile(name, type, size = 1024) {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

test("suy ra MIME từ đuôi file khi trình duyệt không cung cấp", () => {
  const normalized = normalizeUploadFile(makeFile("tulieu.pdf", ""));
  assert.equal(normalized.type, "application/pdf");
  assert.equal(normalized.name, "tulieu.pdf");
});

test("giữ nguyên file khi MIME đã hợp lệ", () => {
  const original = makeFile("anh.png", "image/png");
  assert.equal(normalizeUploadFile(original), original);
});

test("từ chối khi chọn quá 5 file, và báo lỗi số lượng trước mọi lỗi khác", () => {
  const files = Array.from({ length: 6 }, (_, index) => makeFile(`f${index}.exe`, "application/x-msdownload"));
  const result = validateUploadSelection(files);
  assert.equal(result.error, "Mỗi cuộc trò chuyện chỉ nhận tối đa 5 tài liệu.");
  assert.deepEqual(result.files, []);
});

test("từ chối định dạng không hỗ trợ và nêu đúng tên file", () => {
  const result = validateUploadSelection([makeFile("virus.exe", "application/x-msdownload")]);
  assert.equal(
    result.error,
    "Không hỗ trợ định dạng của virus.exe. Chỉ nhận PDF, PNG, JPEG và WebP.",
  );
});

test("từ chối file quá 20 MB", () => {
  const result = validateUploadSelection([makeFile("to.pdf", "application/pdf", 21 * 1024 * 1024)]);
  assert.equal(result.error, "to.pdf vượt quá giới hạn 20 MB.");
});

test("chấp nhận tuyển chọn hợp lệ", () => {
  const result = validateUploadSelection([makeFile("ok.pdf", "application/pdf")]);
  assert.equal(result.error, null);
  assert.equal(result.files.length, 1);
});

test("createUploadItems dựng hàng đợi có id riêng và trạng thái queued", () => {
  const items = createUploadItems([makeFile("a.pdf", "application/pdf", 2048)]);
  assert.equal(items.length, 1);
  assert.equal(items[0].status, "queued");
  assert.equal(items[0].name, "a.pdf");
  assert.equal(items[0].size_bytes, 2048);
  assert.ok(items[0].id.startsWith("upload-"));
  assert.ok(items[0].file instanceof File);
});
