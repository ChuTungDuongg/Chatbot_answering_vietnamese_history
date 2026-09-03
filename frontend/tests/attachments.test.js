import test from "node:test";
import assert from "node:assert/strict";
import { clipboardImages, normalizeUploadFile, validateAttachments, MAX_FILE_SIZE } from "../src/services/attachments.js";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const file = (type, name = "image.png") => new File(["fake image fixture"], name, { type });

test("clipboard image extraction accepts file items only and assigns safe MIME-derived names", () => {
  const items = [
    { kind: "string", type: "text/plain", getAsFile: () => null },
    ...["image/png", "image/jpeg", "image/webp", "image/svg+xml"].map((type) => ({ kind: "file", type, getAsFile: () => file(type, "../../unsafe.png") })),
  ];
  const files = clipboardImages({ items }, 7);
  assert.deepEqual(files.map((f) => f.name), ["clipboard-image-7.png", "clipboard-image-8.jpg", "clipboard-image-9.webp", "clipboard-image-10.unsupported"]);
  assert.equal(validateAttachments(files.slice(0, 3)), "");
  assert.match(validateAttachments(files), /Không hỗ trợ/);
  assert.deepEqual(clipboardImages({ items: items.slice(0, 1) }, 1), []);
});

test("shared validation enforces total count, declared MIME and size for all upload origins", () => {
  assert.match(validateAttachments([file("image/png")], 5), /tối đa 5/);
  assert.match(validateAttachments([{ type: "image/png", size: MAX_FILE_SIZE + 1, name: "huge.png" }]), /20 MB/);
  assert.match(validateAttachments([normalizeUploadFile(file("image/svg+xml", "fake.png"))]), /Không hỗ trợ/);
  assert.equal(normalizeUploadFile(file("", "photo.jpeg")).type, "image/jpeg");
  assert.equal(validateAttachments([file("application/pdf", "document.pdf")]), "");
});

test("ready image-only composer, compact accessible preview and empty user message", async () => {
  const server = await createServer({ server: { middlewareMode: true, hmr: false } });
  try {
    const { default: ChatInput } = await server.ssrLoadModule("/src/components/ChatInput.jsx");
    for (const isUploading of [false, true]) {
      const html = renderToStaticMarkup(React.createElement(ChatInput, { question: "", mode: "central", hasAttachments: true, isUploading }));
      const button = html.match(/<button[^>]*aria-label="Gửi câu hỏi"[^>]*>/)[0];
      assert.equal(button.includes("disabled"), isUploading);
    }
    const { default: Tray } = await server.ssrLoadModule("/src/components/AttachmentTray.jsx");
    const html = renderToStaticMarkup(React.createElement(Tray, { attachments: [], pendingUploads: [{ id: "pending", name: "image.png", preview_url: "blob:fixture", status: "uploading" }] }));
    assert.match(html, /alt="Ảnh đính kèm: image.png"/);
    assert.match(html, /aria-label="Xóa image.png"/);
    const { default: Message } = await server.ssrLoadModule("/src/components/ChatMessage.jsx");
    const message = renderToStaticMarkup(React.createElement(Message, { message: { role: "user", content: "", sources: [{ attachment_id: "a", title: "image.png", source_kind: "attachment" }] } }));
    assert.match(message, /image.png/);
    assert.doesNotMatch(message, /Phân tích nội dung ảnh|Đang tìm/);
  } finally { await server.close(); }
});
