import { test, expect } from "@playwright/test";

async function setup(page, { holdUpload = false, failUpload = false, empty = false } = {}) {
  const conversation = { id: "c-clipboard", title: "Đọc tư liệu từ ảnh", updated_at: "2026-09-03T10:00:00Z" };
  const state = { attachments: [], uploads: [], requests: [], deleted: [], exists: !empty, release: null,
    messages: empty ? [] : Array.from({ length: 4 }, (_, i) => ({ id: `m-${i}`, role: i % 2 ? "assistant" : "user", content: i % 2 ? "Một phần trả lời lịch sử có nhiều nội dung. ".repeat(50) : `Câu hỏi ${i / 2 + 1}`, sources: [], status: "done" })) };
  await page.addInitScript(() => {
    window.__revoked = [];
    const revoke = URL.revokeObjectURL.bind(URL);
    URL.revokeObjectURL = (url) => { window.__revoked.push(url); revoke(url); };
  });
  await page.route("**/api/v1/**", async (route) => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (path === "/api/v1/conversations") {
      if (req.method() === "POST") { state.exists = true; return route.fulfill({ json: conversation }); }
      return route.fulfill({ json: state.exists ? [conversation] : [] });
    }
    if (path === `/api/v1/conversations/${conversation.id}`) return route.fulfill({ json: { conversation, messages: state.messages, attachments: state.attachments } });
    if (path.endsWith("/attachments") && req.method() === "POST") {
      const body = req.postDataBuffer().toString("utf8");
      state.uploads.push({ body, headers: req.headers() });
      if (holdUpload) await new Promise((resolve) => { state.release = resolve; });
      const filename = body.match(/filename="([^"]+)"/)[1];
      const attachment = { id: `image-${state.uploads.length}`, filename, status: failUpload ? "failed" : "ready", chunk_count: failUpload ? 0 : 1,
        error: failUpload ? "Không thể đọc chữ trong ảnh." : null, mime_type: body.match(/Content-Type: (image\/[^\r\n]+)/i)?.[1], size_bytes: 100 };
      state.attachments.push(attachment);
      if (failUpload) return route.fulfill({ status: 422, json: { detail: attachment.error } });
      return route.fulfill({ status: 201, json: { attachment } });
    }
    if (req.method() === "DELETE" && path.includes("/attachments/")) {
      const id = path.split("/").pop(); state.deleted.push(id);
      state.attachments = state.attachments.filter((item) => item.id !== id);
      return route.fulfill({ status: 204 });
    }
    if (path === "/api/v1/chat/stream") {
      const payload = req.postDataJSON(); state.requests.push(payload);
      state.messages.push({ id: `u-${state.requests.length}`, role: "user", content: payload.question,
        sources: state.attachments.filter((item) => payload.attachment_ids?.includes(item.id)).map((item) => ({ attachment_id: item.id, title: item.filename, source_kind: "attachment" })) },
      { id: `a-${state.requests.length}`, role: "assistant", content: "Theo ảnh, tài liệu đề cập đến lịch sử Việt Nam.", status: "done" });
      return route.fulfill({ contentType: "text/event-stream", body: 'event: done\ndata: {}\n\n' });
    }
    return route.fulfill({ status: 404, json: { detail: "Unmocked local test endpoint" } });
  });
  await page.goto("/");
  await expect(page.locator(".sidebar-skeleton")).toHaveCount(0);
  return state;
}

async function paste(page, types = ["image/png"], text = "", oversized = false) {
  return page.getByRole("textbox", { name: "Nội dung câu hỏi" }).evaluate(async (node, { types, text, oversized }) => {
    const transfer = new DataTransfer();
    if (text) transfer.setData("text/plain", text);
    for (const type of types) {
      const canvas = document.createElement("canvas"); canvas.width = canvas.height = 12;
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, type));
      const file = new File(oversized ? [new Uint8Array(21 * 1024 * 1024)] : [blob], "clipboard-image", { type });
      transfer.items.add(file);
    }
    const event = new ClipboardEvent("paste", { clipboardData: transfer, bubbles: true, cancelable: true });
    node.dispatchEvent(event);
    return event.defaultPrevented;
  }, { types, text, oversized });
}

async function docked(page) {
  const metrics = await page.evaluate(() => ({ body: document.documentElement.scrollHeight, height: innerHeight, width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth, footer: document.querySelector(".composer-shell")?.getBoundingClientRect().bottom }));
  expect(metrics.body).toBeLessThanOrEqual(metrics.height + 1);
  expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.width);
  expect(metrics.footer).toBeLessThanOrEqual(metrics.height + 1);
  expect(metrics.footer).toBeGreaterThan(metrics.height - 4);
}

test("text-only and mixed text/image paste retain native text behavior", async ({ page, context }, info) => {
  const state = await setup(page, { empty: true });
  const input = page.getByRole("textbox", { name: "Nội dung câu hỏi" });
  expect(await paste(page, [], "Câu hỏi lịch sử")).toBe(false);
  expect(state.uploads).toHaveLength(0);
  if (info.project.name === "desktop") {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.evaluate(async () => {
      const canvas = document.createElement("canvas"); canvas.width = canvas.height = 12;
      const image = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      await navigator.clipboard.write([new ClipboardItem({ "image/png": image, "text/plain": new Blob(["Giải thích đoạn này"], { type: "text/plain" }) })]);
    });
    await input.focus(); await input.press("Control+V");
    await expect(input).toHaveValue("Giải thích đoạn này");
  } else {
    await input.fill("Giải thích đoạn này");
    expect(await paste(page, ["image/png"], "Giải thích đoạn này")).toBe(false);
  }
  await expect(page.locator(".attachment-preview")).toHaveCount(1);
  await expect(input).toHaveValue("Giải thích đoạn này");
  await expect(page.getByRole("button", { name: "Gửi câu hỏi" })).toBeEnabled();
  await page.getByRole("button", { name: "Gửi câu hỏi" }).click();
  await expect.poll(() => state.requests.length).toBe(1);
  expect(state.requests[0].question).toBe("Giải thích đoạn này");
  expect(state.requests[0].attachment_ids).toEqual(["image-1"]);
});

test("PNG and JPEG previews share multipart pipeline; image-only third turn stays docked", async ({ page }, info) => {
  const state = await setup(page);
  await paste(page, ["image/png", "image/jpeg"]);
  await expect(page.locator(".attachment-preview")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Gửi câu hỏi" })).toBeEnabled();
  expect(state.uploads).toHaveLength(2);
  expect(state.uploads[0].headers["content-type"]).toContain("multipart/form-data");
  expect(state.uploads[0].body).toContain('filename="clipboard-image-1.png"');
  expect(state.uploads[1].body).toContain('filename="clipboard-image-2.jpg"');
  expect(state.uploads[0].body).toContain('name="upload_origin"\r\n\r\nclipboard');
  await docked(page);
  await page.screenshot({ path: info.outputPath("clipboard-third-turn.png") });
  await page.getByRole("button", { name: "Gửi câu hỏi" }).click();
  await expect.poll(() => state.requests.length).toBe(1);
  expect(state.requests[0].question).toBe("");
  expect(state.requests[0].attachment_ids).toEqual(["image-1", "image-2"]);
  expect(JSON.stringify(state.requests[0])).not.toMatch(/base64|data:image|blob:|Phân tích nội dung ảnh/);
  await expect(page.locator(".user-message").last()).toContainText("clipboard-image-1.png");
  await expect(page.locator(".assistant-message").last()).toContainText("Theo ảnh");
  await docked(page);
  await page.setViewportSize({ width: info.project.use.viewport.width, height: 660 });
  await docked(page);
});

test("removing a pending image cleans up the eventual server record and object URL", async ({ page }) => {
  const state = await setup(page, { holdUpload: true });
  await paste(page);
  await expect.poll(() => state.uploads.length).toBe(1);
  await expect(page.getByRole("button", { name: "Gửi câu hỏi" })).toBeDisabled();
  await page.getByRole("button", { name: "Xóa clipboard-image-1.png" }).click();
  await expect(page.locator(".attachment-preview")).toHaveCount(0);
  state.release();
  await expect.poll(() => state.deleted.length).toBe(1);
  await expect.poll(() => page.evaluate(() => window.__revoked.length)).toBe(1);
  await page.getByRole("textbox", { name: "Nội dung câu hỏi" }).fill("Câu hỏi không kèm ảnh");
  await page.getByRole("button", { name: "Gửi câu hỏi" }).click();
  await expect.poll(() => state.requests.length).toBe(1);
  expect(state.requests[0].attachment_ids).toBeUndefined();
});

test("multiple images have a bounded tray; removing a ready image releases its URL and ID", async ({ page }) => {
  const state = await setup(page);
  await paste(page, Array(5).fill("image/png"));
  await expect(page.getByRole("button", { name: "Gửi câu hỏi" })).toBeEnabled();
  await expect(page.locator(".attachment-preview")).toHaveCount(5);
  expect(await page.locator(".attachment-tray").evaluate((node) => node.getBoundingClientRect().height)).toBeLessThanOrEqual(86);
  await docked(page);
  await paste(page);
  await expect(page.getByRole("alert")).toContainText("tối đa 5");
  expect(state.uploads).toHaveLength(5);
  await page.getByRole("button", { name: "Xóa clipboard-image-1.png" }).click();
  await expect(page.locator(".attachment-preview")).toHaveCount(4);
  await page.getByRole("button", { name: "Gửi câu hỏi" }).click();
  await expect.poll(() => state.requests.length).toBe(1);
  expect(state.requests[0].attachment_ids).not.toContain("image-1");
  expect(await page.evaluate(() => window.__revoked.length)).toBe(1);
});

test("unsupported, oversized and excessive clipboard data never upload", async ({ page }) => {
  const state = await setup(page);
  await paste(page, ["image/svg+xml"]);
  await expect(page.getByRole("alert")).toContainText("Không hỗ trợ");
  await paste(page, ["image/png"], "", true);
  await expect(page.getByRole("alert")).toContainText("20 MB");
  await paste(page, Array(6).fill("image/png"));
  await expect(page.getByRole("alert")).toContainText("tối đa 5");
  expect(state.uploads).toHaveLength(0);
  await expect(page.locator(".attachment-preview")).toHaveCount(0);
});

test("failed OCR is shown honestly, remains removable and cannot send image-only", async ({ page }) => {
  const state = await setup(page, { failUpload: true });
  await paste(page);
  await expect(page.getByRole("alert")).toContainText("Không thể đọc chữ");
  await expect(page.locator(".attachment-failed")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Gửi câu hỏi" })).toBeDisabled();
  await page.getByRole("button", { name: "Xóa clipboard-image-1.png" }).click();
  await expect.poll(() => state.deleted.length).toBe(1);
});
