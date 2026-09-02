import { test, expect } from "@playwright/test";

const rawId = "hf_wikipedia_chien_tranh_viet_nam_0001_internal";
const sources = [
  { source_id: rawId, chunk_id: rawId, display_index: 1, title: "Chiến tranh Việt Nam", source_kind: "history", text: "Diễn biến chiến tranh chịu tác động của nhiều yếu tố quân sự, chính trị và ngoại giao. ".repeat(12), url: "https://vi.wikipedia.org/wiki/Chiến_tranh_Việt_Nam" },
  { source_id: "hf_wikipedia_viet_nam_hoa_0002", chunk_id: "hf_wikipedia_viet_nam_hoa_0002", display_index: 2, title: "Việt Nam hóa chiến tranh", source_kind: "wikipedia", text: "Chính sách Việt Nam hóa chiến tranh làm giảm dần mức độ tham chiến trực tiếp của quân đội Mỹ." },
];
const answer = `## Nhiều yếu tố cùng tác động

Thất bại của Mỹ và VNCH có thể được giải thích bởi sự kết hợp của **nhiều yếu tố quân sự, chính trị và chiến lược**. Mỗi yếu tố cần được xem xét trong bối cảnh riêng. [1]

1. **Giới hạn của chiến lược quân sự.** Ưu thế về phương tiện không đồng nghĩa với việc đạt được các mục tiêu chính trị. [1]
2. **Áp lực trong nước.** Mỹ chịu áp lực ngày càng lớn để giảm mức độ can dự trực tiếp. [2]

> Lịch sử cần được đối chiếu từ nhiều nguồn và đặt trong bối cảnh.

| Khía cạnh | Điều cần xem xét |
| --- | --- |
| Quân sự | Mục tiêu và khả năng duy trì chiến lược |
| Chính trị | Sự ủng hộ xã hội và ổn định thể chế |

Các mốc 938, [1945] và [1954] là năm, không phải số nguồn.`;
const question = "Vì sao Mỹ và VNCH lại thua chiến tranh Việt Nam?";

async function setup(page, { empty = false, insufficient = false, holdStream = false } = {}) {
  const content = insufficient ? "Mình chưa tìm thấy đủ bằng chứng đáng tin cậy để trả lời câu hỏi này. Bạn có thể bổ sung thời kỳ hoặc sự kiện cụ thể để mình đối chiếu thêm tư liệu." : answer;
  const assistant = { id: "a1", role: "assistant", mode: "central", status: "done", content, sources: insufficient ? [] : sources,
    debug_trace: { mode: "central", retrieval: { selected: sources }, request: { question, mode: "central" } } };
  const state = { exists: !empty, messages: empty ? [] : [{ id: "u1", role: "user", content: question }, assistant], requests: [], release: null };
  const conversation = { id: "c1", title: "Nhìn lại Chiến tranh Việt Nam", updated_at: "2026-09-03T10:00:00Z" };
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", { value: { writeText: async (text) => { window.__copiedText = text; } }, configurable: true });
  });
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/api/v1/conversations") {
      if (request.method() === "POST") { state.exists = true; return route.fulfill({ json: conversation }); }
      return route.fulfill({ json: state.exists ? [conversation] : [] });
    }
    if (path === "/api/v1/conversations/c1") return route.fulfill({ json: { conversation, messages: state.messages, attachments: [] } });
    if (path === "/api/v1/chat/stream") {
      const payload = request.postDataJSON();
      state.requests.push(payload);
      if (holdStream) await new Promise((resolve) => { state.release = resolve; });
      state.messages = [{ id: "u2", role: "user", content: payload.question }, assistant];
      const events = [["status", { status: "central_answering" }], ["token", { text: content }], ["sources", sources], ["debug_trace", assistant.debug_trace], ["done", {}]];
      return route.fulfill({ contentType: "text/event-stream", body: events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("") });
    }
    return route.fulfill({ status: 404, json: { detail: "Unmocked test endpoint" } });
  });
  await page.goto("/");
  await expect(page.locator(".sidebar-skeleton")).toHaveCount(0);
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.fonts.check('16px "Inter Variable"', "Lịch sử Việt Nam"))).toBe(true);
  return state;
}

async function noOverflow(page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await page.locator(".thread-scroll").evaluate((node) => node.scrollWidth <= node.clientWidth + 1)).toBe(true);
}

test("empty state, composer keyboard behavior and unchanged streaming request", async ({ page }, info) => {
  const state = await setup(page, { empty: true, holdStream: true });
  await expect(page.locator(".suggestion-grid button")).toHaveCount(4);
  await page.screenshot({ path: info.outputPath("empty-dark.png") });
  if (info.project.name !== "desktop") {
    await page.getByRole("button", { name: "Mở thanh bên" }).click();
    const sidebar = page.getByRole("complementary", { name: "Lịch sử trò chuyện" });
    await expect(sidebar).toBeVisible();
    await sidebar.getByRole("button", { name: "Đóng thanh bên" }).click();
  }
  await page.getByRole("button", { name: /Vì sao Cách mạng Tháng Tám thành công/ }).click();
  const input = page.getByRole("textbox", { name: "Nội dung câu hỏi" });
  await expect(input).toBeFocused();
  await expect(input).toHaveValue("Vì sao Cách mạng Tháng Tám thành công?");
  await input.press("Shift+Enter");
  expect(state.requests).toHaveLength(0);
  await input.dispatchEvent("keydown", { key: "Enter", code: "Enter", isComposing: true });
  expect(state.requests).toHaveLength(0);
  await input.fill("Một câu hỏi\n".repeat(10));
  expect(await input.evaluate((node) => node.offsetHeight)).toBeGreaterThan(100);
  await input.fill(question);
  const mode = page.getByRole("button", { name: /Chọn chế độ trả lời/ });
  await mode.press("ArrowDown");
  await page.keyboard.press("End");
  await page.keyboard.press("Enter");
  await expect(mode).toContainText("Central Agent");
  await input.press("Enter");
  await expect(page.getByText("Đang tìm và tổng hợp tư liệu...", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Dừng tạo câu trả lời" })).toBeVisible();
  await expect.poll(() => state.requests.length).toBe(1);
  expect(state.requests[0]).toEqual({ conversation_id: "c1", question, mode: "central", final_k: 6, debug: true });
  state.release();
  await expect(page.getByRole("heading", { name: "Nhiều yếu tố cùng tác động" })).toBeVisible();
  await expect(page.getByRole("button", { name: "2 nguồn", exact: true })).toBeVisible();
  await noOverflow(page);
  await page.reload();
  await expect(page.getByRole("button", { name: /Chọn chế độ trả lời/ })).toContainText("Central Agent");
  await expect(page.locator(".assistant-message")).toContainText("Nhiều yếu tố cùng tác động");
});

test("editorial answer, citations, source drawer focus, copy and debug disclosure", async ({ page }, info) => {
  await setup(page);
  await expect(page.locator(".citation")).toHaveCount(3);
  await expect(page.locator(".message-content").last()).toContainText("[1954]");
  await expect(page.locator(".message-content").last()).not.toContainText("hf_wikipedia");
  await expect(page.locator(".developer-trace")).not.toHaveAttribute("open", "");
  expect(await page.locator("body").innerText()).not.toContain(rawId);
  await page.locator(".thread-scroll").evaluate((node) => { node.scrollTop = 0; });
  await page.screenshot({ path: info.outputPath("conversation-dark.png") });
  const citation = page.getByRole("button", { name: "Nguồn 2: Việt Nam hóa chiến tranh" });
  await citation.click();
  const drawer = page.getByRole("dialog", { name: "Nguồn của câu trả lời" });
  await expect(drawer).toBeVisible();
  await expect(drawer.locator('[data-source-index="2"]')).toHaveAttribute("open", "");
  await expect(drawer.getByRole("button", { name: "Đóng nguồn" })).toBeFocused();
  expect(await drawer.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath("sources-dark.png") });
  await drawer.locator('[data-source-index="1"] summary').click();
  expect(await drawer.locator(".source-drawer-body").evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await drawer.getByRole("button", { name: "Đóng nguồn" }).focus();
  await page.keyboard.press("Shift+Tab");
  expect(await drawer.evaluate((node) => node.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(citation).toBeFocused();
  await page.getByRole("button", { name: "Sao chép câu trả lời" }).click();
  await expect(page.getByText("Đã sao chép", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => window.__copiedText)).toBe(answer);
  await page.getByText("Agent trace", { exact: true }).click();
  await expect(page.locator(".trace-panel-body")).toBeVisible();
  expect(await page.locator("body").innerText()).toContain(rawId);
  await noOverflow(page);
  await page.getByText("Agent trace", { exact: true }).click();
  await page.getByRole("button", { name: /Chuyển sang giao diện sáng/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.locator(".thread-scroll").evaluate((node) => { node.scrollTop = 0; });
  await page.screenshot({ path: info.outputPath("conversation-light.png") });
});

test("insufficient evidence has an intentional muted panel", async ({ page }, info) => {
  await setup(page, { insufficient: true });
  await expect(page.locator(".insufficient-panel")).toContainText("Chưa đủ tư liệu để trả lời chắc chắn.");
  await expect(page.locator(".error-toast")).toHaveCount(0);
  await noOverflow(page);
  await page.screenshot({ path: info.outputPath("insufficient-dark.png") });
});
