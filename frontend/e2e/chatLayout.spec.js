import { test, expect } from "@playwright/test";

const question = (turn) => `Câu hỏi ${turn}: Những yếu tố nào tác động đến sự kiện này?`;
const answer = (turn, paragraphs = 6) => Array.from({ length: paragraphs }, (_, index) =>
  `Đoạn ${index + 1} của câu trả lời ${turn}. Việc tìm hiểu lịch sử cần đặt sự kiện trong bối cảnh, đối chiếu tư liệu và xem xét hoạt động của các lực lượng tham gia. Những yếu tố chính trị, quân sự và xã hội có quan hệ với nhau.`).join("\n\n");

// Real frontend/SSE parser with deterministic incremental responses; no backend.
async function setupChat(page, { initialTurns = 0 } = {}) {
  const conversation = { id: "layout", title: "Cuộc trò chuyện nhiều lượt" };
  const messages = [];
  for (let turn = 1; turn <= initialTurns; turn += 1) {
    messages.push({ id: `u${turn}`, role: "user", content: question(turn), status: "done" },
      { id: `a${turn}`, role: "assistant", content: answer(turn), mode: "central", status: "done", sources: [] });
  }
  await page.addInitScript(() => {
    const originalFetch = window.fetch;
    window.__layoutStream = { requests: [], active: null };
    window.fetch = (url, options) => {
      if (!String(url).endsWith("/api/v1/chat/stream")) return originalFetch(url, options);
      window.__layoutStream.requests.push(JSON.parse(options.body));
      const stream = new ReadableStream({ start(controller) { window.__layoutStream.active = controller; } });
      return Promise.resolve(new Response(stream, { headers: { "Content-Type": "text/event-stream" } }));
    };
  });
  await page.route("**/api/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/v1/conversations") return route.fulfill({ json: [conversation] });
    if (path === "/api/v1/conversations/layout") return route.fulfill({ json: { conversation, messages, attachments: [] } });
    return route.fulfill({ status: 404, json: { detail: "Unexpected fixture request" } });
  });
  await page.goto("/");
  await expect(page.locator(".sidebar-skeleton")).toHaveCount(0);
  await page.evaluate(() => document.fonts.ready);
  let pending = "";
  const emit = async (event, data) => {
    if (event === "answer_delta") pending += data.delta;
    await page.evaluate(({ event, data }) => {
      window.__layoutStream.active.enqueue(new TextEncoder().encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
    }, { event, data });
  };
  return {
    emit,
    async finish(content = "") {
      if (content) await emit("answer_delta", { delta: content });
      const payload = await page.evaluate(() => window.__layoutStream.requests.at(-1));
      const turn = messages.length / 2 + 1;
      messages.push({ id: `u${turn}`, role: "user", content: payload.question, status: "done" },
        { id: `a${turn}`, role: "assistant", content: pending, mode: "central", status: "done", sources: [] });
      pending = "";
      await emit("done", {});
      await page.evaluate(() => { window.__layoutStream.active.close(); window.__layoutStream.active = null; });
      await expect(page.getByRole("button", { name: "Dừng tạo câu trả lời" })).toHaveCount(0);
      await expect(page.locator(".assistant-message")).toHaveCount(turn);
    },
  };
}

async function submit(page, turn) {
  const before = await page.evaluate(() => window.__layoutStream.requests.length);
  const input = page.getByRole("textbox", { name: "Nội dung câu hỏi" });
  await input.fill(question(turn));
  await input.press("Enter");
  await expect.poll(() => page.evaluate(() => window.__layoutStream.requests.length)).toBe(before + 1);
  await expect(page.locator(".assistant-message").last().locator(".thinking-state")).toBeVisible();
}

async function geometry(page) {
  return page.evaluate(() => {
    const box = (selector) => {
      const node = document.querySelector(selector);
      const rect = node.getBoundingClientRect();
      return { top: rect.top, bottom: rect.bottom, height: rect.height,
        scrollHeight: node.scrollHeight, clientHeight: node.clientHeight, scrollTop: node.scrollTop,
        scrollWidth: node.scrollWidth, clientWidth: node.clientWidth };
    };
    return { height: innerHeight, pageY: scrollY,
      document: box("html"), body: box("body"), shell: box(".app-shell"), workspace: box(".chat-workspace"),
      sidebar: box(".chat-sidebar"), scroller: box(".thread-scroll"), composer: box(".composer-shell") };
  });
}

async function assertDocked(page) {
  await expect.poll(async () => {
    const g = await geometry(page);
    return Math.abs(g.composer.bottom - g.height);
  }).toBeLessThanOrEqual(1).catch(async (error) => {
    throw new Error(`${error.message}\nLayout: ${JSON.stringify(await geometry(page), null, 2)}`);
  });
  const g = await geometry(page);
  expect(g.pageY).toBe(0);
  for (const key of ["document", "body", "shell", "workspace"]) {
    expect(g[key].top, `${key}: ${JSON.stringify(g[key])}`).toBe(0);
    expect(g[key].scrollTop, key).toBe(0);
    expect(g[key].scrollHeight, key).toBeLessThanOrEqual(g[key].clientHeight + 1);
    expect(g[key].scrollWidth, key).toBeLessThanOrEqual(g[key].clientWidth + 1);
  }
  expect(g.scroller.bottom).toBeCloseTo(g.composer.top, 0);
  expect(g.scroller.scrollWidth).toBeLessThanOrEqual(g.scroller.clientWidth + 1);
  expect(g.sidebar.height).toBe(g.shell.height);
  expect(g.composer.top).toBeGreaterThan(0);
  expect(await page.locator(".thread-scroll .composer-shell").count()).toBe(0);
  expect(await page.locator(".thread-content").evaluate((node) => getComputedStyle(node).overflowY)).toBe("visible");
}

async function atBottom(page) {
  await expect.poll(() => page.locator(".thread-scroll").evaluate((node) => node.scrollHeight - node.scrollTop - node.clientHeight)).toBeLessThanOrEqual(2);
}

test("five turns keep third-turn loading and completed response inside the viewport", async ({ page }, info) => {
  const fixture = await setupChat(page);
  for (let turn = 1; turn <= 5; turn += 1) {
    await submit(page, turn);
    if (turn === 3) {
      await page.screenshot({ path: info.outputPath("third-turn-loading.png") });
      await info.attach("third-turn-geometry", { body: JSON.stringify(await geometry(page), null, 2), contentType: "application/json" });
      await assertDocked(page);
      await atBottom(page);
      const loading = await page.locator(".assistant-message").last().boundingBox();
      expect(loading.height).toBeLessThan(130);
      const g = await geometry(page);
      expect(loading.y).toBeGreaterThanOrEqual(g.scroller.top);
      expect(loading.y + loading.height).toBeLessThan(g.composer.top);
      expect(g.scroller.scrollHeight).toBeGreaterThan(g.scroller.clientHeight);
    }
    await fixture.finish(answer(turn));
    if (turn >= 3) { await assertDocked(page); await atBottom(page); }
    if (turn === 3) await page.screenshot({ path: info.outputPath("third-turn-completed.png") });
  }
  await page.locator(".thread-scroll").evaluate((node) => { node.scrollTop = 0; });
  await expect(page.locator(".user-message").first()).toBeInViewport();
  await assertDocked(page);
});

test("long response stays scrollable across viewport resizing and sidebar changes", async ({ page }, info) => {
  const fixture = await setupChat(page, { initialTurns: 2 });
  await submit(page, 3);
  await fixture.finish(answer(3, 35) + "\n\n" + "LongUnbrokenText".repeat(90));
  await assertDocked(page);
  await atBottom(page);
  const g = await geometry(page);
  expect(g.scroller.scrollHeight).toBeGreaterThan(g.scroller.clientHeight * 3);
  for (const height of [580, 940]) {
    await page.setViewportSize({ width: page.viewportSize().width, height });
    await assertDocked(page);
    await atBottom(page);
  }
  if (info.project.name === "desktop") {
    await page.locator(".chat-sidebar").getByRole("button", { name: "Đóng thanh bên" }).click();
    await assertDocked(page);
  }
  await page.getByRole("button", { name: "Mở thanh bên" }).click();
  await assertDocked(page);
  await page.locator(".chat-sidebar").getByRole("button", { name: "Đóng thanh bên" }).click();
  await assertDocked(page);
  await page.screenshot({ path: info.outputPath("long-response.png") });
});

test("stream follows near-bottom readers and preserves manual scrollback", async ({ page }) => {
  const fixture = await setupChat(page, { initialTurns: 2 });
  await submit(page, 3);
  await fixture.emit("status", { stage: "central_answering" });
  await fixture.emit("answer_delta", { delta: answer(3, 8) });
  await atBottom(page);
  const scroller = page.locator(".thread-scroll");
  await scroller.evaluate((node) => { node.scrollTop -= 70; });
  await expect.poll(() => scroller.evaluate((node) => node.scrollHeight - node.scrollTop - node.clientHeight)).toBeCloseTo(70, 0);
  await fixture.emit("answer_delta", { delta: "\n\n" + answer(3, 2) });
  await atBottom(page);
  await scroller.evaluate((node) => { node.scrollTop = 180; });
  await expect.poll(() => scroller.evaluate((node) => node.scrollTop)).toBe(180);
  for (let chunk = 0; chunk < 3; chunk += 1) {
    await fixture.emit("answer_delta", { delta: `\n\nLecture ${chunk}. ` + answer(3, 2) });
    await expect(page.locator(".assistant-message").last()).toContainText(`Lecture ${chunk}`);
    expect(await scroller.evaluate((node) => node.scrollTop)).toBe(180);
  }
  await fixture.finish();
  expect(await scroller.evaluate((node) => node.scrollTop)).toBe(180);
  await assertDocked(page);
  // Sending a new question is an explicit request to reveal the latest turn.
  await submit(page, 4);
  await atBottom(page);
  await fixture.finish(answer(4));
});
