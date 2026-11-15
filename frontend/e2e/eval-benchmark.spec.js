import { test, expect } from "@playwright/test";

// Run against tests/eval_browser_server.py.
// That API-only fixture injects an offline executor; this test never uses a provider.
test("owner previews and explicitly starts offline benchmarks, then inspects and compares measurements", async ({ page }, testInfo) => {
  await page.goto("/login");
  expect(await page.evaluate(async () => (await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password: "test-password" }) })).status)).toBe(200);
  const version = `v12-${testInfo.project.name}-${Date.now()}`;
  const fixture = await page.evaluate(async version => {
    async function request(path, method = "GET", body) {
      const result = await fetch(`/api/v1/eval${path}`, { method, headers: { "Content-Type": "application/json" }, ...(body ? { body: JSON.stringify(body) } : {}) });
      if (!result.ok) throw new Error(`${path}: ${result.status} ${await result.text()}`);
      return result.json();
    }
    const capabilities = await request("/benchmark/capabilities");
    if (!capabilities.live_available) throw new Error("Start tests/eval_browser_server.py with its offline executor.");
    const questions = await request("/questions");
    const question = questions.find(q => q.id === "HIST-EVAL-001") || questions[0];
    await request(`/questions/${question.id}`, "PATCH", { ...question, question: "Ai lãnh đạo cuộc khởi nghĩa Lam Sơn?", reference_answer: "Lê Lợi lãnh đạo cuộc khởi nghĩa Lam Sơn.", expected_behavior: { required_tools: ["search_history"], expected_evidence_ids: ["fixture-chunk"], max_tool_calls: 3, require_citations: true } });
    await request("/snapshots", "POST", { version });
    const experiments = [];
    for (const [name, mode] of [["Alpha", "hybrid"], ["Beta", "central"]]) experiments.push(await request("/experiments", "POST", { name: `${name} ${version}`, snapshot_id: version, provider: "openai", model_id: "gpt-6-luna", chat_mode: mode, dense_backend: "faiss", final_k: 6, reasoning_effort: "medium", planning_reasoning_effort: "inherit" }));
    // Exercise wrapping and a bounded selector even on a fresh fixture database.
    for (let i = 0; i < 10; i++) await request("/experiments", "POST", { name: `Layout ${version} ${i} ${"w".repeat(70)}`, snapshot_id: version, provider: "openai", model_id: "gpt-6-luna", chat_mode: "hybrid", dense_backend: "faiss", final_k: 6, reasoning_effort: "medium", planning_reasoning_effort: "inherit" });
    return { questionId: question.id, experiments, capabilities };
  }, version);
  await page.goto("/eval-studio?workspace=benchmarks");
  await expect(page.getByRole("heading", { name: "Benchmarks", exact: true })).toBeVisible();
  const jobs = [];
  for (const experiment of fixture.experiments) {
    await page.getByLabel("Experiment benchmark").selectOption(experiment.id);
    await page.getByLabel("Question IDs (để trống: tất cả)", { exact: true }).fill(fixture.questionId);
    await expect(page.getByLabel("Số lần retry tối đa")).toHaveValue("0");
    await page.getByRole("button", { name: "Xem cấu hình trước khi tạo" }).click();
    await expect(page.getByLabel("Xem trước benchmark")).toContainText("gpt-6-luna");
    const created = page.waitForResponse(response => response.url().endsWith("/api/v1/eval/benchmarks") && response.request().method() === "POST");
    await page.getByRole("button", { name: "Tạo benchmark chưa chạy" }).click();
    const createResponse = await created;
    expect(createResponse.status()).toBe(201);
    const job = await createResponse.json(); jobs.push(job);
    expect(job.status).toBe("ready");
    expect(job.counts.completed).toBe(0);
    const detail = page.getByRole("region", { name: "Chi tiết benchmark", exact: true });
    await expect(detail.locator(".benchmark-hashes").getByText(job.configuration_hash, { exact: true }).first()).toBeVisible();
    const start = detail.getByRole("button", { name: "Start · gọi model có tính phí" });
    await expect(start).toBeDisabled();
    await detail.getByLabel(/Tôi xác nhận gọi model có tính phí/).check();
    await start.click();
    await expect(detail.locator(".benchmark-state")).toHaveText("Hoàn tất", { timeout: 15000 });
    await expect(detail.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "1");
    await detail.getByRole("button", { name: `Xem bằng chứng ${fixture.questionId}` }).click();
    const drawer = page.getByRole("dialog", { name: "Chi tiết run" });
    await expect(drawer.getByText("Offline benchmark fixture answer [fixture-chunk]", { exact: true })).toBeVisible();
    await expect(drawer.getByText("Offline fixture source", { exact: true })).toBeVisible();
    await expect(drawer.getByRole("region", { name: "Đo lường tự động" })).toContainText("checks.required_tools");
    await expect(drawer.locator(".tool-diagnostics").getByText("search_history", { exact: true })).toBeVisible();
    await expect(drawer.getByRole("button", { name: "Đóng", exact: true })).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(drawer).toHaveCount(0);
    await expect(detail.getByRole("button", { name: `Xem bằng chứng ${fixture.questionId}` })).toBeFocused();
  }
  await page.getByRole("button", { name: "So sánh tự động", exact: true }).click();
  const comparison = page.getByRole("region", { name: "So sánh đo lường tự động" });
  for (let i = 0; i < fixture.experiments.length; i++) {
    const experiment = fixture.experiments[i];
    await comparison.getByLabel(experiment.name, { exact: false }).first().check();
    await comparison.getByRole("combobox", { name: `Benchmark của ${experiment.name}`, exact: true }).selectOption(jobs[i].id);
  }
  await comparison.getByRole("button", { name: "So sánh đo lường", exact: true }).click();
  await expect(comparison.getByRole("table", { name: "Các phép đo riêng biệt" })).toBeVisible();
  await expect(comparison.getByText("Điểm human (thủ công)", { exact: true })).toBeVisible();
  await expect(comparison.getByText("Chưa ghi", { exact: true }).first()).toBeVisible();
  await expect(comparison.getByRole("table", { name: "Các phép đo riêng biệt" }).getByText("tools.success_rate", { exact: true })).toBeVisible();
  const layout = await comparison.evaluate(panel => {
    const main = document.querySelector(".eval-main");
    const mainRect = main.getBoundingClientRect(), panelRect = panel.getBoundingClientRect();
    const options = panel.querySelector(".experiment-options");
    const note = panel.querySelector(".comparison-note");
    return { mainClient: main.clientWidth, mainScroll: main.scrollWidth, mainLeft: mainRect.left, mainRight: mainRect.right, panelLeft: panelRect.left, panelRight: panelRect.right, noteClient: note.clientWidth, noteScroll: note.scrollWidth, optionsHeight: options.clientHeight, optionsScroll: options.scrollHeight, optionCount: options.children.length, tableScrolls: [...panel.querySelectorAll(".table-scroll")].map(node => ({ client: node.clientWidth, scroll: node.scrollWidth, overflowX: getComputedStyle(node).overflowX })) };
  });
  expect(layout.mainScroll, "Eval content must fit its own viewport").toBeLessThanOrEqual(layout.mainClient + 1);
  expect(layout.panelLeft).toBeGreaterThanOrEqual(layout.mainLeft);
  expect(layout.panelRight, "Comparison panel must remain inside Eval content").toBeLessThanOrEqual(layout.mainRight + 1);
  expect(layout.noteScroll, "Snapshot note must wrap inside the panel").toBeLessThanOrEqual(layout.noteClient + 1);
  expect(layout.optionsHeight, "Experiment selector must remain a bounded scroll region").toBeLessThanOrEqual(360);
  if (layout.optionCount > 8) expect(layout.optionsScroll).toBeGreaterThan(layout.optionsHeight);
  for (const table of layout.tableScrolls) expect(["auto", "scroll"]).toContain(table.overflowX);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: `../reports/eval-v12-benchmark-${testInfo.project.name}.png`, fullPage: true });
});
