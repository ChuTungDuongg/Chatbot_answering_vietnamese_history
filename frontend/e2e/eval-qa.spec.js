import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { Buffer } from "node:buffer";

async function login(page) {
  await page.goto("/login");
  await page.getByLabel("Mật khẩu", { exact: true }).fill("test-password");
  await page.getByRole("button", { name: "Đăng nhập", exact: true }).click();
  await expect(page).not.toHaveURL(/\/login/);
}
async function editor(page, id) {
  await page.goto("/eval-studio?workspace=bank");
  await page.getByRole("button", { name: new RegExp(id) }).click();
  await expect(page.locator(".editor-id code")).toHaveText(id);
  return page.getByRole("textbox", { name: "Câu hỏi", exact: true });
}

test("two tabs reject a stale edit and retain the unsaved draft", async ({ page, context }, testInfo) => {
  await login(page);
  const first = await editor(page, "HIST-EVAL-003");
  const other = await context.newPage();
  const second = await editor(other, "HIST-EVAL-003");
  const savedText = `Saved from first tab ${testInfo.project.name}`;
  const localText = `Unsaved second tab ${testInfo.project.name}`;
  const saved = page.waitForResponse(r => r.url().includes("/questions/HIST-EVAL-003") && r.request().method() === "PATCH");
  await first.fill(savedText);
  await first.press("ControlOrMeta+s");
  expect((await saved).status()).toBe(200);
  const rejected = other.waitForResponse(r => r.url().includes("/questions/HIST-EVAL-003") && r.request().method() === "PATCH");
  await second.fill(localText);
  await second.press("ControlOrMeta+s");
  expect((await rejected).status()).toBe(409);
  await expect(other.getByRole("alert")).toBeVisible();
  await expect(second).toHaveValue(localText);
  await other.getByRole("navigation").getByRole("button", { name: "Experiments", exact: true }).click();
  await expect(other.getByRole("heading", { name: "Ngân hàng câu hỏi", exact: true })).toBeVisible();
  await expect(second).toHaveValue(localText);
  const stored = await page.evaluate(async () => (await (await fetch("/api/v1/eval/questions")).json()).find(q => q.id === "HIST-EVAL-003"));
  expect(stored.question).toBe(savedText);
  other.on("dialog", dialog => dialog.accept());
  await other.reload();
  await other.getByRole("button", { name: /HIST-EVAL-003/ }).click();
  await expect(second).toHaveValue(savedText);
});

test("workspace URLs retain browser Back, Forward and refresh state", async ({ page }) => {
  await login(page);
  await page.goto("/eval-studio");
  await page.getByRole("navigation").getByRole("button", { name: /Ngân hàng câu hỏi/ }).click();
  await expect(page.getByRole("heading", { name: "Ngân hàng câu hỏi", exact: true })).toBeVisible();
  await page.getByRole("navigation").getByRole("button", { name: "Experiments", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Experiments", exact: true })).toBeVisible();
  await page.goBack();
  await expect(page.getByRole("heading", { name: "Ngân hàng câu hỏi", exact: true })).toBeVisible();
  await page.goForward();
  await expect(page.getByRole("heading", { name: "Experiments", exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Experiments", exact: true })).toBeVisible();
  await expect(page.getByRole("navigation").getByRole("button", { name: "Experiments", exact: true })).toHaveAttribute("aria-current", "page");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("all workspaces support both themes and authenticated backup download", async ({ page }, testInfo) => {
  test.setTimeout(60_000);
  await login(page);
  const workspaces = [
    ["overview", "Tổng quan", "Tổng quan dataset"],
    ["bank", /^Ngân hàng câu hỏi/, "Ngân hàng câu hỏi"],
    ["review", "Human review", "Human review"],
    ["experiments", "Experiments", "Experiments"],
    ["comparison", "Model Comparison", "Model Comparison"],
    ["blind", "Blind A/B", "Blind A/B"],
    ["data", "Dữ liệu & Backup", "Dữ liệu & Backup"],
  ];
  for (const theme of ["light", "dark"]) {
    await page.evaluate(value => localStorage.setItem("vn-history-theme", value), theme);
    await page.goto("/eval-studio");
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    for (const [id, label, heading] of workspaces) {
      await page.getByRole("navigation").getByRole("button", { name: label, exact: true }).click();
      await expect(page.getByRole("heading", { name: heading, exact: true })).toBeVisible();
      if (id === "bank") {
        await page.getByRole("button", { name: /HIST-EVAL-001/ }).click();
        await expect(page.locator(".editor-id code")).toHaveText("HIST-EVAL-001");
        await expect(page.locator(".editor-id code")).toHaveCSS("white-space", "nowrap");
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      await page.screenshot({ path: `../reports/eval-qa-${id}-${theme}-${testInfo.project.name}.png`, fullPage: true });
    }
  }
  const downloaded = page.waitForEvent("download");
  await page.getByRole("link", { name: "Tải backup", exact: true }).click();
  const backup = await downloaded;
  expect(backup.suggestedFilename()).toMatch(/\.zip$/);
  const bytes = await readFile(await backup.path());
  expect(bytes.subarray(0, 4)).toEqual(Buffer.from([0x50, 0x4b, 0x03, 0x04]));
});

test("JSONL run import preserves captured evidence and trace in the inspector", async ({ page }, testInfo) => {
  await login(page);
  const version = `captured-import-${testInfo.project.name}-${Date.now()}`;
  const fixture = await page.evaluate(async version => {
    const snapshot = await fetch("/api/v1/eval/snapshots", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ version }) });
    const experiment = await fetch("/api/v1/eval/experiments", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: version, snapshot_id: version, model_id: "qa-import-model" }) });
    return { snapshotStatus: snapshot.status, experimentStatus: experiment.status, experiment: await experiment.json() };
  }, version);
  expect(fixture.snapshotStatus).toBe(201);
  expect(fixture.experimentStatus).toBe(201);
  const excerpt = `Captured passage ${testInfo.project.name}`;
  const firstAnswer = `Captured import answer ${testInfo.project.name}`;
  const rows = [
    { experiment_id: fixture.experiment.id, question_id: "HIST-EVAL-004", external_id: "captured-one", answer: firstAnswer, trace_captured: true, evidence: [{ chunk_id: "captured-chunk-1", title: "Captured source", excerpt, selected: true, cited: true, final_rank: 1 }], trace: { retrieval_backend: "qdrant", final_context_chunk_ids: ["captured-chunk-1"], cited_chunk_ids: ["captured-chunk-1"], tools: [{ name: "search_documents", status: "completed", duration_ms: 12 }] } },
    { experiment_id: fixture.experiment.id, question_id: "HIST-EVAL-005", external_id: "captured-two", answer: `Second imported answer ${testInfo.project.name}` },
  ];
  await page.goto("/eval-studio?workspace=experiments");
  await page.getByRole("button", { name: new RegExp(version) }).click();
  await page.getByText("Import nhiều runs (JSON / JSONL / CSV)", { exact: true }).click();
  await page.getByLabel("Chọn JSON, JSONL hoặc CSV").setInputFiles({ name: "captured-runs.jsonl", mimeType: "application/json", buffer: Buffer.from(rows.map(row => JSON.stringify(row)).join("\n") + "\n") });
  await page.getByRole("button", { name: "Kiểm tra trước khi nhập", exact: true }).click();
  await expect(page.getByText("2 / 2 bản ghi hợp lệ", { exact: true })).toBeVisible();
  await expect(page.getByText("0 ID đã tồn tại · 0 lỗi", { exact: true })).toBeVisible();
  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Xác nhận nhập toàn bộ", exact: true }).click();
  await expect(page.getByText("Đã xử lý 2 bản ghi.", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: new RegExp(`Second imported answer ${testInfo.project.name}`) })).toBeVisible();
  await page.getByRole("button", { name: new RegExp(firstAnswer) }).click();
  const inspector = page.getByLabel("Evidence Inspector");
  await expect(inspector.getByText(excerpt, { exact: true })).toBeVisible();
  await expect(inspector.getByText("Final context: captured-chunk-1", { exact: true })).toBeVisible();
  await expect(inspector.getByText("Cited chunks: captured-chunk-1", { exact: true })).toBeVisible();
  await expect(inspector.getByText("Backend: qdrant", { exact: true })).toBeVisible();
  await expect(inspector.getByText(/ID: captured-chunk-1 · Selected · Cited/)).toBeVisible();
  await expect(inspector.getByText("search_documents · completed · 12 ms", { exact: true })).toBeVisible();
  await expect(inspector.getByText("Not captured — không có evidence snapshot.", { exact: true })).toHaveCount(0);
  await inspector.scrollIntoViewIfNeeded();
  await page.screenshot({ path: `../reports/eval-qa-import-evidence-${testInfo.project.name}.png`, fullPage: true });
});
