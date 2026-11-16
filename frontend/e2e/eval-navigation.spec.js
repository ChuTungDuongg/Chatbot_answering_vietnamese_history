import { test, expect } from "@playwright/test";

test("owner opens Studio from chat and returns after reload", async ({ page }, testInfo) => {
  await page.goto("/login");
  const status = await page.evaluate(async () => (await fetch("/auth/login", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password: "test-password" }),
  })).status);
  expect(status).toBe(200);
  await page.goto("/");
  const openSidebar = page.getByRole("button", { name: "Mở thanh bên", exact: true });
  if (await openSidebar.isVisible()) await openSidebar.click();
  await expect.poll(async () => (await page.locator(".chat-sidebar").boundingBox()).x).toBeGreaterThanOrEqual(0);
  const link = page.getByRole("link", { name: "Eval Studio", exact: true });
  await expect(link).toBeVisible();
  await expect(link).toHaveAttribute("href", "/eval-studio");
  await link.focus();
  await expect(link).toBeFocused();
  await page.screenshot({ path: `../reports/eval-navigation-${testInfo.project.name}.png` });
  await link.press("Enter");
  await expect(page).toHaveURL(/\/eval-studio$/);
  await expect(page.getByRole("heading", { name: "Tổng quan dataset" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Tổng quan dataset" })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(/\/$/);
  if (await openSidebar.isVisible()) await openSidebar.click();
  await expect(link).toBeVisible();
});
