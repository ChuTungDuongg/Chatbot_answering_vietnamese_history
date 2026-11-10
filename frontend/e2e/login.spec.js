import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { randomBytes } from "node:crypto";

const html = readFileSync(new URL("../../app/security/login.html", import.meta.url), "utf8");

test("login form handles password visibility, errors, loading and successful navigation", async ({ page }) => {
  const password = randomBytes(20).toString("hex");
  await page.route("**/login", (route) => route.fulfill({ status: 200, contentType: "text/html", body: html }));
  await page.route("**/auth/login", async (route) => {
    const valid = route.request().postDataJSON().password === password;
    await new Promise((resolve) => setTimeout(resolve, 100));
    await route.fulfill({ status: valid ? 200 : 401, contentType: "application/json", body: "{}" });
  });
  await page.route("http://127.0.0.1:4173/", (route) => route.fulfill({ status: 200, contentType: "text/html", body: "<h1>Chat</h1>" }));
  await page.goto("/login");
  const input = page.getByLabel("Mật khẩu", { exact: true });
  await expect(input).toHaveAttribute("type", "password");
  await page.getByRole("button", { name: "Hiện mật khẩu" }).click();
  await expect(input).toHaveAttribute("type", "text");
  await page.getByRole("button", { name: "Ẩn mật khẩu" }).click();
  await input.fill(randomBytes(10).toString("hex"));
  await page.getByRole("button", { name: "Đăng nhập" }).click();
  await expect(page.getByRole("alert")).toContainText("Mật khẩu không đúng");
  await input.fill(password);
  await page.getByRole("button", { name: "Đăng nhập" }).click();
  await expect(page.getByRole("button", { name: "Đang đăng nhập..." })).toBeDisabled();
  await expect(page.getByRole("heading", { name: "Chat" })).toBeVisible();
});
