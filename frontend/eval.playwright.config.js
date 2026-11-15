import { defineConfig } from "@playwright/test";
import process from "node:process";
export default defineConfig({ testDir: "./e2e", testMatch: "eval*.spec.js", workers: 1, reporter: "list", use: { baseURL: "http://127.0.0.1:8765", channel: process.env.CI ? "chromium" : "chrome", reducedMotion: "reduce", screenshot: "only-on-failure" }, projects: [ { name: "desktop", use: { viewport: { width: 1440, height: 900 } } }, { name: "tablet", use: { viewport: { width: 834, height: 1112 } } }, { name: "mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } } ] });
