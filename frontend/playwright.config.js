import { defineConfig } from "@playwright/test";

// Uses an already installed browser. No browser/weights download or backend is needed.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  workers: 3,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    channel: "chrome",
    launchOptions: { args: ["--disable-gpu"] },
    colorScheme: "dark",
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    { name: "tablet", use: { viewport: { width: 834, height: 1112 } } },
    { name: "mobile", use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    env: { VITE_API_BASE_URL: "http://127.0.0.1:4173" },
    reuseExistingServer: false,
  },
});
