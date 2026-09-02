import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    include: ["tests/ui/**/*.test.{js,jsx}"],
    environment: "jsdom",
    setupFiles: ["tests/ui/setup.js"],
    globals: false,
    restoreMocks: true,
  },
});
