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
    // restoreMocks chỉ khôi phục spy của vi.spyOn, KHÔNG xoá lịch sử gọi của
    // vi.fn() tạo trong factory của vi.mock. Thiếu dòng này thì một test khẳng
    // định "không được gọi" sẽ đỏ vì lần gọi của test chạy trước nó.
    clearMocks: true,
  },
});
