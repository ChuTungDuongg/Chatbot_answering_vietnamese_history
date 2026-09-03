import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";

import { THEME_STORAGE_KEY, useTheme } from "../../src/hooks/useTheme.js";
import { useChatMode } from "../../src/hooks/useChatMode.js";

beforeEach(() => {
  window.localStorage.clear();
  window.matchMedia = (query) => ({
    matches: false,
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  });
});

test("useTheme đọc lựa chọn đã lưu", () => {
  window.localStorage.setItem(THEME_STORAGE_KEY, "dark");
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("dark");
});

test("useTheme lật giữa dark và light rồi ghi vào localStorage và thẻ html", () => {
  const { result } = renderHook(() => useTheme());
  const first = result.current.theme;

  act(() => result.current.toggleTheme());

  expect(result.current.theme).not.toBe(first);
  expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe(result.current.theme);
  expect(document.documentElement.dataset.theme).toBe(result.current.theme);
});

test("useChatMode mặc định hybrid và lưu lại lựa chọn mới", () => {
  const { result } = renderHook(() => useChatMode());
  expect(result.current.mode).toBe("hybrid");

  act(() => result.current.setMode("central"));

  expect(result.current.mode).toBe("central");
  expect(window.localStorage.getItem("vn-history-chat-mode-v2")).toBe("central");
});
