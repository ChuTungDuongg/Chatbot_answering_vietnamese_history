import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import ResponseDetailSlider from "../../src/components/ResponseDetailSlider.jsx";
import { persistResponseMode, readResponseMode } from "../../src/config/responseModes.js";

test("three-position slider is keyboard accessible and maps labels to modes", () => {
  const onChange = vi.fn();
  const { rerender } = render(<ResponseDetailSlider value="standard" onChange={onChange} />);
  const slider = screen.getByRole("slider", { name: /Độ chi tiết/ });
  expect(slider).toHaveAttribute("aria-valuetext", "Tiêu chuẩn");
  fireEvent.change(slider, { target: { value: "0" } });
  expect(onChange).toHaveBeenCalledWith("concise");
  rerender(<ResponseDetailSlider value="detailed" onChange={onChange} />);
  expect(slider).toHaveAttribute("aria-valuetext", "Chi tiết");
});

test("response mode is persisted with standard as default", () => {
  window.localStorage.clear();
  expect(readResponseMode()).toBe("standard");
  persistResponseMode("detailed");
  expect(readResponseMode()).toBe("detailed");
  window.localStorage.setItem("vn-history-response-mode", "invalid");
  expect(readResponseMode()).toBe("standard");
});
