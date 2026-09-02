import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

function Hello({ name }) {
  return <p>Xin chào {name}</p>;
}

test("hạ tầng Vitest render được component React", () => {
  render(<Hello name="Sử Việt" />);
  expect(screen.getByText("Xin chào Sử Việt")).toBeInTheDocument();
});
