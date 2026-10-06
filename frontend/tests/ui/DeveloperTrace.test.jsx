import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import DeveloperTrace from "../../src/components/DeveloperTrace";

test("renders planner rounds and actual termination without exposing hidden reasoning", () => {
  render(<DeveloperTrace trace={{ mode: "central",
    planning: { rounds: [{ round: 1, model_ms: 1200, input_tokens: 200, output_tokens: 30 }],
      hidden_reasoning: "private planner text" },
    generation: { settings: { max_new_tokens: 1536 }, finish_reason: "length", truncated: true },
    performance: { e2e_ms: 2500 } }} />);
  expect(screen.getByText("Planning")).toBeInTheDocument();
  expect(screen.getByText(/"model_ms": 1200/)).toBeInTheDocument();
  expect(screen.getByText(/"finish_reason": "length"/)).toBeInTheDocument();
  expect(screen.queryByText(/private planner text/)).not.toBeInTheDocument();
});
