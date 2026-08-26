from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common.jsonl import read_jsonl, write_jsonl


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    if "trajectory" in row:
        return row
    if "messages" in row:
        return {"source": row.get("source", "external"), "messages": row["messages"], "trajectory": row.get("trajectory", {})}
    question = row.get("question") or row.get("input") or row.get("query")
    answer = row.get("answer") or row.get("output") or row.get("final_answer")
    tools = row.get("tool_calls") or row.get("actions") or []
    first_tool = tools[0] if tools else None
    target = (
        {
            "action": "tool",
            "tool_name": first_tool.get("name") or first_tool.get("tool_name"),
            "arguments": first_tool.get("arguments") or first_tool.get("parameters") or {},
        }
        if isinstance(first_tool, dict)
        else {"action": "finish", "sufficient": bool(answer), "missing_information": []}
    )
    prompt = {"question": str(question or ""), "observations": [], "evidence_ids": []}
    return {
        "source": row.get("source", "external"),
        "messages": [
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "training_prompt": prompt,
        "training_target": target,
        "trajectory": {"tool_calls": tools},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize xLAM/AgentInstruct/Hotpot-style rows for research-agent SFT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/training/research_agent/normalized.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [normalize_row(row) for row in read_jsonl(args.input)]
    print(f"Wrote {write_jsonl(Path(args.output), rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



