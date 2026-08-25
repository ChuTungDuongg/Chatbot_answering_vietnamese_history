from __future__ import annotations

import argparse
import re
from typing import Any

from training.common.datasets import first_user_assistant, load_messages
from training.common.jsonl import write_jsonl
from training.history_answerer.evaluate import parse_source_ids


CONTEXT_RE = re.compile(r"(?ms)^\[([^\]]+)\]\s*([^\n]*)\n(.*?)(?=^\[[^\]]+\]|\Z)")


def parse_question_and_evidence(user_text: str) -> tuple[str, list[dict[str, Any]]]:
    question_part, _, context_part = user_text.partition("Tài liệu tham khảo:")
    question = question_part.replace("Câu hỏi:", "", 1).strip()
    evidence = [
        {
            "evidence_id": match.group(1).strip(),
            "source_type": "local",
            "title": match.group(2).strip() or None,
            "url": None,
            "chunk_id": match.group(1).strip(),
            "text": match.group(3).strip(),
            "retrieval_score": None,
        }
        for match in CONTEXT_RE.finditer(context_part)
    ]
    return question, evidence


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    user_text, assistant = first_user_assistant(row)
    question, evidence = parse_question_and_evidence(user_text)
    selected_ids = parse_source_ids(assistant)
    by_id = {item["evidence_id"]: item for item in evidence}
    status = "sufficient" if selected_ids else "insufficient"
    return {
        "id": row.get("id"),
        "question": question,
        "evidence": evidence,
        "output": {
            "selected_ids": selected_ids,
            "selected_evidence": [
                {
                    "evidence_id": evidence_id,
                    "relevance": 1.0,
                    "claims": [],
                    "compressed_text": str(by_id.get(evidence_id, {}).get("text", ""))[:1200],
                }
                for evidence_id in selected_ids
                if evidence_id in by_id
            ],
            "rejected_ids": [item["evidence_id"] for item in evidence if item["evidence_id"] not in selected_ids],
            "compressed_context": "\n\n".join(
                str(by_id.get(evidence_id, {}).get("text", ""))[:1200]
                for evidence_id in selected_ids
            ),
            "conflicts": [],
            "missing_information": [] if selected_ids else ["Không có evidence đủ để trả lời."],
            "summary": "Evidence đã được lọc từ context huấn luyện." if selected_ids else "Evidence không đủ.",
            "status": status,
            "sufficient": bool(selected_ids),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build evidence critic/compressor SFT rows from grounded QA data.")
    parser.add_argument("--input", default="Dataset/merged_jsonl/all_messages.jsonl")
    parser.add_argument("--output", default="artifacts/training/evidence_agent/critic_rows.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Wrote {write_jsonl(args.output, [build_row(row) for row in load_messages(args.input)])} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



