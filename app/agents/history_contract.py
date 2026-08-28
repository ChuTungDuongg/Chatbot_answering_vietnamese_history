from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


QUESTION_PREFIX = "Câu hỏi:"
REFERENCES_PREFIX = "Tài liệu tham khảo:"
SOURCE_PREFIX = "Nguồn được dùng:"
ANSWER_PREFIX = "Trả lời:"

SAFE_INSUFFICIENT_ANSWER = (
    "Không đủ bằng chứng trong các tài liệu đã chọn để trả lời chắc chắn câu hỏi này."
)
SAFE_OOD_ANSWER = (
    "Câu hỏi này nằm ngoài phạm vi hệ thống lịch sử Việt Nam, nên không thể trả lời "
    "bằng corpus hiện tại."
)

_USER_RE = re.compile(
    rf"\A{re.escape(QUESTION_PREFIX)}\r?\n(?P<question>.*?)"
    rf"\r?\n\r?\n{re.escape(REFERENCES_PREFIX)}\r?\n(?P<references>.*)\Z",
    re.S,
)
_REFERENCE_RE = re.compile(
    r"(?m)^\[([^\]\r\n]+)\](?:[ \t]+([^\r\n]*))?\r?\n"
)
_OUTPUT_RE = re.compile(
    rf"\A\s*{re.escape(SOURCE_PREFIX)}\s*(?P<sources>[^\r\n]*)"
    rf"\r?\n(?:\s*\r?\n)?\s*{re.escape(ANSWER_PREFIX)}\s*(?P<answer>.*)\Z",
    re.I | re.S,
)
_BRACKET_ID_RE = re.compile(r"\[([^\]]+)\]")


class HistoryAnswerContractError(RuntimeError):
    """History adapter output violated the citation/output contract learned in SFT."""


@dataclass(frozen=True)
class ParsedHistoryAnswer:
    answer: str
    source_ids: list[str]
    raw_output: str


def _canonical_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    canonical: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in evidence:
        evidence_id = str(item.get("chunk_id") or item.get("evidence_id") or "").strip()
        text = str(item.get("text") or item.get("compressed_text") or "").strip()
        if not evidence_id or not text or evidence_id in seen:
            continue
        seen.add(evidence_id)
        canonical.append(
            {
                "chunk_id": evidence_id,
                "title": str(item.get("title") or "").strip(),
                "text": text,
            }
        )
    return canonical


def build_history_answerer_user_text(
    question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Render the exact user-message structure used by canonical History SFT."""
    question = str(question).strip()
    if not question:
        raise ValueError("History question must not be empty.")

    blocks: list[str] = []
    for item in _canonical_evidence(evidence):
        header = f"[{item['chunk_id']}]"
        if item["title"]:
            header += f" {item['title']}"
        blocks.append(f"{header}\n{item['text']}")

    references = "\n\n".join(blocks)
    return f"{QUESTION_PREFIX}\n{question}\n\n{REFERENCES_PREFIX}\n{references}"


def build_history_answerer_messages(
    question: str,
    evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    # Canonical History SFT contains no system or conversation-history message.
    return [
        {
            "role": "user",
            "content": build_history_answerer_user_text(question, evidence),
        }
    ]


def parse_history_training_user_text(text: str) -> tuple[str, list[dict[str, str]]]:
    """Parse canonical rows for structural golden replay tests and audit tooling."""
    match = _USER_RE.match(str(text))
    if not match:
        raise ValueError("History user message does not match the canonical SFT structure.")

    references = match.group("references")
    matches = list(_REFERENCE_RE.finditer(references))
    evidence: list[dict[str, str]] = []
    for index, reference in enumerate(matches):
        body_start = reference.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(references)
        evidence.append(
            {
                "chunk_id": reference.group(1).strip(),
                "title": str(reference.group(2) or "").strip(),
                "text": references[body_start:body_end].strip(),
            }
        )
    return match.group("question").strip(), evidence


def parse_history_answer_output(
    output: str,
    *,
    allowed_source_ids: list[str],
) -> ParsedHistoryAnswer:
    raw_output = str(output).strip()
    match = _OUTPUT_RE.match(raw_output)
    if not match:
        raise HistoryAnswerContractError(
            "History adapter output must contain 'Nguồn được dùng:' followed by 'Trả lời:'."
        )

    answer = match.group("answer").strip()
    if not answer:
        raise HistoryAnswerContractError("History adapter returned an empty answer body.")

    source_text = match.group("sources").strip()
    source_ids = list(dict.fromkeys(item.strip() for item in _BRACKET_ID_RE.findall(source_text)))
    residue = _BRACKET_ID_RE.sub("", source_text).replace(",", "").replace("[]", "").strip()
    if residue:
        raise HistoryAnswerContractError("History source line contains non-canonical citation text.")

    allowed = set(allowed_source_ids)
    cited_anywhere = list(dict.fromkeys(
        item.strip() for item in _BRACKET_ID_RE.findall(raw_output) if item.strip()
    ))
    unknown = [item for item in cited_anywhere if item not in allowed]
    if unknown:
        raise HistoryAnswerContractError(f"History adapter invented citation IDs: {unknown}")

    return ParsedHistoryAnswer(
        answer=answer,
        source_ids=source_ids,
        raw_output=raw_output,
    )
