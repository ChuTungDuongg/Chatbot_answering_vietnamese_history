from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agents.comparison import (
    TARGET_A,
    TARGET_B,
    comparison_dimension_coverage,
    group_comparison_evidence,
)
from app.rag.retrieval import extract_comparison_targets


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
_CAUSE_CUES = {
    "nguyen nhan",
    "vi sao",
    "tai sao",
    "dan den",
    "suy yeu",
}
_SIGNIFICANCE_CUES = {
    "y nghia",
    "vai tro",
    "he qua",
    "tac dong",
}
_ANALYSIS_CUES = _CAUSE_CUES | _SIGNIFICANCE_CUES | {
    "phan tich",
    "danh gia",
}
_FACTUAL_PREFIXES = {
    "ai",
    "khi nao",
    "o dau",
    "nhan vat nao",
    "vua nao",
    "tuong nao",
    "trieu dai nao",
    "su kien nao",
    "nam nao",
}
_FACTUAL_CUES = {
    "duoc menh danh",
    "la ai",
    "ten gi",
    "ten la gi",
}


class HistoryAnswerContractError(RuntimeError):
    """History adapter output violated the citation/output contract learned in SFT."""


@dataclass(frozen=True)
class ParsedHistoryAnswer:
    answer: str
    source_ids: list[str]
    raw_output: str


def _normalize_text(value: str) -> str:
    value = str(value).lower()
    replacements = {
        "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a", "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a", "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
        "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e", "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
        "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o", "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o", "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u", "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
        "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y", "đ": "d",
    }
    return "".join(replacements.get(char, char) for char in value)


def _history_question_type(question: str) -> str:
    normalized = _normalize_text(question)
    if len(extract_comparison_targets(question)) >= 2:
        return "compare"
    if any(cue in normalized for cue in _CAUSE_CUES):
        return "cause"
    if any(cue in normalized for cue in _SIGNIFICANCE_CUES):
        return "significance"
    if any(cue in normalized for cue in _ANALYSIS_CUES):
        return "analysis"
    if any(normalized == prefix or normalized.startswith(f"{prefix} ") for prefix in _FACTUAL_PREFIXES):
        return "factual"
    if any(cue in normalized for cue in _FACTUAL_CUES):
        return "factual"
    return "general"


def _deep_depth_instruction(question: str) -> str:
    question_type = _history_question_type(question)
    if question_type == "compare":
        return (
            "Yêu cầu trả lời:\n"
            "- Xác định rõ hai đối tượng được so sánh.\n"
            "- Nêu điểm giống nhau chỉ khi có nguồn shared hoặc có hỗ trợ riêng cho cả hai đối tượng.\n"
            "- Nêu điểm khác nhau với đúng đối tượng được nguồn hỗ trợ; không hoán đổi dữ kiện một phía.\n"
            "- Kết thúc bằng nhận định lịch sử ngắn, chỉ dựa trên tài liệu tham khảo.\n"
            "- Có thể dùng đoạn văn, bảng ngắn hoặc các mục nếu phù hợp; không ép khuôn cố định."
        )
    if question_type == "factual":
        return (
            "Yêu cầu trả lời:\n"
            "- Trả lời trực tiếp ngay ở câu đầu.\n"
            "- Nếu nguồn hỗ trợ, thêm bối cảnh ngắn về nhân vật/sự kiện và ý nghĩa của danh xưng hoặc vai trò.\n"
            "- Không biến câu hỏi sự kiện đơn giản thành bài luận."
        )
    if question_type == "cause":
        return (
            "Yêu cầu trả lời:\n"
            "- Nêu kết luận chung về nguyên nhân trước.\n"
            "- Triển khai các nguyên nhân trực tiếp, khác nhau và được nguồn hỗ trợ.\n"
            "- Giải thích quan hệ giữa các nguyên nhân khi tài liệu cho phép.\n"
            "- Tổng hợp ngắn, không thêm nguyên nhân ngoài tài liệu."
        )
    if question_type == "significance":
        return (
            "Yêu cầu trả lời:\n"
            "- Nêu trực tiếp ý nghĩa lịch sử của sự kiện.\n"
            "- Phân biệt hệ quả trước mắt, tác động chính trị/chủ quyền và ý nghĩa lâu dài nếu nguồn hỗ trợ.\n"
            "- Liên kết các ý thành nhận định lịch sử mạch lạc, không chỉ liệt kê dữ kiện."
        )
    return (
        "Yêu cầu trả lời:\n"
        "- Trả lời trực tiếp câu hỏi lịch sử.\n"
        "- Tổng hợp các ý quan trọng, không trùng lặp, được tài liệu tham khảo hỗ trợ.\n"
        "- Giải thích quan hệ nguyên nhân, hệ quả hoặc ý nghĩa khi nguồn cho phép.\n"
        "- Không thêm ý ngoài tài liệu."
    )


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


def _comparison_group_instruction(question: str, evidence: list[dict[str, Any]]) -> str:
    groups = group_comparison_evidence(question, evidence)
    if not groups:
        return ""
    dimensions = comparison_dimension_coverage(question, evidence)

    def ids(items: list[dict[str, Any]]) -> str:
        values = [
            str(item.get("chunk_id") or item.get("evidence_id") or "").strip()
            for item in items
            if str(item.get("chunk_id") or item.get("evidence_id") or "").strip()
        ]
        return ", ".join(f"[{value}]" for value in values) if values else "không có"

    target_a_dimensions = sorted(dimensions.get(TARGET_A, {}).get("supported_dimensions", {}))
    target_b_dimensions = sorted(dimensions.get(TARGET_B, {}).get("supported_dimensions", {}))
    two_sided_dimensions = list(dimensions.get("two_sided_dimensions") or [])
    one_sided = dimensions.get("one_sided_dimensions") or {}
    limited_instruction = (
        "Không có chiều so sánh hai phía đủ rõ; chỉ đối chiếu những điểm riêng được hỗ trợ và nói ngắn gọn về giới hạn đó."
        if dimensions.get("limited_to_supported_dimensions")
        else "Chỉ dùng các chiều hai phía dưới đây làm điểm giống/khác chung."
    )
    return (
        "\n\nBản đồ bằng chứng so sánh:\n"
        f"- target_a ({groups[TARGET_A]['name']}): {ids(groups[TARGET_A]['evidence'])}; chiều hỗ trợ: {', '.join(target_a_dimensions) or 'không rõ'}\n"
        f"- target_b ({groups[TARGET_B]['name']}): {ids(groups[TARGET_B]['evidence'])}; chiều hỗ trợ: {', '.join(target_b_dimensions) or 'không rõ'}\n"
        f"- shared: {ids(groups['shared_evidence'])}\n"
        f"- unknown: {ids(groups['unknown_evidence'])}\n"
        f"- chiều so sánh có hỗ trợ hai phía: {', '.join(two_sided_dimensions) or 'không có'}\n"
        f"- thông tin một phía target_a: {', '.join(one_sided.get(TARGET_A, [])) or 'không có'}\n"
        f"- thông tin một phía target_b: {', '.join(one_sided.get(TARGET_B, [])) or 'không có'}\n"
        f"{limited_instruction}\n"
        "Quy tắc: bằng chứng target_a chỉ dùng cho phần riêng của target_a; target_b chỉ dùng cho phần riêng của target_b; "
        "shared có thể dùng cho điểm giống nhau; unknown không được ép gán cho target nào."
    )


def _natural_opening_instruction() -> str:
    return (
        "Yêu cầu văn phong:\n"
        "Trả lời trực tiếp bằng tiếng Việt tự nhiên và tổng hợp tài liệu một cách thầm lặng. Không dùng lời dẫn chung chung như "
        "\"Theo tài liệu\", \"Tài liệu nêu\", \"Tài liệu cho thấy\", \"Các nguồn cho thấy\", "
        "\"Dựa trên các nguồn được cung cấp\"; không nhắc đến retrieval, evidence, bằng chứng đã chọn/kiểm chứng hoặc cách câu trả lời được tạo. "
        "Vẫn giữ nguyên cách dẫn một sử liệu, tác giả hay hồi ký có tên riêng khi tài liệu nêu rõ."
    )


def _history_retry_instruction(
    *,
    retry_reason: str,
    previous_quality_issues: list[str] | None,
) -> str:
    issue_text = ", ".join(previous_quality_issues or []) or retry_reason
    return (
        "Yêu cầu trả lời lại:\n"
        f"- Lần trả lời trước còn thiếu thông tin quan trọng đã có trong cùng tài liệu tham khảo ({issue_text}).\n"
        "- Chỉ dùng các tài liệu tham khảo bên dưới; không thêm dữ kiện ngoài nguồn.\n"
        "- Trả lời đúng câu hỏi lịch sử của người dùng, tổng hợp các ý trực tiếp liên quan và không trùng lặp.\n"
        "- Giải thích quan hệ giữa các ý khi tài liệu hỗ trợ; không nhắc đến quy trình truy xuất hay đánh giá bằng chứng.\n"
        "- Giữ đúng định dạng output: Nguồn được dùng: ... rồi Trả lời: ..."
    )


def build_history_answerer_user_text(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    answer_depth: str = "standard",
    avoid_generic_source_prefix: bool = False,
    retry_reason: str | None = None,
    previous_quality_issues: list[str] | None = None,
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
    policy_parts: list[str] = []
    if answer_depth == "deep":
        policy_parts.append(_deep_depth_instruction(question))
    if answer_depth == "deep" and len(extract_comparison_targets(question)) >= 2:
        policy_parts.append(_comparison_group_instruction(question, evidence).strip())
    if avoid_generic_source_prefix:
        policy_parts.append(_natural_opening_instruction())
    if retry_reason:
        policy_parts.append(
            _history_retry_instruction(
                retry_reason=retry_reason,
                previous_quality_issues=previous_quality_issues,
            )
        )

    policy = "\n\n".join(part for part in policy_parts if part.strip())
    question_block = f"{QUESTION_PREFIX}\n{question}"
    if policy:
        question_block = f"{question_block}\n\n{policy}"
    return f"{question_block}\n\n{REFERENCES_PREFIX}\n{references}"


def build_history_answerer_messages(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    answer_depth: str = "standard",
    avoid_generic_source_prefix: bool = False,
    retry_reason: str | None = None,
    previous_quality_issues: list[str] | None = None,
) -> list[dict[str, str]]:
    # Canonical History SFT contains no system or conversation-history message.
    return [
        {
            "role": "user",
            "content": build_history_answerer_user_text(
                question,
                evidence,
                answer_depth=answer_depth,
                avoid_generic_source_prefix=avoid_generic_source_prefix,
                retry_reason=retry_reason,
                previous_quality_issues=previous_quality_issues,
            ),
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
