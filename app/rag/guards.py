import re
from typing import Any

import numpy as np

from app.rag.retrieval import clean_text, match_norm
from app.services.rag_service import RAGService


REFUSAL_PATTERNS = [
    r"không đủ (?:thông tin|bằng chứng|dữ liệu)",
    r"tài liệu (?:được cung cấp )?(?:không|chưa) (?:nêu|cho biết|cung cấp)",
    r"không thể trả lời (?:chắc chắn|chính xác)?",
    r"ngoài phạm vi",
    r"không liên quan đến lịch sử việt nam",
]


ISSUE_INSTRUCTIONS = {
    "empty_answer": "Câu trả lời bị rỗng.",
    "boilerplate_opening": 'Bỏ cách mở đầu rập khuôn như "Theo tài liệu".',
    "technical_chunk_word": 'Không dùng từ kỹ thuật "chunk".',
    "winner_not_answered_directly": (
        "Phải trả lời trực tiếp bên nào thắng/giành ưu thế, "
        "hoặc nói evidence không cho phép kết luận tuyệt đối."
    ),
    "missing_outcome": "Bổ sung phần kết quả/kết cục được evidence hỗ trợ.",
    "missing_significance": "Bổ sung phần ý nghĩa/tác động được evidence hỗ trợ.",
    "missing_context": "Bổ sung bối cảnh/hoàn cảnh dẫn tới sự kiện.",
    "comparison_not_explicit": "So sánh rõ từng bên và điểm giống/khác; không trộn thông tin.",
    "missing_document_content": "Nêu các nội dung/điều khoản chính được evidence hỗ trợ.",
    "multi_part_answer_too_short": "Câu hỏi có nhiều ý; trả lời đủ từng ý.",
    "invalid_source_id": "Chỉ dùng chunk_id có trong evidence.",
    "missing_source": "Phải ghi ít nhất một source nếu đưa ra factual answer.",
    "unsupported_year": "Loại hoặc sửa niên đại không có trong evidence.",
}


class AnswerGuards:
    def __init__(self, service: RAGService):
        self.service = service

    # ========================================================
    # Config
    # ========================================================

    @property
    def guard_config(self) -> dict[str, Any]:
        if not self.service.config:
            return {}
        return self.service.config.get("guards", {}) or {}

    @property
    def strict_source_required(self) -> bool:
        return bool(self.guard_config.get("strict_source_required", True))

    @property
    def strict_unsupported_year_guard(self) -> bool:
        return bool(self.guard_config.get("strict_unsupported_year_guard", True))

    @property
    def enable_completeness_rewrite(self) -> bool:
        return bool(self.guard_config.get("enable_completeness_rewrite", True))

    @property
    def max_rewrite_attempts(self) -> int:
        return int(self.guard_config.get("max_rewrite_attempts", 1))

    @property
    def safe_ood_answer(self) -> str:
        return self.guard_config.get(
            "safe_ood_answer",
            "Câu hỏi này nằm ngoài phạm vi hệ thống lịch sử Việt Nam, "
            "nên tôi không trả lời bằng corpus hiện tại.",
        )

    @property
    def safe_insufficient_answer(self) -> str:
        return self.guard_config.get(
            "safe_insufficient_answer",
            "Không đủ bằng chứng trong các tài liệu truy xuất "
            "để trả lời chắc chắn câu hỏi này.",
        )

    # ========================================================
    # Refusal detection
    # ========================================================

    @staticmethod
    def is_refusal(text: str) -> bool:
        text = clean_text(text)

        return any(
            re.search(pattern, text, flags=re.I)
            for pattern in REFUSAL_PATTERNS
        )

    # ========================================================
    # Year extraction
    # ========================================================

    @staticmethod
    def extract_year_set(text: str) -> set[int]:
        return {
            int(value)
            for value in re.findall(r"(?<!\d)(\d{3,4})(?!\d)", clean_text(text))
        }

    # ========================================================
    # Source + year validation
    # ========================================================

    def validate_parsed_answer(
        self,
        parsed: dict[str, Any],
        used_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        allowed_ids = {
            str(chunk["chunk_id"])
            for chunk in used_context
        }

        model_ids = [
            str(source_id)
            for source_id in parsed.get("source_ids", [])
        ]

        valid_ids = [
            source_id
            for source_id in model_ids
            if source_id in allowed_ids
        ]

        invalid_ids = [
            source_id
            for source_id in model_ids
            if source_id not in allowed_ids
        ]

        evidence_chunks = [
            self.service.chunk_by_id[source_id]
            for source_id in valid_ids
            if source_id in self.service.chunk_by_id
        ]

        evidence_text = "\n".join(
            f"{clean_text(chunk.get('title', ''))}\n{clean_text(chunk.get('text', ''))}"
            for chunk in evidence_chunks
        )

        answer = parsed.get("answer", "")

        if evidence_chunks:
            unsupported_years = sorted(
                self.extract_year_set(answer) - self.extract_year_set(evidence_text)
            )
        else:
            unsupported_years = sorted(self.extract_year_set(answer))

        guard_issues: list[str] = []

        if invalid_ids:
            guard_issues.append("invalid_source_id")

        if (
            self.strict_source_required
            and not valid_ids
            and not self.is_refusal(answer)
        ):
            guard_issues.append("missing_source")

        if (
            self.strict_unsupported_year_guard
            and unsupported_years
            and not self.is_refusal(answer)
        ):
            guard_issues.append("unsupported_year")

        return {
            "answer": answer,
            "valid_ids": valid_ids,
            "invalid_ids": invalid_ids,
            "unsupported_years": unsupported_years,
            "evidence_chunks": evidence_chunks,
            "guard_issues": guard_issues,
            "format_ok": parsed.get("format_ok", False),
        }

    # ========================================================
    # Answer quality critic
    # ========================================================

    def critique_answer(
        self,
        question: str,
        answer: str,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        facets = analysis.get("facets", ["general"])
        normalized_answer = match_norm(answer)
        issues: list[str] = []

        if not clean_text(answer):
            return {
                "pass": False,
                "issues": ["empty_answer"],
                "facets": facets,
            }

        if self.is_refusal(answer):
            return {
                "pass": True,
                "issues": [],
                "facets": facets,
            }

        if re.match(r"\s*(theo|dua tren|can cu).*tai lieu", normalized_answer):
            issues.append("boilerplate_opening")

        if re.search(r"\bchunk\b", normalized_answer):
            issues.append("technical_chunk_word")

        if "winner" in facets:
            winner_terms = [
                "gianh chien thang",
                "thang loi",
                "that bai",
                "gianh uu the",
                "uu the quan su",
                "khong the ket luan",
                "khong co mot ben",
                "khong the coi",
                "khong co ben nao",
            ]

            if not any(term in normalized_answer for term in winner_terms):
                issues.append("winner_not_answered_directly")

        if "outcome" in facets:
            outcome_terms = [
                "ket qua",
                "ket thuc",
                "thang loi",
                "that bai",
                "cham dut",
                "gianh",
                "buoc",
                "thanh lap",
                "thoai vi",
                "tao co so",
                "dan den",
                "he qua",
            ]

            if not any(term in normalized_answer for term in outcome_terms):
                issues.append("missing_outcome")

        if "significance" in facets:
            significance_terms = [
                "y nghia",
                "danh dau",
                "khang dinh",
                "mo ra",
                "gop phan",
                "tac dong",
                "vai tro",
                "tao tien de",
                "cung co",
            ]

            if not any(term in normalized_answer for term in significance_terms):
                issues.append("missing_significance")

        if "context" in facets:
            context_terms = [
                "boi canh",
                "hoan canh",
                "truoc khi",
                "sau khi",
                "trong khi",
                "xam luoc",
                "khong chien",
                "duoi su",
                "do ",
                "vi ",
                "khi ",
            ]

            if not any(term in normalized_answer for term in context_terms):
                issues.append("missing_context")

        if "compare" in facets:
            compare_terms = [
                "trong khi",
                "con ",
                "deu ",
                "khac",
                "giong",
                "so voi",
                "ve mat",
                "tuong dong",
            ]

            if not any(term in normalized_answer for term in compare_terms):
                issues.append("comparison_not_explicit")

        if "content" in facets:
            content_terms = [
                "noi dung",
                "quy dinh",
                "dieu khoan",
                "ngung ban",
                "cam ket",
                "trao tra",
                "tong tuyen cu",
            ]

            if not any(term in normalized_answer for term in content_terms):
                issues.append("missing_document_content")

        word_count = len(
            re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", clean_text(answer))
        )

        if analysis.get("is_multi_part") and word_count < 35:
            issues.append("multi_part_answer_too_short")

        issues = list(dict.fromkeys(issues))

        return {
            "pass": not issues,
            "issues": issues,
            "facets": facets,
        }

    # ========================================================
    # Semantic support score
    # ========================================================

    def answer_support_score(
        self,
        answer: str,
        source_chunks: list[dict[str, Any]],
    ) -> float | None:
        if not answer or not source_chunks or self.service.embedder is None:
            return None

        answer_embedding = self.service.embedder.encode(
            ["query: " + answer],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]

        passage_texts = [
            "passage: "
            + clean_text(
                f"{chunk.get('title', '')}\n{chunk.get('text', '')}"
            )
            for chunk in source_chunks
        ]

        passage_embeddings = self.service.embedder.encode(
            passage_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        return float(np.max(passage_embeddings @ answer_embedding))