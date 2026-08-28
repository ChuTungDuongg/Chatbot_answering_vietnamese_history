from __future__ import annotations

from typing import Any

from app.chat.attachments import is_attachment_question
from app.rag.retrieval import clean_text, match_norm


class ResearchRetrievalRuntime:
    """Active retrieval utilities with no answer prompts or generation passes."""

    def __init__(self, service: Any, retriever: Any):
        self.service = service
        self.retriever = retriever

    def _cfg(self, section: str, name: str, default: Any) -> Any:
        config = self.service.config or {}
        values = config.get(section, {}) or {}
        return values.get(name, default)

    @property
    def max_history_messages(self) -> int:
        return max(0, int(self._cfg("prompt", "max_history_messages", 6)))

    @property
    def retrieval_history_messages(self) -> int:
        return max(0, int(self._cfg("generation", "retrieval_history_messages", 4)))

    @property
    def temporary_min_dense_score(self) -> float:
        return float(self._cfg("generation", "temporary_min_dense_score", 0.72))

    @staticmethod
    def normalize_history(
        history: list[dict[str, str]] | None,
        current_question: str | None = None,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in history or []:
            role = str(message.get("role", "")).strip().lower()
            content = clean_text(message.get("content", ""))
            if role in {"user", "assistant"} and content:
                normalized.append({"role": role, "content": content})

        if (
            current_question
            and normalized
            and normalized[-1]["role"] == "user"
            and match_norm(normalized[-1]["content"]) == match_norm(current_question)
        ):
            normalized.pop()
        return normalized

    @staticmethod
    def needs_history_for_retrieval(
        question: str,
        history: list[dict[str, str]],
    ) -> bool:
        if not history:
            return False
        normalized = match_norm(question)
        reference_terms = {
            "ong ay", "ba ay", "nguoi ay", "nguoi nay", "nhan vat nay",
            "nhan vat do", "su kien nay", "su kien do", "tran nay", "tran do",
            "trieu dai nay", "trieu dai do", "thoi ky nay", "thoi ky do",
            "dieu nay", "dieu do", "viec nay", "viec do", "tai lieu nay",
            "file nay", "pdf nay", "hinh nay", "anh nay", "noi dung tren",
            "nhan vat tren", "su kien tren", "cau tra loi tren", "nhu vua noi",
            "nhu tren", "truoc do", "sau do", "khi do", "luc do",
        }
        if any(term in normalized for term in reference_terms):
            return True
        return normalized.startswith((
            "con ", "vay ", "the ", "tiep theo", "sau nay", "sau do",
            "truoc do", "tai sao lai", "vi sao lai", "sinh nam nao",
            "mat nam nao", "dien ra khi nao", "xay ra khi nao",
            "ket thuc khi nao", "bat dau khi nao",
        ))

    def build_retrieval_question(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> tuple[str, bool]:
        if not self.needs_history_for_retrieval(question, history):
            return question, False
        recent = history[-self.retrieval_history_messages:]
        lines = [
            f"{'Người dùng' if item['role'] == 'user' else 'Trợ lý'}: "
            f"{clean_text(item['content'])[:600]}"
            for item in recent
        ]
        if not lines:
            return question, False
        return (
            "Ngữ cảnh hội thoại trước đó:\n"
            + "\n".join(lines)
            + "\n\nCâu hỏi hiện tại:\n"
            + question,
            True,
        )

    def temporary_context_is_relevant(
        self,
        question: str,
        contexts: list[dict[str, Any]],
    ) -> bool:
        if not contexts:
            return False
        if is_attachment_question(question):
            return True
        best = max(
            (float(item.get("temporary_dense_score", -1.0) or -1.0) for item in contexts),
            default=-1.0,
        )
        return best >= self.temporary_min_dense_score
