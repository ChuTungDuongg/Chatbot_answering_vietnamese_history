"""Conservative guardrails for untrusted evidence and grounded output.

Pattern checks are a supplementary filter, not a guarantee against injection.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


_INJECTION = re.compile(
    r"(?:ignore|disregard|override|forget)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+"
    r"(?:instructions?|prompts?|messages?)|"
    r"(?:reveal|print|leak|exfiltrate|send)\s+(?:the\s+)?(?:system\s+prompt|api\s+key|secret|password)|"
    r"(?:bo qua|quen|ghi de)\s+(?:tat ca\s+)?(?:chi dan|huong dan|lenh)\s+(?:truoc|he thong)|"
    r"(?:tiet lo|in ra|gui)\s+(?:mat khau|api key|system prompt|bi mat)|"
    r"<\|(?:im_start|im_end|system|assistant|developer)\|>|"
    r"</?(?:system|developer|tool_call|function_call|plan_status|think)>|"
    r"\[(?:system|developer|assistant)\]\s*:", re.I,
)
_ID = re.compile(r"^[\w:./-]{1,200}$", re.UNICODE)


class GuardrailError(ValueError):
    """A safe, user-facing guardrail rejection."""


def _normalized(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).replace("đ", "d")


def suspicious(value: Any) -> bool:
    return bool(_INJECTION.search(_normalized(str(value or ""))))


def guard_question(question: str) -> None:
    if suspicious(question):
        raise GuardrailError("Câu hỏi chứa yêu cầu thay đổi chỉ dẫn hoặc tiết lộ thông tin bí mật.")


def safe_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in contexts:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "")
        text = str(row.get("text") or "")
        title = str(row.get("title") or "")
        if not _ID.fullmatch(chunk_id) or not text.strip() or suspicious(text) or suspicious(title):
            continue
        result.append(row)
    return result


def safe_observation(contexts: list[dict[str, Any]], error: str | None = None) -> str:
    import json
    if error:
        return json.dumps({"success": False, "error": error if error in {
            "tool_not_allowed", "external_evidence_required"} else "tool_failed"})
    evidence = [{"chunk_id": str(c["chunk_id"]), "source_kind": str(c.get("source_kind") or "history"),
                 "title": str(c.get("title") or "")[:250], "text": str(c.get("text") or "")[:1500],
                 "metadata": {k: v for k, v in (c.get("metadata") if isinstance(c.get("metadata"), dict) else {}).items()
                              if k in {"page_id", "language", "truncated"} and isinstance(v, (str, int, bool))}}
                for c in contexts[:4]]
    return json.dumps({"untrusted_evidence": evidence}, ensure_ascii=False)


class HiddenMarkupFilter:
    """Suppress explicit hidden-planning/tool markup, including split deltas."""
    tags = ("think", "tool_call", "function_call", "plan_status")

    def __init__(self):
        self.pending = ""
        self.hidden = None

    def feed(self, text: str, *, final: bool = False) -> str:
        self.pending += text
        out = []
        while self.pending:
            lower = self.pending.lower()
            if self.hidden:
                closing = f"</{self.hidden}>"
                end = lower.find(closing)
                if end < 0:
                    self.pending = "" if final else self.pending[-(len(closing) - 1):]
                    break
                self.pending = self.pending[end + len(closing):]
                self.hidden = None
                continue
            start = lower.find("<")
            if start < 0:
                out.append(self.pending)
                self.pending = ""
                break
            out.append(self.pending[:start])
            self.pending = self.pending[start:]
            lower = self.pending.lower()
            openings = tuple(f"<{tag}>" for tag in self.tags)
            match = next((tag for tag in self.tags if lower.startswith(f"<{tag}>")), None)
            if match:
                self.hidden = match
                self.pending = self.pending[len(match) + 2:]
            elif not final and any(tag.startswith(lower) for tag in openings):
                break
            else:
                out.append(self.pending[0])
                self.pending = self.pending[1:]
        return "".join(out)
