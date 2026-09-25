"""Deterministic evidence checks; lexical support is not semantic entailment."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


YEAR = re.compile(r"(?<!\d)(?:[5-9]\d{2}|1\d{3}|20\d{2})(?!\d)")
REFUSAL = re.compile(
    r"không (?:có )?đủ (?:thông tin|bằng chứng|tư liệu)|không thể (?:xác minh|trả lời)|"
    r"ngoài phạm vi|insufficient (?:evidence|information)|cannot verify|outside (?:the )?scope",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def score_grounding(answer: str, question: str, sources: list[dict[str, Any]], *,
                    required_facts: list[str] | None = None,
                    answerable: bool | None = None,
                    in_domain: bool | None = None) -> dict[str, Any]:
    answer_years = sorted(set(YEAR.findall(answer)))
    evidence_texts = [str(source["text"]) for source in sources if source.get("text")]
    observed_years = set(YEAR.findall("\n".join(evidence_texts)))
    unsupported_years = sorted(set(answer_years) - observed_years) if evidence_texts else None
    normalized_answer = _normalize(answer)
    required_fact_recall = None
    if required_facts:
        required_fact_recall = sum(_normalize(fact) in normalized_answer for fact in required_facts) / len(required_facts)
    refused = bool(REFUSAL.search(answer))
    return {
        "required_fact_phrase_recall": required_fact_recall,
        "answer_years": answer_years,
        "unverified_years": unsupported_years,
        "unverified_year_rate": len(unsupported_years) / len(answer_years)
        if unsupported_years is not None and answer_years else None,
        "insufficient_answer_behavior": float(refused) if answerable is False else None,
        "ood_refusal_behavior": float(refused) if in_domain is False else None,
        "unnecessary_refusal": float(refused) if answerable is True else None,
    }
