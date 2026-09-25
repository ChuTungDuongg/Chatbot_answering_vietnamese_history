"""Surface-form answer similarity; these scores are not historical truth checks."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if token == other
                           else max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def score_answer(answer: str, gold_answer: str | None) -> dict[str, float | None]:
    names = ("exact_match", "token_precision", "token_recall", "token_f1", "rouge_l_f1")
    if gold_answer is None:
        return dict.fromkeys(names)
    predicted, gold = tokens(answer), tokens(gold_answer)
    if not predicted and not gold:
        return dict.fromkeys(names, 1.0)
    if not predicted or not gold:
        return dict.fromkeys(names, 0.0)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    precision, recall = overlap / len(predicted), overlap / len(gold)
    lcs = _lcs_length(predicted, gold)
    rouge_precision, rouge_recall = lcs / len(predicted), lcs / len(gold)
    return {"exact_match": float(predicted == gold), "token_precision": precision,
            "token_recall": recall,
            "token_f1": 2 * precision * recall / (precision + recall) if overlap else 0.0,
            "rouge_l_f1": 2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
            if lcs else 0.0}
