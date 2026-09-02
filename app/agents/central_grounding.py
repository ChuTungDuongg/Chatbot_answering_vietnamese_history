from __future__ import annotations

import re

from app.agents.central_citations import BRACKET_RE
from app.agents.central_evidence import SynthesisEvidence, normalize_entity


_WORD = r"[^\W\d_]+"
_WORDS_RE = re.compile(_WORD, re.UNICODE)
_PERSON_STARTS = set("nguyen tran le ly ngo dinh phan pham ho hoang huynh vu vo dang do bui truong trinh duong mai dao ton luu cao mac".split())
_STATE_STARTS = {"dai", "viet", "nam", "au", "van"}
_DYNASTY_RE = re.compile(r"\b(?:nhà|triều đại|triều)\s+([^\W\d_]+)", re.I)
_OFFICES_RE = re.compile(
    r"\b(?:(?:phó|pho)\s+)?(?:tổng thống|tong thong|thủ tướng|thu tuong|hoàng đế|hoang de|"
    r"hoàng hậu|hoang hau|quốc trưởng|quoc truong|tổng bí thư|tong bi thu|"
    r"chủ tịch|chu tich|bộ trưởng|bo truong|đại tướng|dai tuong|thống tướng|thong tuong|"
    r"tư lệnh|tu lenh|toàn quyền|toan quyen)\b", re.I,
)
_GOVERNMENT_RE = re.compile(
    r"\b(?:Việt Nam (?:Dân chủ Cộng hòa|Cộng hòa)|Cộng hòa (?:miền Nam Việt Nam|Xã hội chủ nghĩa Việt Nam))\b",
    re.I,
)
_EVENT_RE = re.compile(
    r"\b(?:trận|chiến thắng|chiến dịch|hiệp định|cách mạng)\s+", re.I,
)


def _named_tokens(answer: str) -> list[str]:
    found = [match.group(0) for pattern in (_OFFICES_RE, _GOVERNMENT_RE) for match in pattern.finditer(answer)]
    found.extend(
        match.group(0) for match in _DYNASTY_RE.finditer(answer)
        if normalize_entity(match.group(1)) in _PERSON_STARTS | {"duong", "tong", "minh", "thanh", "han", "tuy", "nguyen"}
    )
    words = list(_WORDS_RE.finditer(answer))
    for i, word in enumerate(words):
        if normalize_entity(word.group()) not in _PERSON_STARTS | _STATE_STARTS or not word.group()[0].isupper():
            continue
        end = i + 1
        while end < min(len(words), i + 5):
            next_word = words[end]
            if not next_word.group()[0].isupper() or answer[words[end - 1].end():next_word.start()] != " ":
                break
            end += 1
        if end - i >= 2:
            found.append(answer[word.start():words[end - 1].end()])
    for match in _EVENT_RE.finditer(answer):
        tail = answer[match.end():]
        tokens = list(_WORDS_RE.finditer(tail))
        end = 0
        for token in tokens[:5]:
            if not token.group()[0].isupper() or tail[end:token.start()].strip():
                break
            end = token.end()
        if end:
            found.append(answer[match.start():match.end() + end])
    return list(dict.fromkeys(found))


def grounding_risks(answer: str, question: str, packet: list[SynthesisEvidence]) -> dict[str, list[str]]:
    """Conservative token-presence risk signal, not entailment or a fact verdict.

    Only the question and model-visible selected text/title count as support.
    Citation IDs, discarded rows, and conversation history cannot confer support.
    """
    support = " ".join([question, *(f"{item.title} {item.text}" for item in packet)])
    normalized_support = f" {normalize_entity(support)} "
    prose = BRACKET_RE.sub(lambda m: m.group(0) if m.group(1).isdigit() else "", answer)
    years = []
    for match in re.finditer(r"(?<!\w)[1-9]\d{2,3}(?!\w)", prose):
        # A duration such as "hơn 1000 năm" is not a calendar year.
        if re.match(r"\s+năm\b", prose[match.end():], re.I):
            continue
        if not re.search(rf"(?<!\d){match.group()}(?!\d)", support):
            years.append(match.group())
    names = [name for name in _named_tokens(prose) if f" {normalize_entity(name)} " not in normalized_support]
    return {
        "unsupported_named_claims": list(dict.fromkeys(names)),
        "unsupported_years": list(dict.fromkeys(years)),
    }
