from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+")
ANSWER_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|[;\r\n]+|,\s+(?=(?:chấm dứt|đồng thời|tự\b|hiệu\b|chống\b|"
    r"nhằm\b|do\b|việc\b|kết quả\b|ý nghĩa\b|nguyên nhân\b|tại\b|ở\b|vào\b|ngày\b))|"
    r"\s+và\s+(?=(?:mất\b|qua đời\b))",
    re.I,
)
GENERIC_MISSING_PREFIX = "thieu evidence cho cac phan con lai cua cau hoi"
STOPWORDS = {
    "ai", "bao", "biet", "cho", "co", "cac", "cua", "do", "da", "duoc", "gi", "hay", "khi",
    "la", "ma", "mot", "nao", "nhung", "o", "ra", "sau", "theo", "the", "trong", "tu", "va",
    "vao", "ve", "voi", "doi", "lai", "nay", "do", "su", "viec", "tai", "lieu", "nguon",
}


@dataclass(frozen=True)
class AnswerRequirement:
    key: str
    label: str
    answer_fragment: str
    anchor_terms: tuple[str, ...]
    confidence: str = "high"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageAssessment:
    requirements: tuple[AnswerRequirement, ...]
    supported_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]

    @property
    def full(self) -> bool:
        return bool(self.requirements) and not self.missing_keys

    @property
    def useful(self) -> bool:
        return bool(self.supported_keys)

    @property
    def confident(self) -> bool:
        return bool(self.requirements) and all(item.confidence == "high" for item in self.requirements)

    @property
    def partial(self) -> bool:
        return self.useful and bool(self.missing_keys)

    def as_dict(self) -> dict[str, object]:
        return {
            "requirements": [item.as_dict() for item in self.requirements],
            "supported_components": list(self.supported_keys),
            "missing_components": list(self.missing_keys),
            "full_gold_answer_coverage": self.full,
            "useful_answer_coverage": self.useful,
            "coverage_confident": self.confident,
        }


def accentfold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text).casefold().replace("đ", "d"))
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def content_tokens(text: str) -> set[str]:
    return {
        accentfold(match.group(0))
        for match in TOKEN_RE.finditer(str(text))
        if len(match.group(0)) > 1 and accentfold(match.group(0)) not in STOPWORDS
    }


def _answer_clauses(answer: str) -> list[str]:
    cleaned = re.sub(r"^\s*Theo\s+(?:tài liệu|nguồn)[^,]*,\s*", "", str(answer), flags=re.I)
    clauses = [" ".join(item.split()).strip(" ,") for item in ANSWER_SPLIT_RE.split(cleaned)]
    return [item for item in clauses if item]


def _slot(key: str, label: str, anchors: Iterable[str]) -> tuple[str, str, tuple[str, ...]]:
    return key, label, tuple(sorted(content_tokens(" ".join(anchors))))


def question_slots(question: str) -> list[tuple[str, str, tuple[str, ...]]]:
    folded = accentfold(question)
    raw = question.casefold()
    slots: list[tuple[str, str, tuple[str, ...]]] = []

    impact_match = re.search(r"doi voi\s+(.+?)\s+va\s+(.+?)(?:\?|$)", folded)
    if impact_match:
        for index, target in enumerate(impact_match.groups(), 1):
            target = target.strip(" .,?")
            slots.append(_slot(f"impact_{index}", f"ý nghĩa hoặc tác động đối với {target}", [target]))

    birth_death = bool(re.search(r"\bsinh\b.+\b(?:mất|qua đời)\b", raw))
    if birth_death:
        slots.append(_slot("birth_time", "thời gian sinh", ["sinh ngày tháng năm"] ))
        slots.append(_slot("death_time", "thời gian mất hoặc qua đời", ["mất qua đời ngày tháng năm"] ))
    elif re.search(r"\b(?:khi nao|ngay nao|nam nao|thoi gian|vao nam nao)\b", folded):
        slots.append(_slot("time", "thời gian hoặc ngày tháng được hỏi", ["ngày thời gian khi"] ))
    if re.search(r"\b(?:o dau|tai dau|dia diem|noi nao|khu vuc nao)\b", folded):
        slots.append(_slot("place", "địa điểm được hỏi", ["địa điểm nơi tại ở"] ))
    if re.search(r"\b(?:do\s+ai\s+(?:lãnh đạo|chỉ đạo)|ai\s+(?:lãnh đạo|chỉ đạo))\b", raw):
        leader = "người chỉ đạo sự kiện được hỏi" if "chi dao" in folded else "người lãnh đạo sự kiện được hỏi"
        slots.append(_slot("person_leader", leader, ["lãnh đạo chỉ đạo người do"] ))
    elif re.search(r"\b(?:do\s+ai|ai\s+làm|người nào)\b", raw):
        slots.append(_slot("person", "người hoặc nhân vật được hỏi", ["người do làm gồm"] ))
    elif re.search(r"\bai\b", raw):
        slots.append(_slot("person", "người hoặc nhân vật được hỏi", ["người ai"] ))
    if re.search(r"\bnhan vat\b", folded):
        slots.append(_slot("person", "người hoặc nhân vật được hỏi", ["nhân vật người"] ))
    if re.search(r"chong(?: lai)?\s+(?:ai|luc luong nao|doi tuong nao)", folded):
        slots.append(_slot("opponent", "lực lượng hoặc đối tượng bị chống lại", ["chống lực lượng đối tượng"] ))
    if re.search(r"(?:\bcó\b|\bmang\b|\blấy\b).{0,30}\bhiệu\b|\bhiệu\s+và\s+tự\b", raw):
        slots.append(_slot("style_name", "hiệu của nhân vật", ["hiệu"] ))
    if re.search(r"\bhiệu\s+và\s+tự\b|\btự\s+là\s+gì\b|(?:\bcó\b|\bmang\b|\blấy\b).{0,30}\btự\b", raw):
        slots.append(_slot("courtesy_name", "tự của nhân vật", ["tự"] ))
    if re.search(r"\bnguyen nhan\b", folded):
        slots.append(_slot("cause", "nguyên nhân được hỏi", ["nguyên nhân vì bởi do"] ))
    if not impact_match and re.search(r"\by nghia\b", folded):
        slots.append(_slot("significance", "ý nghĩa được hỏi", ["ý nghĩa tác động vai trò"] ))
    if re.search(r"\bdien bien\b", folded):
        slots.append(_slot("development", "diễn biến được hỏi", ["diễn biến tiến triển"] ))
    if not impact_match and re.search(r"\bket qua\b", folded):
        slots.append(_slot("result", "kết quả được hỏi", ["kết quả dẫn đến đạt được"] ))
    if re.search(r"\b(?:muc dich|nham.+?gi)\b", folded):
        slots.append(_slot("purpose", "mục đích được hỏi", ["mục đích nhằm để"] ))
    if re.search(r"\b(?:chuc vu|vai tro|nhiem vu)\b", folded):
        slots.append(_slot("position", "chức vụ, vai trò hoặc nhiệm vụ được hỏi", ["chức vụ vai trò nhiệm vụ phong làm giữ chức"] ))
    if re.search(r"\b(?:bao nhieu|so luong|may nguoi|may lan)\b", folded):
        slots.append(_slot("count", "số lượng được hỏi", ["số lượng bao nhiêu"] ))
    if re.search(r"\b(?:su kien nao|tran nao|chien dich nao)\b", folded):
        slots.append(_slot("event", "sự kiện được hỏi", ["sự kiện trận chiến dịch"] ))

    unique: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    for item in slots:
        unique.setdefault(item[0], item)
    return list(unique.values())


def _slot_cues(key: str) -> set[str]:
    return {
        "time": {"ngay", "thang", "nam", "luc", "vao"},
        "birth_time": {"sinh", "ngay", "thang", "nam"},
        "death_time": {"mat", "qua doi", "ngay", "thang", "nam"},
        "place": {"tai", "o", "dia diem", "noi"},
        "person": {"lanh dao", "chi dao", "do", "nguoi"},
        "person_leader": {"lanh dao", "chi dao", "do", "nguoi"},
        "opponent": {"chong", "danh", "luc luong", "doi tuong"},
        "style_name": {"hieu"},
        "courtesy_name": {"tu"},
        "cause": {"nguyen nhan", " vi ", "boi", " do "},
        "significance": {
            "y nghia", "tac dong", "vai tro", "day la", "cho thay", "danh dau", "giup", "khien",
            "tao", "dat nen", "ton vinh", "khang dinh", "mo ra", "cham dut", "dau moc",
        },
        "development": {"dien bien", "sau do", "tiep theo", "lan luot", "tu ", "den "},
        "result": {
            "ket qua", "dan den", "thang loi", "danh bai", "that bai", "tieu diet", "bi bat", "chiem", "tan vo",
            "rut", "giai phong", "dat duoc", "sup do", "dau hang", "that thu",
        },
        "purpose": {"muc dich", "nham", " de ", "chu truong"},
        "position": {"chuc vu", "phong", "giu"},
        "count": {"so luong", "bao nhieu", "quan", "nguoi", "lan"},
        "event": {"su kien", "tran", "chien dich"},
    }.get(key, set())


def _clause_score(key: str, anchors: tuple[str, ...], clause: str) -> float:
    folded = accentfold(clause)
    tokens = content_tokens(clause)
    score = 3.0 * len(tokens & set(anchors))
    score += 5.0 * sum(1 for cue in _slot_cues(key) if cue in folded)
    if key == "time" and re.search(r"\b\d{1,4}(?:[/.-]\d{1,2})?\b", folded):
        score += 8.0
    if key.startswith("impact_"):
        score += 5.0 * len(tokens & set(anchors))
    return score


def extract_answer_requirements(question: str, gold_answer: str) -> tuple[AnswerRequirement, ...]:
    clauses = _answer_clauses(gold_answer)
    if not clauses:
        return ()
    slots = question_slots(question)
    if not slots:
        return (
            AnswerRequirement(
                key="answer",
                label="nội dung trả lời được hỏi",
                answer_fragment=" ".join(clauses),
                anchor_terms=tuple(sorted(content_tokens(question))),
                confidence="heuristic",
            ),
        )

    requirements: list[AnswerRequirement] = []
    for key, label, anchors in slots:
        ranked = sorted(
            ((_clause_score(key, anchors, clause), index, clause) for index, clause in enumerate(clauses)),
            key=lambda item: (-item[0], item[1]),
        )
        has_confident_match = bool(ranked and ranked[0][0] > 0)
        fragment = ranked[0][2] if has_confident_match else " ".join(clauses)
        requirements.append(
            AnswerRequirement(
                key=key,
                label=label,
                answer_fragment=fragment,
                anchor_terms=anchors,
                confidence="high" if has_confident_match else "heuristic",
            )
        )
    return tuple(requirements)


def requirement_supported(
    requirement: AnswerRequirement,
    *,
    question: str,
    evidence_text: str,
) -> bool:
    fragment_folded = " ".join(accentfold(requirement.answer_fragment).split())
    evidence_folded = " ".join(accentfold(evidence_text).split())
    if not fragment_folded:
        return False
    semantic_predicates = {
        "person_leader": ("lanh dao", "chi dao"),
        "opponent": ("chong",),
        "cause": (" vi ", "boi", " do ", "nguyen nhan"),
        "purpose": ("nham", " de ", "muc dich", "chu truong"),
    }
    required_predicates = semantic_predicates.get(requirement.key)
    padded_evidence = f" {evidence_folded} "
    if required_predicates and not any(item in padded_evidence for item in required_predicates):
        return False
    raw_evidence = str(evidence_text).casefold()
    if requirement.key == "style_name" and not re.search(r"\bhiệu\b", raw_evidence):
        return False
    if requirement.key == "courtesy_name" and not re.search(r"\btự\b", raw_evidence):
        return False
    if fragment_folded and fragment_folded in evidence_folded:
        return True

    question_values = content_tokens(question)
    fragment_values = content_tokens(requirement.answer_fragment) - question_values
    if not fragment_values:
        fragment_values = content_tokens(requirement.answer_fragment)
    evidence_values = content_tokens(evidence_text)
    numeric_values = set(re.findall(r"\b\d{1,4}\b", fragment_folded)) - set(re.findall(r"\b\d{1,4}\b", accentfold(question)))
    if numeric_values and not numeric_values <= set(re.findall(r"\b\d{1,4}\b", evidence_folded)):
        return False
    overlap = len(fragment_values & evidence_values)
    recall = overlap / max(len(fragment_values), 1)
    minimum = 1 if len(fragment_values) <= 2 else 2
    return overlap >= minimum and recall >= 0.55


def assess_answer_coverage(question: str, gold_answer: str, evidence_texts: Iterable[str]) -> CoverageAssessment:
    requirements = extract_answer_requirements(question, gold_answer)
    combined = "\n".join(str(item) for item in evidence_texts if str(item).strip())
    supported = tuple(
        item.key for item in requirements if requirement_supported(item, question=question, evidence_text=combined)
    )
    missing = tuple(item.key for item in requirements if item.key not in supported)
    return CoverageAssessment(requirements=requirements, supported_keys=supported, missing_keys=missing)


def specific_missing_information(assessment: CoverageAssessment) -> list[str]:
    by_key = {item.key: item for item in assessment.requirements}
    return [f"Thiếu evidence về {by_key[key].label}." for key in assessment.missing_keys]


def evidence_relevance(question: str, evidence_text: str) -> float:
    """Question relevance only; it intentionally does not encode row status."""
    query = content_tokens(question)
    if not query:
        return 0.5
    overlap = len(query & content_tokens(evidence_text)) / len(query)
    return round(min(1.0, 0.5 + 0.5 * math.sqrt(overlap)), 4)
