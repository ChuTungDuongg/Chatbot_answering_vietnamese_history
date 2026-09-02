from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_COMPARISON_PATTERNS = (
    re.compile(r"\bso\s+s[aá]nh\s+(.+?)\s+(?:v[aà]|v[oớ]i)\s+(.+?)(?:[.?!]|$)", re.I),
    re.compile(r"\bđ[oố]i\s+chi[eế]u\s+(.+?)\s+(?:v[aà]|v[oớ]i)\s+(.+?)(?:[.?!]|$)", re.I),
    re.compile(r"\b(?:đi[eể]m\s+)?gi[oố]ng\s+v[aà]\s+kh[aá]c\s+(?:nhau\s+)?gi[uữ]a\s+(.+?)\s+v[aà]\s+(.+?)(?:[.?!]|$)", re.I),
    re.compile(r"\bs[uự]\s+kh[aá]c\s+bi[eệ]t\s+gi[uữ]a\s+(.+?)\s+v[aà]\s+(.+?)(?:[.?!]|$)", re.I),
    re.compile(r"\b(.+?)\s+kh[aá]c\s+(.+?)\s+nh[uư]\s+th[eế]\s+n[aà]o(?:[.?!]|$)", re.I),
    re.compile(r"\b(.+?)\s+v[aà]\s+(.+?)\s+c[oó]\s+g[iì]\s+gi[oố]ng\s*/?\s*kh[aá]c\s+nhau(?:[.?!]|$)", re.I),
)

_CAUSE_CUES = (
    "nguyen nhan", "vi sao", "tai sao", "boi canh", "dieu kien",
    "dan den", "thuc day", "hinh thanh", "ra doi",
)
_SIGNIFICANCE_CUES = (
    "y nghia", "vai tro", "tac dong", "anh huong", "tam quan trong",
    "dong gop", "gia tri lich su",
)
_CONSEQUENCE_CUES = ("he qua", "ket qua", "hau qua", "dan toi", "de lai")
_EVALUATION_CUES = ("danh gia", "nhan xet", "binh luan", "nhin nhan")
_COMPARISON_CUES = (
    "so sanh", "doi chieu", "giong va khac", "khac biet", "giong/khac",
)


@dataclass(frozen=True)
class CentralQuestionAnalysis:
    question: str
    question_type: str | None
    analytical: bool
    comparison_targets: tuple[str, ...] = ()
    subject: str | None = None


def _ascii_fold_vietnamese(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
        if not unicodedata.combining(char)
    )


_BIOGRAPHY_TOPIC = r"(?:tieu su|cuoc doi|su nghiep)"
_BIOGRAPHY_PATTERNS = (
    re.compile(r"(?P<subject>[^,;?!.]+?)\s+(?:la\s+ai|(?:da\s+|tung\s+)?giu\s+chuc\s+vu\s+gi)\b"),
    re.compile(rf"\b{_BIOGRAPHY_TOPIC}(?:\s+va\s+{_BIOGRAPHY_TOPIC})*\s+(?:cua\s+)?(?P<subject>[^,;?!.]+)"),
    re.compile(r"\b(?:hoat dong|vai tro|(?:nhung\s+|cac\s+)?chuc vu)\s+cua\s+(?P<subject>[^,;?!.]+)"),
)
_SUBJECT_PREFIX = re.compile(
    r"^(?:(?:hay|xin|cho toi biet|cho toi|cho biet|noi ve|ke ve|tom tat|gioi thieu|ve|ong|ba|nhan vat)\s+)+"
)
_SUBJECT_END = re.compile(r"\s+(?:va|trong|doi voi|la|co|nhu the nao|tung|da)\b")
_NON_PERSON_START = re.compile(
    r"^(?:ong ta|ba ta|ai|lich su|chien thang|cuoc|cach mang|nha|quoc gia|chinh phu|phong trao|su kien)\b"
)
_ACCENTED_NON_PERSON_START = re.compile(r"^(?:trận|họ|đảng|triều đại)\b", re.I)


def extract_biography_subject(question: str) -> str | None:
    # NFC makes Vietnamese folding length-preserving, so captures retain original accents.
    compact = unicodedata.normalize("NFC", " ".join(question.split()))
    folded = _ascii_fold_vietnamese(compact)
    for pattern in _BIOGRAPHY_PATTERNS:
        for match in pattern.finditer(folded):
            start, end = match.span("subject")
            prefix = _SUBJECT_PREFIX.match(folded[start:end])
            if prefix:
                start += prefix.end()
            suffix = _SUBJECT_END.search(folded[start:end])
            if suffix:
                end = start + suffix.start()
            subject = compact[start:end].strip(" :\"'“”")
            words = subject.split()
            if (
                2 <= len(words) <= 7
                and not _NON_PERSON_START.match(_ascii_fold_vietnamese(subject))
                and not _ACCENTED_NON_PERSON_START.match(subject)
            ):
                return subject
    return None


def _clean_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,;:-")
    value = re.sub(r"^(?:giữa|hai|sự kiện|nhân vật|chủ thể)\s+", "", value, flags=re.I).strip()
    value = re.sub(
        r"\s+(?:về|xét\s+về|theo\s+(?:các\s+)?tiêu\s+chí|trên\s+(?:các\s+)?phương\s+diện)\s+.+$",
        "",
        value,
        flags=re.I,
    ).strip(" ,;:-")
    return value


def extract_comparison_targets(question: str) -> tuple[str, ...]:
    compact = " ".join(question.split())
    for pattern in _COMPARISON_PATTERNS:
        match = pattern.search(compact)
        if not match:
            continue
        first = _clean_target(match.group(1))
        second = _clean_target(match.group(2))
        if first and second and first.casefold() != second.casefold():
            return (first, second)
    return ()


def analyze_central_question(question: str) -> CentralQuestionAnalysis:
    normalized = _ascii_fold_vietnamese(" ".join(question.split()))
    targets = extract_comparison_targets(question)
    subject = extract_biography_subject(question)
    question_type: str | None = None
    if targets or any(cue in normalized for cue in _COMPARISON_CUES):
        question_type = "comparison"
    elif subject or re.search(rf"\b{_BIOGRAPHY_TOPIC}\b|\bla ai\b|\bgiu chuc vu gi\b", normalized):
        question_type = "biography"
    elif any(cue in normalized for cue in _CAUSE_CUES):
        question_type = "cause"
    elif any(cue in normalized for cue in _CONSEQUENCE_CUES):
        question_type = "consequence"
    elif any(cue in normalized for cue in _SIGNIFICANCE_CUES):
        question_type = "significance"
    elif any(cue in normalized for cue in _EVALUATION_CUES):
        question_type = "evaluation"
    analytical = question_type in {"comparison", "cause", "consequence", "significance", "evaluation", "biography"}
    return CentralQuestionAnalysis(
        question=question,
        question_type=question_type,
        analytical=analytical,
        comparison_targets=targets,
        subject=subject if question_type == "biography" else None,
    )


def build_research_instruction(analysis: CentralQuestionAnalysis, allowed_tools: set[str]) -> str:
    if "search_history" in allowed_tools:
        preferred = "search_history"
    elif "search_wikipedia" in allowed_tools:
        preferred = "search_wikipedia"
    else:
        preferred = sorted(allowed_tools)[0] if allowed_tools else "một công cụ nghiên cứu phù hợp"

    if analysis.question_type == "comparison" and len(analysis.comparison_targets) >= 2:
        first, second = analysis.comparison_targets[:2]
        return (
            "Đây là câu hỏi phân tích so sánh và chưa có bằng chứng. "
            f"Hãy dùng {preferred} trước, bảo đảm có bằng chứng cho cả hai mục tiêu: "
            f"{first!r} và {second!r}. Chỉ phát tool call hợp lệ."
        )
    return (
        "Đây là câu hỏi phân tích lịch sử và chưa có bằng chứng. "
        f"Hãy dùng {preferred} trước để thu thập nguồn phù hợp. Chỉ phát tool call hợp lệ."
    )


def analytical_answer_issues(
    *,
    analysis: CentralQuestionAnalysis,
    answer: str,
    source_ids: list[str],
    evidence_available: bool,
) -> list[str]:
    if not analysis.analytical:
        return []

    folded = _ascii_fold_vietnamese(answer)
    issues: list[str] = []
    if evidence_available and not source_ids:
        issues.append("missing_valid_citations")
    # Length alone is not a quality failure: evidence may support only a compact answer.

    if analysis.question_type == "comparison":
        missing_targets = [
            target for target in analysis.comparison_targets
            if _ascii_fold_vietnamese(target) not in folded
        ]
        if missing_targets:
            issues.append("comparison_target_missing")
        if not any(cue in folded for cue in ("giong", "tuong dong", "diem chung")):
            issues.append("comparison_similarity_missing")
        if not any(cue in folded for cue in ("khac", "khac biet", "trai lai", "trong khi")):
            issues.append("comparison_difference_missing")
        if not any(cue in folded for cue in ("y nghia", "tac dong", "anh huong", "vai tro", "he qua")):
            issues.append("historical_significance_missing")
        if not any(cue in folded for cue in ("vi", "boi canh", "dan den", "cho thay", "do do", "tu do")):
            issues.append("explanatory_content_missing")
    return list(dict.fromkeys(issues))
