from __future__ import annotations

import re
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


def _ascii_fold_vietnamese(text: str) -> str:
    replacements = {
        "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a", "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a", "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e", "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o", "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o", "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u", "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y", "đ": "d",
    }
    return "".join(replacements.get(ch, ch) for ch in text.casefold())


def _clean_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,;:-")
    value = re.sub(r"^(?:giữa|hai|sự kiện|nhân vật|chủ thể)\s+", "", value, flags=re.I).strip()
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
    question_type: str | None = None
    if targets or any(cue in normalized for cue in _COMPARISON_CUES):
        question_type = "comparison"
    elif any(cue in normalized for cue in _CAUSE_CUES):
        question_type = "cause"
    elif any(cue in normalized for cue in _CONSEQUENCE_CUES):
        question_type = "consequence"
    elif any(cue in normalized for cue in _SIGNIFICANCE_CUES):
        question_type = "significance"
    elif any(cue in normalized for cue in _EVALUATION_CUES):
        question_type = "evaluation"
    analytical = question_type in {"comparison", "cause", "consequence", "significance", "evaluation"}
    return CentralQuestionAnalysis(
        question=question,
        question_type=question_type,
        analytical=analytical,
        comparison_targets=targets,
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
    words = answer.split()
    issues: list[str] = []
    if evidence_available and not source_ids:
        issues.append("missing_valid_citations")
    if len(words) < 120:
        issues.append("analytical_answer_too_shallow")

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
