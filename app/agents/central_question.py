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

CAUSE_MARKERS = ("nguyen nhan", "vi sao", "tai sao", "ly do")
_CAUSE_CUES = (*CAUSE_MARKERS, "boi canh", "dieu kien",
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
    event: str | None = None
    actors: tuple[str, ...] = ()
    outcome: str | None = None
    facets: tuple[str, ...] = ()
    related_entities: tuple[str, ...] = ()
    relation_requested: bool = False
    relation_phrase: str | None = None
    comparison_targets_raw: tuple[str, ...] = ()
    comparison_canonical_targets: tuple[str, ...] = ()
    comparison_target_entity_types: tuple[str | None, ...] = ()
    target_resolution_events: tuple[dict, ...] = ()
    administrative_level: str | None = None
    time_scope: tuple[int, ...] = ()
    freshness_required: bool = False
    freshness_reason: str | None = None
    premise_requires_validation: bool = False
    raw_event_clause: str | None = None
    answer_depth: str = "simple_fact"
    viewpoint_requested: bool = False

    def telemetry(self) -> dict:
        return {
            "question_type": self.question_type, "answer_depth": self.answer_depth, "subject": self.subject,
            "viewpoint_requested": self.viewpoint_requested,
            "event": self.event, "actors": list(self.actors), "outcome": self.outcome,
            "comparison_targets": list(self.comparison_targets),
            "facet": self.facets[0] if self.facets else None, "facets": list(self.facets),
            "related_entities": list(self.related_entities), "relation_requested": self.relation_requested,
            "relation_phrase": self.relation_phrase,
            "comparison_targets_raw": list(self.comparison_targets_raw or self.comparison_targets),
            "comparison_targets_normalized": list(self.comparison_targets),
            "comparison_canonical_targets": list(self.comparison_canonical_targets or self.comparison_targets),
            "comparison_target_entity_types": list(self.comparison_target_entity_types),
            "target_resolution_events": list(self.target_resolution_events),
            "administrative_level": self.administrative_level,
            "requested_administrative_level": self.administrative_level,
            "time_scope": list(self.time_scope), "freshness_required": self.freshness_required,
            "freshness_reason": self.freshness_reason,
            "canonical_event": self.event, "raw_event_clause": self.raw_event_clause,
            "requested_facets": list(self.facets),
            "event_resolution_events": ([{"raw": self.raw_event_clause, "canonical": self.event,
                                          "reason": "analytical_clause_boundary"}] if self.raw_event_clause and self.raw_event_clause != self.event else []),
        }


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
    relation = extract_biography_relation(question)
    if relation:
        subject = relation[0]
    question_type: str | None = None
    if relation:
        question_type = "biography"
        targets = ()
    elif targets or any(cue in normalized for cue in _COMPARISON_CUES):
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
    event, analytical_subject, actors, outcome = extract_analytical_target(question)
    from app.agents.central_administration import administrative_question
    from app.agents.central_targets import resolve_comparison_targets
    admin = administrative_question(question) if not targets and question_type != "biography" else {}
    from app.agents.central_depth import answer_depth
    facets = ("identity", "relationship") if relation else extract_requested_facets(question, question_type)
    return resolve_comparison_targets(CentralQuestionAnalysis(
        question=question,
        question_type=question_type,
        analytical=analytical,
        comparison_targets=targets,
        subject=admin.get("subject", subject if question_type == "biography" else analytical_subject if not targets else None),
        event=admin.get("event", event if not targets and question_type != "biography" else None),
        actors=actors if question_type != "biography" else (),
        outcome=outcome if question_type != "biography" else None,
        facets=facets,
        answer_depth=answer_depth(question_type, event=event, subject=analytical_subject,
                                  facets=facets, administrative=bool(admin)),
        viewpoint_requested=bool(re.search(r"\b(?:quan diem|nhan dinh cua|theo nhan dinh|trich dan|loi noi|phat bieu cua)\b", normalized)),
        related_entities=(relation[1],) if relation else (),
        relation_requested=bool(relation), relation_phrase=relation[2] if relation else None,
        raw_event_clause=raw_event_clause(question) if event else None,
        **{key: value for key, value in admin.items() if key not in {"event", "subject"}},
    ))


def extract_biography_relation(question: str) -> tuple[str, str, str] | None:
    """Bounded two-person grammar, preserving names; no historical equivalences."""
    compact = unicodedata.normalize("NFC", " ".join(question.split())).strip(" .?!")
    folded = _ascii_fold_vietnamese(compact)
    patterns = (
        r"^(?P<a>.+?)\s+(?P<phrase>(?:co )?(?:lien he|quan he)(?: gi| nhu the nao)? voi)\s+(?P<b>.+?)(?: khong)?$",
        r"^(?P<a>.+?)\s+(?P<phrase>(?:co )?tung (?:gap(?: go)?|lam viec|doi dau)(?: voi)?)\s+(?P<b>.+?)(?: khong)?$",
        r"^(?P<a>.+?)\s+va\s+(?P<b>.+?)\s+(?P<phrase>co lien quan gi den nhau)$",
        r"^(?P<phrase>(?:moi )?quan he giua)\s+(?P<a>.+?)\s+va\s+(?P<b>.+)$",
        r"^(?P<a>.+?)\s+(?P<phrase>doi voi)\s+(?P<b>.+)$",
    )
    def name(value):
        value = value.strip(" ,;:")
        biography = extract_biography_subject(value)
        if biography:
            return biography
        value = re.sub(r"^(?:vai trò của|tiểu sử của)\s+", "", value, flags=re.I)
        words = value.split()
        return value if 2 <= len(words) <= 7 and all(word[0].isupper() and word.isalpha() for word in words) else None
    for pattern in patterns:
        match = re.match(pattern, folded)
        if not match:
            continue
        first, second = (name(compact[slice(*match.span(key))]) for key in ("a", "b"))
        if first and second and _ascii_fold_vietnamese(first) != _ascii_fold_vietnamese(second):
            return first, second, compact[slice(*match.span("phrase"))]
    return None


# Grammatical patterns and a small actor-abbreviation vocabulary, not a historical database.
_OUTCOMES = ((r"\b(?:suy yeu|suy thoai)\b", "suy yếu"),
             (r"\b(?:that bai|thua)\b", "thất bại"),
             (r"\b(?:thanh cong|thang loi)\b", "thành công"),
             (r"\b(?:sup do|suy vong)\b", "sụp đổ"))
# Existing abbreviation metadata; applied to any parsed actor rather than to a
# particular question. Unlisted entities keep the user's name verbatim.
ACTOR_ALIASES = ((r"\b(?:my|hoa ky)\b", "Mỹ"), (r"\b(?:vnch|viet nam cong hoa)\b", "Việt Nam Cộng hòa"))


def normalized_actor_text(text):
    folded = _ascii_fold_vietnamese(text)
    for pattern, name in ACTOR_ALIASES:
        folded = re.sub(pattern, _ascii_fold_vietnamese(name), folded)
    return " ".join(re.findall(r"[a-z0-9]+", folded))


def causal_subject_phrase(question):
    """Reusable cause/outcome grammar; names are captures, never policy keys."""
    compact = unicodedata.normalize("NFC", " ".join(question.split())).strip(" .?!")
    folded = _ascii_fold_vietnamese(compact)
    outcomes = "(?:" + "|".join(pattern for pattern, _ in _OUTCOMES) + ")"
    markers = "(?:" + "|".join(CAUSE_MARKERS) + ")"
    patterns = (
        rf"^{markers}(?: nao)?(?: dan (?:den|toi))?\s+(?:su )?{outcomes}\s+cua\s+(?P<subject>.+?)(?:\s+(?:trong|la gi|va he qua)\b|$)",
        rf"^{markers}\s+(?P<subject>.+?)\s+(?:lai\s+)?{outcomes}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, folded)
        if match:
            subject = compact[slice(*match.span("subject"))].strip(" ,;:")
            if re.search(r"\b(?:nay|do|ay)$", _ascii_fold_vietnamese(subject)):
                return None  # Anaphora requires history; it is not an actor name.
            return subject
    return None


_EVENT_START = r"(?:chien tranh|cach mang|chien dich|chien thang|tran|hiep dinh|khoi nghia|phong trao)"
_TARGET_END = r"\s+(?:thanh cong|that bai|thang loi|suy yeu|sup do|ket thuc|co\s|da\s|dan den|de lai|ve\s|la gi|nhu the nao|vi sao|tai sao)|(?:\s+va|[,;])\s+(?:dieu do|no|he qua|hau qua|anh huong|tac dong|y nghia|nguyen nhan|ket qua|boi canh|dien bien|co y nghia)\b"
_FACET_CUES = {
    "cause": CAUSE_MARKERS,
    "context": ("boi canh", "dieu kien"), "objective": ("muc tieu", "muc dich"),
    "actors": ("luc luong", "chu the"), "method": ("tinh chat", "phuong phap", "dien bien"),
    "result": ("ket qua",), "consequence": ("he qua", "hau qua", "anh huong lau dai", "tac dong lau dai"),
    "significance": ("y nghia", "vai tro"),
    "military": ("quan su",), "political": ("chinh tri",), "economic": ("kinh te",),
    "domestic": ("xa hoi", "trong nuoc"), "international": ("ngoai giao", "quoc te"),
}


def raw_event_clause(question):
    compact = unicodedata.normalize("NFC", " ".join(question.split())).strip(" .?!")
    match = re.search(rf"\b{_EVENT_START}\b", _ascii_fold_vietnamese(compact))
    return compact[match.start():] if match else None


def extract_analytical_target(question: str) -> tuple[str | None, str | None, tuple[str, ...], str | None]:
    compact = unicodedata.normalize("NFC", " ".join(question.split())).strip(" .?!")
    folded = _ascii_fold_vietnamese(compact)
    outcome = next((value for pattern, value in _OUTCOMES if re.search(pattern, folded)), None)
    event = None
    match = next((candidate for candidate in re.finditer(rf"\b{_EVENT_START}\s+", folded)
                  if compact[candidate.start():candidate.end()].strip().casefold() != "trần"
                  and not re.search(r"(?:nha|trieu dai)\s+$", folded[:candidate.start()])), None)
    if match:
        end_match = re.search(_TARGET_END, folded[match.end():])
        end = match.end() + end_match.start() if end_match else len(compact)
        event = compact[match.start():end].strip(" ,;:")
        event = event[:1].upper() + event[1:]
        if _ascii_fold_vietnamese(event) in {"chien tranh viet nam", "cach mang thang tam"}:
            # Preserve conventional Vietnamese event capitalization.
            event = {"chien tranh viet nam": "Chiến tranh Việt Nam", "cach mang thang tam": "Cách mạng Tháng Tám"}[_ascii_fold_vietnamese(event)]
    dynasty = re.search(r"\b(?:nha|trieu dai(?:\s+nha)?)\s+([^\s,;.?!]+)", folded)
    subject = "Nhà " + compact[dynasty.start(1):dynasty.end(1)] if dynasty else None
    from app.agents.central_targets import entity_head
    captured = causal_subject_phrase(question)
    actors = []
    if captured and not entity_head(captured)[0]:
        for actor in re.split(r"\s+(?:và|cùng)\s+|\s*,\s*", captured, flags=re.I):
            # Bound free-form captures; do not turn explanatory clauses into entities.
            if 1 <= len(actor.split()) <= 7:
                actors.append(next((name for pattern, name in ACTOR_ALIASES if re.fullmatch(pattern, _ascii_fold_vietnamese(actor))), actor))
        if actors and not event and not subject:
            subject = " và ".join(actors)
    elif not captured:
        actors = [name for pattern, name in ACTOR_ALIASES if re.search(pattern, folded)]
    return event, subject, tuple(actors), outcome


def extract_requested_facets(question: str, kind: str | None) -> tuple[str, ...]:
    folded = _ascii_fold_vietnamese(question)
    if kind == "comparison":
        for target in extract_comparison_targets(question):
            folded = folded.replace(_ascii_fold_vietnamese(target), " ")
    explicit = tuple(key for key, cues in _FACET_CUES.items() if any(re.search(rf"\b{re.escape(cue)}\b", folded) for cue in cues))
    if explicit:
        return explicit
    if kind == "comparison":
        return ("context", "objective", "actors", "method", "result", "significance")
    return ("cause",) if kind == "cause" else ()


def plan_analytical_queries(analysis: CentralQuestionAnalysis, max_variants: int = 2) -> dict[str, list[str]]:
    """Bounded independent plans. Keys are canonical targets (or the original question)."""
    if analysis.relation_requested:
        related = analysis.related_entities[0]
        return {analysis.subject: [analysis.subject], related: [related],
                "relationship": [f"{analysis.subject} {related}"]}
    if analysis.comparison_targets:
        facet_words = {"context": "bối cảnh", "objective": "mục tiêu", "actors": "lực lượng",
                       "method": "tính chất", "result": "kết quả", "significance": "ý nghĩa",
                       "military": "quân sự", "political": "chính trị", "economic": "kinh tế",
                       "domestic": "xã hội", "international": "ngoại giao"}
        suffix = " ".join(facet_words[f] for f in analysis.facets if f in facet_words)
        from app.agents.central_targets import canonical_for
        return {target: [canonical_for(analysis, target), f"{canonical_for(analysis, target)} {suffix or 'bối cảnh kết quả ý nghĩa'}",
                         f"{target} lực lượng diễn biến kết quả"][:max_variants]
                for target in analysis.comparison_targets}
    target = analysis.event or analysis.subject
    from app.agents.central_facets import multi_facet, analytical_facets, FACET_QUERY
    if multi_facet(analysis) and target:
        return {f"facet:{facet}": list(dict.fromkeys([
            f"{target} {FACET_QUERY[facet]}" + (f" {analysis.outcome}" if facet == "cause" and analysis.outcome else "") + (" " + " ".join(analysis.actors) if analysis.actors else ""),
            f"{target} {FACET_QUERY[facet]} phân tích",
        ]))[:max_variants] for facet in analytical_facets(analysis)}
    if analysis.administrative_level:
        suffix = "nguyên nhân mục tiêu" if analysis.question_type == "cause" else "nội dung kết quả"
        return {target: [f"{target} {suffix}", f"{target} lý do chủ trương"][:max_variants]}
    if analysis.question_type == "cause" and target:
        causal = f"{target} nguyên nhân" + (f" {analysis.outcome}" if analysis.outcome else "")
        actors = " ".join(analysis.actors)
        followup = f"{target} bối cảnh quân sự chính trị {actors}".strip() if analysis.answer_depth == "broad_analysis" else analysis.question
        return {target: list(dict.fromkeys([f"{causal} {actors}".strip(), followup]))[:max_variants]}
    return {analysis.question: [analysis.question]}


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
        from app.agents.central_analytical import target_mentions
        missing_targets = [
            target for target in analysis.comparison_targets
            if not target_mentions(answer, target)
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
