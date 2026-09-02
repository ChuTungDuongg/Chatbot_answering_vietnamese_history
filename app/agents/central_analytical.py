"""Deterministic Central-only entity, facet and coverage policy; no model dependencies."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlsplit

from app.agents.central_question import CentralQuestionAnalysis, _ascii_fold_vietnamese
from app.agents.central_viewpoints import annotate_viewpoints
from app.agents.central_relationships import annotate_relationship, relationship_coverage
from app.agents.central_targets import canonical_for, entity_type_consistent
from app.agents.central_administration import annotate_administration, administrative_coverage


def normalize_entity(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _ascii_fold_vietnamese(value)))


def target_key(value: str) -> str:
    person_tran = value.strip().casefold().startswith("trần ")
    value = normalize_entity(value)
    prefix = r"^(?:chien thang|chien dich)\s+" if person_tran else r"^(?:chien thang|chien dich|tran)\s+"
    value = re.sub(prefix, "", value)
    return re.sub(r"\bnam\s+(?=\d{3,4}\b)", "", value).strip()


def target_mentions(text: str, target: str) -> bool:
    key = target_key(target)
    return bool(key and f" {key} " in f" {target_key(text)} ")


DIMENSION_CUES = {
    "context": ("boi canh", "khung hoang", "truoc tinh hinh", "dieu kien", "chien tranh the gioi"),
    "objective": ("muc tieu", "muc dich", "nham", "gianh chinh quyen", "doc lap"),
    "actors": ("luc luong", "quan doi", "nhan dan", "nong dan", "chinh phu"),
    "method": ("khoi nghia", "chien dich", "dau tranh", "tan cong", "chien luoc", "phuong phap"),
    "result": ("ket qua", "thang loi", "that bai", "thanh cong", "dau hang", "sup do", "suy yeu", "rut quan"),
    "significance": ("y nghia", "tac dong", "anh huong", "mo ra", "cham dut", "buoc ngoat"),
    "military": ("quan su", "quan doi", "chien truong", "chien dich", "tac chien"),
    "strategy": ("chien luoc", "muc tieu chien tranh", "sai lam", "danh gia sai"),
    "political": ("chinh tri", "chinh quyen", "chinh phu", "trieu dinh", "quan lai", "tham nhung"),
    "economic": ("kinh te", "hau can", "vien tro", "thue", "ruong dat", "tiep te", "tai chinh"),
    "domestic": ("xa hoi", "phan chien", "long dan", "nong dan", "bat binh", "du luan"),
    "international": ("quoc te", "ngoai giao", "hiep dinh", "dong minh", "dam phan"),
    "opponent": ("doi phuong", "suc chien dau", "huy dong", "quan giai phong"),
}
CAUSE_DIMENSIONS = {"military", "strategy", "political", "economic", "domestic", "international", "opponent"}
_CAUSAL = ("nguyen nhan", "vi", "do", "dan den", "khien", "suy yeu", "khung hoang", "that bai", "thanh cong")
_CAUSE_LINK = re.compile(r"\b(?:nguyen nhan|nho|boi vi|gop phan|dan den|khien|tao dieu kien|lien quan den|tac dong den|lam suy yeu|lam giam)\b|\b(?:thanh cong|thang loi|that bai|suy yeu|sup do)\b.{0,45}\b(?:do|vi|boi)\b")
_CAUSE_CONDITIONS = re.compile(r"\b(?:chuan bi|to chuc|lanh dao|thoi co|co hoi|huy dong|phoi hop|ung ho|doan ket|khoang trong quyen luc|tuong quan luc luong|boi canh|dieu kien|khung hoang|ap luc|tham nhung|thieu hut)\b")
_CAUSE_DISTRACTIONS = re.compile(r"\b(?:bai tho|tho ca|tien doan|du doan|loi sam|le ky niem|tuong niem|hau chien|hau qua|giai doan sau)\b|\bsau (?:khi )?(?:gianh chinh quyen|thang loi|thanh cong|cuoc cach mang|cuoc khoi nghia)\b")
_PLACE_PREFIX = re.compile(r"^(?:duong|quang truong|tuong dai|khu di tich|bao tang|le ky niem|du lich|thanh pho)\b")
_ARTIFACT = re.compile(r"\b(?:bai hat|ca khuc|nhac pham|tac pham am nhac|duong pho|nha ga|ga tau|quang truong|tuong dai|quan thuoc|phuong thuoc|le ky niem)\b")


def viewpoint_sensitive(text: str) -> bool:
    return bool(annotate_viewpoints(text))


def cause_sentence_relevance(text: str) -> int:
    """Small ranking preference, not a new sufficiency threshold or fact table."""
    folded = normalize_entity(text)
    if _CAUSE_DISTRACTIONS.search(folded):
        return -1
    if _CAUSE_LINK.search(folded):
        return 2
    return 1 if _CAUSE_CONDITIONS.search(folded) else 0


def entity_consistency(row: dict[str, Any], target: str, *, person: bool = False) -> tuple[bool, bool, str | None]:
    """Return associated, canonical overview, rejection reason. Metadata is only supporting evidence."""
    raw_title = str(row.get("title") or row.get("page_title") or row.get("source_title") or "")
    if not raw_title and row.get("url"):
        raw_title = unquote(urlsplit(str(row["url"])).path.rsplit("/", 1)[-1]).replace("_", " ")
    title = normalize_entity(raw_title)
    base = normalize_entity(re.sub(r"\([^)]*\)", "", raw_title))
    text = str(row.get("text") or row.get("content") or row.get("snippet") or "")
    key = normalize_entity(target) if person else target_key(target)
    title_key = base if person else target_key(re.sub(r"\([^)]*\)", "", raw_title))
    canonical = title_key == key
    if person:
        associated = canonical or any(f" {key} " in f" {normalize_entity(value)} " for value in (text, raw_title))
    else:
        associated = canonical or target_mentions(text, target) or target_mentions(raw_title, target)
    if not person:
        metadata = row.get("metadata") or {}
        # An event mention in a song/station/commemorative page does not establish
        # causal evidence. Explicit questions about the artifact keep that page.
        if not canonical and not _ARTIFACT.search(normalize_entity(target)):
            lead = normalize_entity(text[:350])
            place_title = re.match(r"^(?:Quận|Huyện|Phường|Ga|Trạm)\s", raw_title, re.I)
            explicit_place_target = re.match(r"^(?:Quận|Huyện|Phường|Ga|Trạm)\s", target, re.I)
            artifact_kind = str(metadata.get("page_type") or metadata.get("entity_type") or "").casefold() in {
                "song", "music composition", "street", "square", "monument", "station", "district", "commemoration",
            }
            if _ARTIFACT.search(title) or re.search(r"\bla (?:mot |ten mot )?" + _ARTIFACT.pattern, lead) or (place_title and not explicit_place_target) or artifact_kind:
                return False, False, "event_incidental_artifact"
        scope = normalize_entity(str(metadata.get("country") or metadata.get("domain_scope") or ""))
        if ("trung quoc" in title or scope in {"trung quoc", "china"}) and "trung quoc" not in normalize_entity(target):
            return False, False, "foreign_entity_scope"
        if _PLACE_PREFIX.match(title) and not _PLACE_PREFIX.match(normalize_entity(target)):
            return False, False, "event_place_or_commemoration_collision"
        if re.search(r"\((?:thành phố|city|địa danh)\)", raw_title, re.I):
            return False, False, "event_city_collision"
        # A calendar/month page is not the revolution, even if its body mentions it.
        if re.match(r"^(?:cach mang|chien tranh|khoi nghia)\b", key) and title and not associated:
            return False, False, "event_entity_mismatch"
        if key.startswith("cach mang ") and title == key.removeprefix("cach mang "):
            return False, False, "event_calendar_collision"
    return associated, canonical, None


def annotate_evidence(row: dict[str, Any], analysis: CentralQuestionAnalysis, target: str | None = None) -> dict[str, Any]:
    row = dict(row)
    text = str(row.get("text") or row.get("content") or row.get("snippet") or "")
    folded = f" {normalize_entity(text)} "
    target = target or row.get("comparison_target") or analysis.event or analysis.subject
    associated, overview, reason = entity_consistency(row, target, person=analysis.question_type == "biography") if target else (True, False, None)
    type_match = entity_type_consistent(row, target) if target and analysis.comparison_targets else True
    canonical_match = True
    if target in analysis.comparison_targets:
        canonical = canonical_for(analysis, target)
        canonical_match = entity_consistency(row, canonical)[0]
        overview = overview or entity_consistency(row, canonical)[1]
        associated = associated and canonical_match and type_match
    cause_score, cause_downranked = 0, False
    dimension_text = folded
    if analysis.question_type == "cause":
        sentences = [s for s in re.split(r"[.!?;\n]", text) if s.strip()]
        relevance = [(s, cause_sentence_relevance(s)) for s in sentences]
        # Do not turn a poem/aftermath's vocabulary into causal coverage. Other
        # requested facets still retain those sentences as ordinary evidence.
        if not set(analysis.facets) & {"result", "significance", "consequence"}:
            dimension_text = " " + normalize_entity(" ".join(s for s, score in relevance if score >= 0)) + " "
            cause_downranked = sum(score < 0 for _, score in relevance) > sum(score > 0 for _, score in relevance)
        cause_score = max((score for _, score in relevance), default=0)
    dimensions = [dim for dim, cues in DIMENSION_CUES.items() if any(f" {cue} " in dimension_text for cue in cues)]
    chronology = False
    if analysis.question_type == "cause" and analysis.outcome in {"suy yếu", "sụp đổ"} and analysis.subject:
        dynasty = normalize_entity(analysis.subject).removeprefix("nha ").removeprefix("trieu dai ")
        # Infer aftermath from text, not an external chronology table. Compare sentences so
        # an incidental reference to collapse does not make an aftermath chunk causal.
        sentences = [normalize_entity(s) for s in re.split(r"[.!?;]", text) if s.strip()]
        after = sum(any(cue in s for cue in (f"hau {dynasty}", "sau khi", "sau su sup do", "phuc hung")) for s in sentences)
        during = sum(any(cue in s for cue in ("cuoi trieu", "cuoi thoi", "nguyen nhan", "ruong dat", "thue", "tham nhung")) for s in sentences)
        chronology = after > during
    viewpoints = annotate_viewpoints(text)
    row.update({
        "target_consistent": associated and not reason, "overview_anchor": overview and not reason,
        "entity_type_consistent": type_match, "canonical_target_consistent": canonical_match,
        "entity_filter_reason": reason, "evidence_dimensions": dimensions,
        "causal_relevance": cause_score > 0 or any(f" {cue} " in dimension_text for cue in _CAUSAL),
        "cause_facet_score": cause_score, "cause_focus_downranked": cause_downranked,
        "chronology_downranked": chronology, "viewpoint_sensitive": bool(viewpoints),
        "viewpoint_annotations": viewpoints,
    })
    if analysis.administrative_level:
        return annotate_administration(row, analysis)
    return annotate_relationship(row, analysis) if analysis.relation_requested else row


def evidence_targets(row: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys([*(row.get("comparison_targets") or []), *([row["comparison_target"]] if row.get("comparison_target") else [])]))


def coverage_select(rows: list[dict[str, Any]], analysis: CentralQuestionAnalysis, limit: int) -> list[dict[str, Any]]:
    """Greedy new-dimension gain with an overview reservation; ranks are batch-local."""
    pending = list(rows)
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    requested = set(analysis.facets)
    while pending and len(selected) < limit:
        def priority(row):
            dims = set(row.get("evidence_dimensions") or [])
            return (
                bool(row.get("target_consistent")) and row.get("entity_type_consistent", True),
                row.get("administrative_time_consistent", True) and row.get("freshness_sufficient", True),
                not row.get("chronology_downranked", False),
                not row.get("cause_focus_downranked", False) if analysis.question_type == "cause" else True,
                bool(row.get("overview_anchor")) and not any(s.get("overview_anchor") for s in selected),
                bool(row.get("target_consistent")),
                row.get("cause_facet_score", 0) > 0 if analysis.question_type == "cause" else False,
                len((dims - covered) & requested) * 2 + len((dims - covered) & CAUSE_DIMENSIONS) + len(dims - covered),
                bool(row.get("causal_relevance")) if analysis.question_type == "cause" else False,
                -int(row.get("retrieval_rank", 999)),
            )
        best = max(pending, key=priority)
        pending.remove(best)
        selected.append(best)
        covered.update(best.get("evidence_dimensions") or [])
    return selected


def strong_evidence(row: dict[str, Any], *, min_chars: int) -> bool:
    text = str(row.get("text") or "").strip()
    return bool(row.get("target_consistent") and row.get("citable", True)
                and row.get("entity_type_consistent", True) and row.get("canonical_target_consistent", True)
                and len(text) >= min_chars and len(text.split()) >= 12
                and len(row.get("evidence_dimensions") or []) >= 2
                and not row.get("chronology_downranked") and not row.get("cause_focus_downranked"))


def coverage_report(selected: list[dict[str, Any]], candidates: list[dict[str, Any]], analysis: CentralQuestionAnalysis, config) -> tuple[bool, dict[str, Any]]:
    dims = sorted({dim for row in selected for dim in row.get("evidence_dimensions", [])})
    balance = {}
    for target in analysis.comparison_targets:
        group = [annotate_evidence(row, analysis, target) for row in selected if target in evidence_targets(row)]
        strong = list({normalize_entity(str(row.get("text") or "")): row for row in group
                       if strong_evidence(row, min_chars=config.strong_evidence_min_chars)}.values())
        group_dims = {dim for row in strong for dim in row.get("evidence_dimensions", [])}
        balance[target] = {
            "candidate_count": sum(target in evidence_targets(row) for row in candidates),
            "selected_count": len(group), "strong_evidence_count": len(strong),
            "dimensions_covered": sorted(group_dims),
            "adequate": len(strong) >= config.comparison_min_strong_sources and len(group_dims) >= 2,
        }
    administrative = {}
    if analysis.administrative_level:
        sufficient, administrative = administrative_coverage(selected, candidates, analysis, config.strong_evidence_min_chars)
        reason = administrative["evidence_sufficiency_reason"]
    elif analysis.relation_requested:
        sufficient, relationship = relationship_coverage(selected, analysis)
        reason = relationship["evidence_sufficiency_reason"]
    elif analysis.question_type == "comparison":
        sufficient = len(balance) >= 2 and all(item["adequate"] for item in balance.values())
        reason = "both_comparison_targets_covered" if sufficient else "comparison_target_coverage_insufficient"
    elif analysis.question_type == "cause" and (analysis.event or analysis.subject):
        strong = [row for row in selected if strong_evidence(row, min_chars=config.strong_evidence_min_chars)]
        causal = any(row.get("causal_relevance") for row in strong)
        breadth = {dim for row in strong for dim in row.get("evidence_dimensions", [])} & CAUSE_DIMENSIONS
        sufficient = bool(strong and causal and len(breadth) >= 2)
        reason = "causal_target_and_dimension_coverage" if sufficient else "causal_coverage_insufficient"
    elif analysis.question_type == "biography" and analysis.subject:
        sufficient = any(row.get("target_consistent") and row.get("text") for row in selected)
        reason = "biography_subject_evidence" if sufficient else "biography_subject_evidence_missing"
    else:
        sufficient = any(row.get("text") and row.get("citable", True) for row in selected)
        reason = "usable_grounded_evidence" if sufficient else "no_usable_evidence"
    return sufficient, {
        "comparison_balance": balance,
        "comparison_target_evidence_counts": {target: item["selected_count"] for target, item in balance.items()},
        "overview_anchor_selected": [row.get("chunk_id") for row in selected if row.get("overview_anchor")],
        "evidence_dimensions_covered": dims, "evidence_sufficiency_reason": reason,
        "evidence_sufficient": sufficient,
        "viewpoint_sensitive_evidence_count": sum(bool(row.get("viewpoint_sensitive")) for row in selected),
        **(relationship if analysis.relation_requested else {}),
        **administrative,
    }
