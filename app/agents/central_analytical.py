"""Deterministic Central-only entity, facet and coverage policy; no model dependencies."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote, urlsplit

from app.agents.central_question import CentralQuestionAnalysis, _ascii_fold_vietnamese


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
_PLACE_PREFIX = re.compile(r"^(?:duong|quang truong|tuong dai|khu di tich|bao tang|le ky niem|du lich|thanh pho)\b")
_VIEWPOINT = re.compile(r'"[^"\n]{8,}"|“[^”\n]{8,}”|\b(?:lu tay sai|chung ta|chung toi|ta nhat dinh|bon de quoc|phan dong|nguy quan|nguy quyen)\b', re.I)


def viewpoint_sensitive(text: str) -> bool:
    return bool(_VIEWPOINT.search(_ascii_fold_vietnamese(text)))


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
    dimensions = [dim for dim, cues in DIMENSION_CUES.items() if any(f" {cue} " in folded for cue in cues)]
    chronology = False
    if analysis.question_type == "cause" and analysis.outcome in {"suy yếu", "sụp đổ"} and analysis.subject:
        dynasty = normalize_entity(analysis.subject).removeprefix("nha ").removeprefix("trieu dai ")
        # Infer aftermath from text, not an external chronology table. Compare sentences so
        # an incidental reference to collapse does not make an aftermath chunk causal.
        sentences = [normalize_entity(s) for s in re.split(r"[.!?;]", text) if s.strip()]
        after = sum(any(cue in s for cue in (f"hau {dynasty}", "sau khi", "sau su sup do", "phuc hung")) for s in sentences)
        during = sum(any(cue in s for cue in ("cuoi trieu", "cuoi thoi", "nguyen nhan", "ruong dat", "thue", "tham nhung")) for s in sentences)
        chronology = after > during
    row.update({
        "target_consistent": associated and not reason, "overview_anchor": overview and not reason,
        "entity_filter_reason": reason, "evidence_dimensions": dimensions,
        "causal_relevance": any(f" {cue} " in folded for cue in _CAUSAL),
        "chronology_downranked": chronology, "viewpoint_sensitive": viewpoint_sensitive(text),
    })
    return row


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
                not row.get("chronology_downranked", False),
                bool(row.get("overview_anchor")) and not any(s.get("overview_anchor") for s in selected),
                bool(row.get("target_consistent")),
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
                and len(text) >= min_chars and len(text.split()) >= 12
                and len(row.get("evidence_dimensions") or []) >= 2
                and not row.get("chronology_downranked"))


def coverage_report(selected: list[dict[str, Any]], candidates: list[dict[str, Any]], analysis: CentralQuestionAnalysis, config) -> tuple[bool, dict[str, Any]]:
    dims = sorted({dim for row in selected for dim in row.get("evidence_dimensions", [])})
    balance = {}
    for target in analysis.comparison_targets:
        group = [row for row in selected if target in evidence_targets(row)]
        strong = list({normalize_entity(str(row.get("text") or "")): row for row in group
                       if strong_evidence(row, min_chars=config.strong_evidence_min_chars)}.values())
        group_dims = {dim for row in strong for dim in row.get("evidence_dimensions", [])}
        balance[target] = {
            "candidate_count": sum(target in evidence_targets(row) for row in candidates),
            "selected_count": len(group), "strong_evidence_count": len(strong),
            "dimensions_covered": sorted(group_dims),
            "adequate": len(strong) >= config.comparison_min_strong_sources and len(group_dims) >= 2,
        }
    if analysis.question_type == "comparison":
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
    }
