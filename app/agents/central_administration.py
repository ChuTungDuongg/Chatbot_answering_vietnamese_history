"""Excerpt-local administrative granularity, dates and policy-cause grounding."""
from __future__ import annotations

import re
import unicodedata
from datetime import date

from app.agents.central_question import _ascii_fold_vietnamese


REFORM = re.compile(r"\b(?:sap nhap|sap xep|hop nhat|cai cach|tai co cau|to chuc lai|dieu chinh dia gioi)\b")
POLICY_CAUSE = re.compile(r"\b(?:nguyen nhan|ly do|muc tieu|muc dich|nham|do yeu cau|de tinh gon|chu truong|yeu cau.{0,30}(?:quan ly|bo may))\b")
LEVEL_NAMES = {"province": "cấp tỉnh", "district": "cấp huyện", "commune": "cấp xã", "unspecified": "hành chính"}


def administrative_levels(text):
    remainder = unicodedata.normalize("NFC", text).lower()
    levels = set()
    # Consume compound types before examining their constituent words.
    for pattern, level in (
        (r"\bthành phố trực thuộc trung ương\b|\bcấp tỉnh\b", "province"),
        (r"\bthành phố thuộc tỉnh\b|\bthị xã\b|\bcấp huyện\b", "district"),
        (r"\bthị trấn\b|\bcấp xã\b", "commune"),
    ):
        if re.search(pattern, remainder):
            levels.add(level)
            remainder = re.sub(pattern, " ", remainder)
    # Only unit nouns in the original accented text: 'hiệu quả', 'tình hình',
    # and 'xã hội' are not unit levels.
    remainder = re.sub(r"\bxã hội\b", "", remainder)
    for pattern, level in ((r"\btỉnh\b", "province"), (r"\b(?:huyện|quận)\b", "district"),
                           (r"\b(?:xã|phường)\b", "commune")):
        if re.search(pattern, remainder):
            levels.add(level)
    return sorted(levels)


def administrative_question(question, *, today=None):
    today = today or date.today()
    folded = _ascii_fold_vietnamese(question)
    levels = administrative_levels(question)
    if not REFORM.search(folded) or not (levels or "hanh chinh" in folded or "bo may" in folded):
        return {}
    level = levels[0] if len(levels) == 1 else "unspecified"
    years = tuple(sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", question)}))
    contemporary = bool(re.search(r"\b(?:hiện nay|gần đây|mới đây|đang|hiện tại|bây giờ)\b", question, re.I))
    recent = contemporary or any(today.year - 2 <= year <= today.year + 1 for year in years)
    country = "Việt Nam" if "viet nam" in folded else None
    event = "Sắp xếp đơn vị hành chính " + LEVEL_NAMES[level]
    event += (" " + country if country else "") + (" " + " ".join(map(str, years)) if years else "")
    return {"administrative_level": level, "time_scope": years, "freshness_required": recent,
            "freshness_reason": "recent_policy_or_administrative_event" if recent else None,
            "event": event, "subject": country,
            "premise_requires_validation": bool(re.search(r"\b(?:bat dau|lan dau|chi den)\b", folded))}


def annotate_administration(row, analysis):
    text = str(row.get("text") or row.get("content") or row.get("snippet") or "")
    if row.get("retrieval_tool") == "search_wikipedia":
        # A title can route a non-citable search hit to fetch; only the fetched
        # excerpt may establish administrative level, time and causes.
        text = str(row.get("title") or "") + ". " + text
    levels = administrative_levels(text)
    requested = analysis.administrative_level
    sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)
    relevant_indices = [i for i, s in enumerate(sentences) if (requested == "unspecified" or requested in administrative_levels(s))
                        and REFORM.search(_ascii_fold_vietnamese(s))]
    relevant = [sentences[i] for i in relevant_indices]
    matched = bool(relevant)
    causal = []
    for index in relevant_indices:
        span = sentences[index]
        # Bind an immediately following objective to its explicit unit/phase;
        # never borrow objectives from a different level elsewhere on the page.
        if index + 1 < len(sentences) and not administrative_levels(sentences[index + 1]):
            following = sentences[index + 1]
            if not re.search(r"\b(?:19|20)\d{2}\b", following) and POLICY_CAUSE.search(_ascii_fold_vietnamese(following)):
                span += " " + following
        if POLICY_CAUSE.search(_ascii_fold_vietnamese(span)) and (not analysis.time_scope or any(
            re.search(rf"\b{year}\b", span) for year in analysis.time_scope)):
            causal.append(span)
    # A date range such as 2023–2030 is not confirmation of a 2025 phase.
    explicit_time = not analysis.time_scope or any(
        re.search(rf"\b{year}\b", s) for s in relevant for year in analysis.time_scope)
    metadata = row.get("metadata") or {}
    dated = [str(row.get(key) or metadata.get(key) or "") for key in ("published_at", "updated_at", "last_modified", "revision_timestamp")]
    threshold = max(analysis.time_scope) if analysis.time_scope else date.today().year
    dated_current = any(re.match(r"\d{4}", value) and int(value[:4]) >= threshold for value in dated)
    live = row.get("retrieval_tool") in {"fetch_wikipedia_page", "fetch_web_page"}
    fresh = not analysis.freshness_required or (explicit_time and matched and (dated_current or (live and bool(analysis.time_scope))))
    return {**row, "evidence_administrative_levels": levels, "administrative_level_consistent": matched,
            "target_consistent": matched, "overview_anchor": matched and bool(causal),
            "administrative_time_consistent": explicit_time, "freshness_sufficient": fresh,
            "policy_cause_spans": causal, "causal_relevance": bool(causal),
            "cause_facet_score": 2 if causal else 0,
            "evidence_role": "core" if matched and explicit_time else "background"}


def administrative_answer_issues(answer, analysis, packet):
    if not analysis.administrative_level:
        return []
    beginnings = re.compile(r"\b(?:bat dau|lan dau|khoi dau)\b")
    def asserts_start(text):
        folded = _ascii_fold_vietnamese(text)
        return beginnings.search(folded) and not re.search(r"\b(?:khong|chua|chang)\b.{0,65}\b(?:bat dau|lan dau|khoi dau)\b", folded)
    claims = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if asserts_start(s)
              and REFORM.search(_ascii_fold_vietnamese(s))]
    if not claims:
        return []
    spans = [s for source in packet for s in re.split(r"(?<=[.!?])\s+|\n+", source.text)
             if asserts_start(s) and REFORM.search(_ascii_fold_vietnamese(s))]
    for claim in claims:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", claim))
        levels = set(administrative_levels(claim) or [analysis.administrative_level])
        if not any(years <= set(re.findall(r"\b(?:19|20)\d{2}\b", span))
                   and levels <= set(administrative_levels(span)) for span in spans):
            return ["unsupported_administrative_premise"]
    return []


def administrative_coverage(selected, candidates, analysis, min_chars):
    rows = [annotate_administration(row, analysis) for row in selected]
    core = [r for r in rows if r["administrative_level_consistent"] and r["administrative_time_consistent"]
            and r["freshness_sufficient"] and r.get("citable", True) and len(r.get("text", "")) >= min_chars]
    causes = [r for r in core if r["policy_cause_spans"]]
    sufficient = bool(causes if analysis.question_type == "cause" else core)
    observed = [annotate_administration(row, analysis) for row in candidates]
    return sufficient, {
        "requested_administrative_level": analysis.administrative_level,
        "evidence_administrative_levels": sorted({level for r in observed for level in r["evidence_administrative_levels"]}),
        "administrative_level_match_count": sum(r["administrative_level_consistent"] for r in observed),
        "administrative_level_mismatch_count": sum(not r["administrative_level_consistent"] for r in observed),
        "cause_evidence_count": len(causes), "fresh_core_evidence_count": len(core),
        "premise_validation_status": "requires_evidence_qualified_wording" if analysis.premise_requires_validation and sufficient else "not_established" if analysis.premise_requires_validation else "not_requested",
        "evidence_sufficiency_reason": "administrative_level_time_and_cause_covered" if sufficient else "administrative_level_time_or_cause_insufficient",
    }


def administrative_contract(analysis):
    if not analysis.administrative_level:
        return ""
    return (f"\nPhạm vi cần chứng minh: {analysis.event}. Chỉ dùng đoạn đúng cấp hành chính và thời điểm "
            "để giải thích nguyên nhân; danh sách đơn vị nhập vào nhau không chứng minh lý do. "
            "Không mặc nhiên chấp nhận tiền đề 'bắt đầu/lần đầu': chỉ khẳng định mốc khởi đầu khi nguồn nêu rõ. "
            "Nếu nguồn phân biệt chương trình chung trước đó và đợt ở cấp cụ thể, giải thích sự phân biệt đó; "
            "nếu chưa xác lập thời điểm bắt đầu, nói rõ nguồn chỉ xác lập đợt được hỏi. Không tự suy diễn tiền đề.")
