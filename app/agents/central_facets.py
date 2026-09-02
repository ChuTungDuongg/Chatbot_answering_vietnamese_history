"""Deterministic analytical facet coverage and partial-answer contracts."""
from __future__ import annotations

import re

from app.agents.central_question import _ascii_fold_vietnamese
from app.agents.central_viewpoints import annotate_viewpoints


FACET_LABELS = {"cause": "Nguyên nhân", "consequence": "Hệ quả lâu dài", "significance": "Ý nghĩa",
                "context": "Bối cảnh", "result": "Kết quả", "method": "Diễn biến / phương pháp"}
FACET_QUERY = {"cause": "nguyên nhân", "consequence": "hệ quả lâu dài", "significance": "ý nghĩa lịch sử",
               "context": "bối cảnh", "result": "kết quả", "method": "diễn biến"}
FACET_SIGNALS = {
    "consequence": r"\b(?:he qua|hau qua|lau dai|di chung|de lai|sau chien tranh|hau chien|ve sau|nhieu thap ky|keo dai nhieu nam)\b",
    "significance": r"\b(?:y nghia|vai tro|buoc ngoat|danh dau|mo ra)\b",
    "context": r"\b(?:boi canh|truoc khi|truoc tinh hinh|dieu kien|khung hoang)\b",
    "result": r"\b(?:ket qua|ket thuc|thang loi|that bai|dau hang|rut quan|gianh duoc)\b",
    "method": r"\b(?:dien bien|phuong phap|chien luoc|tien cong|tan cong|giai doan|dau tranh)\b",
}


def analytical_facets(analysis):
    if analysis.question_type == "comparison" or analysis.relation_requested or analysis.administrative_level:
        return ()
    return tuple(facet for facet in analysis.facets if facet in FACET_LABELS)


def multi_facet(analysis):
    return bool(analysis.event or analysis.subject) and len(analytical_facets(analysis)) > 1


def evidence_facets(row):
    text = _ascii_fold_vietnamese(str(row.get("text") or row.get("content") or row.get("snippet") or ""))
    facets = [facet for facet, pattern in FACET_SIGNALS.items() if re.search(pattern, text)]
    if row.get("cause_facet_score", 0) > 0 and not row.get("cause_focus_downranked"):
        facets.insert(0, "cause")
    return list(dict.fromkeys(facets))


def neutral_preference(analysis):
    return analysis.question_type in {"cause", "comparison", "consequence", "significance", "evaluation"} and not re.search(
        r"\b(?:quan diem|nhan dinh cua|theo nhan dinh|trich dan|loi noi|phat bieu cua)\b", _ascii_fold_vietnamese(analysis.question))


def viewpoint_cost(text):
    spans = sorted((a["start"], a["end"]) for a in annotate_viewpoints(text) if a["requires_attribution"])
    end, total = 0, 0
    for start, stop in spans:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total / max(1, len(text))


def facet_strong(row, facet, min_chars):
    return (row.get("target_consistent") and row.get("citable", True)
            and row.get("entity_type_consistent", True) and row.get("canonical_target_consistent", True)
            and len(str(row.get("text") or "")) >= min_chars and facet in row.get("evidence_facets", [])
            and (facet != "cause" or not row.get("chronology_downranked")))


def facet_coverage(selected, candidates, analysis, min_chars):
    requested = analytical_facets(analysis)
    balance = {}
    for facet in requested:
        group = [row for row in selected if facet in row.get("evidence_facets", [])]
        strong = {str(row.get("text") or ""): row for row in group if facet_strong(row, facet, min_chars)}
        balance[facet] = {"candidate_count": sum(facet in row.get("evidence_facets", []) for row in candidates),
                          "selected_count": len(group), "strong_evidence_count": len(strong), "adequate": bool(strong)}
    covered = [facet for facet in requested if balance[facet]["adequate"]]
    unresolved = [facet for facet in requested if facet not in covered]
    return {"requested_facets": list(requested), "covered_facets": covered, "unresolved_facets": unresolved,
            "facet_balance": balance, "partial_answer": bool(covered and unresolved)}


def facet_limitation(facet):
    return f"Bằng chứng hiện có chưa đủ để kết luận chắc chắn về phần {FACET_LABELS[facet].lower()}."


def facet_contract(debug):
    if len(debug.get("requested_facets", [])) < 2:
        return ""
    covered = debug.get("covered_facets", [])
    missing = debug.get("unresolved_facets", [])
    return ("\nTrả lời riêng từng phần được hỏi bằng các đề mục: "
            + ", ".join(FACET_LABELS[f] for f in covered)
            + ". Chỉ dùng bằng chứng có evidence_facets phù hợp, không nhầm kết quả trực tiếp với hệ quả lâu dài. "
            + " ".join(f"Phần {FACET_LABELS[f]} chưa được xác lập: chỉ ghi '{facet_limitation(f)}' Không bổ sung suy đoán." for f in missing))


def facet_answer_issues(answer, debug):
    if len(debug.get("requested_facets", [])) < 2:
        return []
    folded = _ascii_fold_vietnamese(answer)
    issues = []
    for facet in debug.get("covered_facets", []):
        if not any(re.search(rf"\b{re.escape(_ascii_fold_vietnamese(label))}\b", folded)
                   for label in (FACET_LABELS[facet], FACET_QUERY[facet])):
            issues.append(f"{facet}_section_missing")
    for facet in debug.get("unresolved_facets", []):
        # A missing facet must be an explicit limitation, not memory-based filler.
        lines = [line for line in re.split(r"\n\s*\n", answer) if re.search(
            FACET_SIGNALS.get(facet, r"\bnguyen nhan\b"), _ascii_fold_vietnamese(line))]
        if not lines or any(not re.search(r"\b(?:chua du|khong du|chua co|chua xac lap|chua the)\b", _ascii_fold_vietnamese(line))
                            for line in lines if not re.fullmatch(r"\s*#{1,6}[^\n]+", line)):
            issues.append(f"unsupported_{facet}_section")
    return issues
