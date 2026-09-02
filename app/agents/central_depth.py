"""Question depth and excerpt-backed causal plans. No model or historical fact table."""
from __future__ import annotations

import re

from app.agents.central_question import _ascii_fold_vietnamese, normalized_actor_text


def answer_depth(kind, *, event=None, subject=None, facets=(), administrative=False):
    if kind == "comparison":
        return "comparison"
    if kind == "biography":
        return "biography_summary"
    if kind not in {"cause", "consequence", "significance", "evaluation"}:
        return "simple_fact"
    # A bounded policy/date or an explicitly isolated facet is a focused explanation.
    narrowed = set(facets) & {"military", "political", "economic", "domestic", "international"}
    if administrative or narrowed:
        return "focused_explanation"
    return "broad_analysis" if event or subject else "focused_explanation"


# A dimension needs a proposition, not an isolated heading or a list of topics.
# In particular, "chiến lược quân sự / mục tiêu chính trị" is not three causes.
DIMENSION_SIGNALS = {
    "military": r"quan doi|chien truong|tac chien|chien dich|ton that|luc luong quan su|suc manh quan su|dau hang",
    "strategy": r"chien luoc|muc tieu chien tranh|danh gia sai",
    "political": r"chinh quyen|chinh phu|trieu dinh|quan lai|tham nhung|the che|to chuc chinh tri|bat on chinh tri|khung hoang chinh tri|khoang trong quyen luc|lanh dao",
    "domestic": r"phan chien|du luan|quoc hoi|ap luc trong nuoc|ung ho trong nuoc",
    "international": r"ngoai giao|dam phan|hiep dinh|quoc te|dong minh",
    "economic": r"kinh te|hau can|vien tro|tiep te|tai chinh|chi phi|ngan sach|thue|ruong dat",
    "opponent": r"doi phuong|quan giai phong|suc chien dau",
    "social": r"long dan|nong dan|ung ho cua nhan dan|ung ho cua nguoi dan|co so trong nhan dan|huy dong nhan dan|nhan dan tham gia|luc luong nhan dan",
}
DIMENSION_LABELS = {
    "military": "quân sự", "strategy": "chiến lược", "political": "chính trị / thể chế",
    "domestic": "ràng buộc trong nước", "international": "quốc tế / ngoại giao",
    "economic": "kinh tế / hậu cần", "opponent": "năng lực đối phương", "social": "cơ sở xã hội",
}
_MECHANISM = re.compile(
    r"\b(?:vi|do|nho|khien|dan den|gop phan|lam|gay|han che|suy yeu|khung hoang|"
    r"khong dat|khong the|khong du|thieu|kho khan|ap luc|giam|tang|huy dong|"
    r"duy tri|to chuc|phoi hop|kiem soat|ngan can|buoc|bat binh|tap trung|nang|tao dieu kien)\b")
_DISTRACTION = re.compile(r"\b(?:bai tho|tho ca|le ky niem|tuong niem|hau qua|hau chien|sau chien tranh|danh muc|tu khoa)\b")
_CITATION = re.compile(r"\[[^\]\n]+\]")


def dimension_spans(text):
    """Conservative surface evidence, not semantic entailment or whole-page labels."""
    spans = {}
    # Keep conjuncts together when they share a causal predicate, but not sentences.
    for clause in re.split(r"[.!?;\n]+", _CITATION.sub("", text)):
        clause = clause.strip(" *#:-")
        folded = _ascii_fold_vietnamese(clause)
        if len(clause.split()) < 8 or not _MECHANISM.search(folded) or _DISTRACTION.search(folded):
            continue
        if re.search(r"\b(?:chua du bang chung|bang chung hien co chua|chua xac lap)\b", folded):
            continue
        for dimension, pattern in DIMENSION_SIGNALS.items():
            if re.search(rf"\b(?:{pattern})\b", folded):
                spans.setdefault(dimension, []).append(clause)
    return spans


def actor_scope(text, actors):
    folded = f" {normalized_actor_text(text)} "
    return [actor for actor in actors if f" {normalized_actor_text(actor)} " in folded]


def evidence_plan(packet, analysis, config):
    from app.agents.central_analytical import annotate_evidence, strong_evidence
    sources = {}
    for item in packet:
        row = annotate_evidence({"title": item.title, "text": item.text}, analysis)
        if not strong_evidence(row, min_chars=config.strong_evidence_min_chars):
            continue
        for dimension in dimension_spans(item.text):
            sources.setdefault(dimension, []).append(item.alias)
    covered_actors = {actor for item in packet for actor in actor_scope(item.text, analysis.actors)}
    return {"answer_depth": analysis.answer_depth,
            "unresolved_actor_scopes": [actor for actor in analysis.actors if actor not in covered_actors],
            "strong_evidence_dimensions": sorted(sources),
            "analytical_dimension_sources": sources}


def depth_contract(analysis, plan):
    if analysis.answer_depth == "simple_fact":
        return "\nEXPECTED DEPTH: simple_fact\nTrả lời ngắn gọn đúng dữ kiện được hỏi, thường một hoặc hai câu có trích dẫn."
    if analysis.answer_depth != "broad_analysis":
        return f"\nEXPECTED DEPTH: {analysis.answer_depth}\nGiải thích ở mức bằng chứng hỗ trợ và đúng phạm vi được hỏi."
    dimensions = plan["strong_evidence_dimensions"]
    supported = "\n".join(f"- {dim} ({DIMENSION_LABELS[dim]}): " + " ".join(f"[{a}]" for a in plan["analytical_dimension_sources"][dim]) for dim in dimensions)
    return ("\nEXPECTED DEPTH: broad_analysis\nSUPPORTED ANALYTICAL DIMENSIONS:\n" + (supported or "Chưa xác lập cơ chế cụ thể.")
            + "\nGiải thích kết quả qua các nguyên nhân KHÁC NHAU được nguồn hỗ trợ: nguyên nhân → vì sao quan trọng → hệ quả đối với kết quả được hỏi. "
            "Không coi những cách diễn đạt lại cùng một ý là các nguyên nhân riêng. Dùng ngôn ngữ lịch sử trung lập, tránh khẳng định tuyệt đối và lời văn thiên kiến; trích dẫn mỗi đoạn có dữ kiện. "
            "Có thể mở đầu bằng luận điểm ngắn, phát triển các đoạn hoặc gạch đầu dòng và kết lại ngắn gọn; không ép mẫu sáu mục. "
            + ("Khi có nhiều phương diện mạnh như ở đây, thường phát triển 4–6 yếu tố có ý nghĩa, khoảng 300–600 từ tiếng Việt; không chỉ viết một đoạn tóm tắt. " if len(dimensions) >= 4 else
               "Nguồn hiện chỉ hỗ trợ ít phương diện mạnh: giải thích những nguyên nhân đó và nói rõ giới hạn; nếu chỉ có hai nguyên nhân thì chỉ trình bày hai. ")
            + "Đây là mục tiêu linh hoạt theo bằng chứng, không phải độ dài tối thiểu. Không lặp ý hoặc thêm dữ kiện để đủ số từ. "
            + ("Phân biệt phạm vi của từng chủ thể: " + ", ".join(analysis.actors) + "." if analysis.actors else "")
            + (" Bằng chứng chưa xác lập riêng phạm vi " + ", ".join(plan["unresolved_actor_scopes"]) + "; nói rõ giới hạn, không chuyển nguyên nhân giữa các chủ thể." if plan.get("unresolved_actor_scopes") else ""))


def answer_coverage(answer, plan, analysis, config):
    supported = set(plan.get("strong_evidence_dimensions", []))
    expressed = set(dimension_spans(answer)) & supported
    active = analysis.answer_depth == "broad_analysis" and analysis.question_type == "cause"
    shallow = active and len(supported) >= config.analytical_coverage_support_threshold and len(expressed) < config.analytical_coverage_min_dimensions
    return {"answer_dimensions_expressed": sorted(expressed),
            "analytical_coverage_ratio": round(len(expressed) / len(supported), 4) if supported else None,
            "analytical_coverage_too_shallow": shallow}
