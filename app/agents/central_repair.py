"""Bounded host-side repair helpers. Never splice arbitrary clauses or invent prose."""
from __future__ import annotations

import re

from app.agents.central_depth import answer_coverage
from app.agents.central_question import _ascii_fold_vietnamese


def issue_fingerprint(issues, citations, risks, coverage):
    normalize = lambda text: " ".join(_ascii_fold_vietnamese(text).split())
    return frozenset(
        [("issue", issue) for issue in issues]
        + [("viewpoint", issue["type"], normalize(issue["answer_claim"]),
            normalize(issue.get("matched_sensitive_span") or ""), issue.get("attribution_hint")) for issue in citations.viewpoint_issues]
        + [(key, normalize(value)) for key in ("unsupported_named_claims", "unsupported_years") for value in risks.get(key, [])]
        + [("invalid_citation", value) for value in citations.invalid]
        + [("target", value) for value in citations.target_mismatches]
        + ([("coverage", *coverage.get("answer_dimensions_expressed", []))] if coverage.get("analytical_coverage_too_shallow") else [])
    )


def issue_count(issues, citations, risks):
    return (len(set(issues) - {"unattributed_viewpoint", "unsupported_evidence_claim"})
            + len(citations.viewpoint_issues) + len(risks.get("unsupported_named_claims", []))
            + len(risks.get("unsupported_years", [])))


def remove_optional_viewpoint(answer, issues, citations, analysis, plan, config):
    if analysis.answer_depth != "broad_analysis" or analysis.question_type != "cause" or set(issues) != {"unattributed_viewpoint"} or len(citations.viewpoint_issues) != 1:
        return None
    paragraphs = re.split(r"\n\s*\n", answer)
    claim = citations.viewpoint_issues[0]["answer_claim"].strip()
    matches = [i for i, p in enumerate(paragraphs) if re.sub(r"\[S\d+\]", "", p).strip() == claim]
    if len(matches) != 1:
        return None
    index = matches[0]
    paragraph = paragraphs[index].strip()
    # Only a standalone, complete prose sentence. Lists, headings, tables and
    # multi-sentence paragraphs need a model rewrite to preserve structure.
    if paragraph.startswith(("#", "|", "-", "*", '“', '"')) or re.match(r"\d+[.)]", paragraph):
        return None
    if len(re.findall(r"[.!?](?:\s|$)", claim)) != 1 or not claim.endswith((".", "!", "?")):
        return None
    following = paragraphs[index + 1:]
    if following and re.match(r"(?:dieu (?:nay|do)|nhan dinh (?:nay|do)|vi vay|do do|tu do)\b", _ascii_fold_vietnamese(following[0].strip())):
        return None
    remaining = paragraphs[:index] + following
    candidate = "\n\n".join(remaining).strip()
    expressed = answer_coverage(candidate, plan, analysis, config)["answer_dimensions_expressed"]
    if len(expressed) < max(3, config.analytical_coverage_min_dimensions) or sum(bool(re.search(r"\[S\d+\]", p)) for p in remaining) < 2:
        return None
    return candidate
