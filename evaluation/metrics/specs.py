"""Explicit metric definitions. Rates are fractions; time is milliseconds.

Boolean signals measure request incidence, NOT historical truth. Count ratios
pool only records with both counts. Annotation-only metrics remain unknown until
reviewed. Each result exposes its eligible record count and denominator.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    signal: str
    description: str
    direction: str = "higher"
    denominator: str | None = None
    gate: str | None = None
    annotation: bool = False
    statistic: str = "mean"


GROUPS = {
    "reliability": [
        Metric("final_answer_success_rate", "success", "Requests with host status ok."),
        Metric("explicit_failure_rate", "failure", "Requests with an explicit host failure reason.", "lower"),
        Metric("validation_failure_rate", "validation_failure", "Requests ending answer_validation_failed.", "lower"),
        Metric("evidence_insufficient_rate", "insufficient", "Requests ending with insufficient evidence.", "context"),
        Metric("partial_answer_rate", "partial", "Requests explicitly marked partial by host.", "context"),
        Metric("repair_rate", "repair", "Requests using deterministic or generation repair.", "lower"),
        Metric("repair_success_rate", "success", "Host-success fraction among repaired requests.", gate="repair"),
        Metric("one_generation_success_rate", "one_generation_success", "Host success with exactly one model call."),
    ],
    "grounding": [
        Metric("grounding_validation_pass_rate", "grounding_pass", "Host grounding checks passed; not a truth label."),
        Metric("unsupported_named_claim_rate", "unsupported_names", "Requests with flagged unsupported named claims.", "lower"),
        Metric("unsupported_year_rate", "unsupported_years", "Requests with flagged unsupported years.", "lower"),
        Metric("target_consistency_rate", "target_consistent_sources", "Selected source target-consistency fraction.", denominator="target_checked_sources"),
        Metric("facet_coverage_rate", "facet_coverage", "Host-supported requested facet fraction."),
        Metric("filter_collapse_rate", "filter_collapse", "Requests explicitly marked retrieval filter collapse.", "lower"),
        Metric("evidence_sufficiency_accuracy", "evidence_sufficiency_correct", "Reviewer correctness of evidence sufficiency decision.", annotation=True),
    ],
    "citations": [
        Metric("citation_validity_rate", "valid_citations", "Valid source-reference occurrences / checked occurrences.", denominator="total_citations"),
        Metric("invalid_citation_rate", "invalid_citations", "Invalid source-reference occurrences / checked occurrences.", "lower", "total_citations"),
        Metric("uncited_factual_paragraph_rate", "uncited_paragraphs", "Uncited new-fact paragraphs / classified new-fact paragraphs.", "lower", "factual_paragraphs"),
        Metric("citation_target_match_rate", "target_matched_citations", "Target-matching citations / target-checked citations.", denominator="target_checked_citations"),
        Metric("citation_alignment_repair_rate", "citation_alignment", "Requests using deterministic citation alignment.", "lower"),
        Metric("citation_repair_success_rate", "citation_repair_success", "Citation checks resolved among citation-repaired requests.", gate="citation_repair"),
    ],
    "behavior": [
        Metric("tool_call_rate", "used_tools", "Requests executing at least one tool."),
        Metric("tool_call_success_rate", "successful_tools", "Successful / completed tool trace entries.", denominator="checked_tools"),
        Metric("tool_parse_failure_rate", "parse_failure", "Requests with at least one tool parse failure.", "lower"),
        Metric("malformed_tool_call_rate", "malformed_call", "Requests with at least one malformed tool call.", "lower"),
        Metric("search_history_usage_rate", "search_history", "Requests executing search_history."),
        Metric("mean_tool_calls", "tool_calls", "Mean executed tool calls.", "context"),
        Metric("unnecessary_tool_call_rate", "unnecessary_tool_calls", "Reviewer-marked unnecessary / executed calls.", "lower", "tool_calls", annotation=True),
        Metric("legacy_role_agent_call_count", "legacy_calls", "Total legacy Research/Evidence/History generations; required zero.", "lower", statistic="sum"),
    ],
    "answer_quality": [
        Metric("answer_depth_compliance", "depth_compliant", "Host depth policy passed for checked answers."),
        Metric("analytical_dimension_coverage", "dimension_coverage", "Expressed / strongly supported dimensions, from visible excerpts."),
        Metric("requested_facet_coverage", "facet_coverage", "Requested facets supported in selected evidence."),
        Metric("multi_actor_coverage", "actor_coverage", "Required actors supported in selected evidence; multi-actor only."),
        Metric("comparison_balance", "comparison_balance", "Adequately supported comparison targets / requested targets."),
        Metric("relationship_answer_validity", "relationship_valid", "Host relationship validation passed; not a truth label."),
        Metric("partial_answer_correctness", "partial_answer_correct", "Reviewer correctness among explicitly partial answers.", gate="partial", annotation=True),
        *[Metric(name, name, "Optional reviewer score [0, 1].", annotation=True)
          for name in ("historical_correctness", "completeness", "relevance", "neutrality", "coherence")],
    ],
    "viewpoint": [
        Metric("direct_quote_attribution_error_rate", "quote_issue", "Requests with a direct-quote attribution issue.", "lower"),
        Metric("viewpoint_paraphrase_issue_rate", "paraphrase_issue", "Requests with a viewpoint paraphrase issue.", "lower"),
        Metric("viewpoint_false_positive_rate", "viewpoint_flagged", "Flagged requests among reviewer-labelled negatives.", "lower", gate="viewpoint_negative"),
        Metric("viewpoint_repair_rate", "viewpoint_repair", "Requests receiving viewpoint repair.", "lower"),
        Metric("viewpoint_repair_success_rate", "viewpoint_repair_success", "Resolved viewpoint checks among viewpoint repairs.", gate="viewpoint_repair"),
    ],
    "efficiency": [
        Metric("mean_latency_ms", "latency_ms", "Mean complete request latency.", "lower"),
        Metric("median_latency_ms", "latency_ms", "Median complete request latency.", "lower", statistic="median"),
        Metric("p95_latency_ms", "latency_ms", "Nearest-rank p95 complete request latency.", "lower", statistic="p95"),
        *[Metric("mean_" + name, name, "Mean observed " + name + ".", "context")
          for name in ("model_calls", "tool_calls", "input_tokens", "output_tokens")],
        Metric("repair_generation_rate", "repair_generation", "Requests with a repair model generation.", "lower"),
        Metric("cold_start_rate", "cold_start", "Requests that began with an unloaded model.", "context"),
        Metric("cold_mean_latency_ms", "latency_ms", "Mean latency among cold starts.", "lower", gate="cold_start"),
        Metric("warm_mean_latency_ms", "latency_ms", "Mean latency among known warm starts.", "lower", gate="warm_start"),
    ],
}
