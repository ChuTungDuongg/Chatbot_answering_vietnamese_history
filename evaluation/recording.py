"""Translate existing Central traces conservatively; preserve raw JSON for rescoring."""
from evaluation.schema import EvaluationRecord


def from_result(question, metadata, result, *, latency_ms=None):
    debug = result.get("central_debug") or {}
    performance = result.get("performance_debug") or {}
    provenance = result.get("answer_provenance") or {}
    analysis = result.get("analysis")
    signals = {}

    def put(name, value):
        if value is not None:
            signals[name] = value

    status = result.get("status")
    if status is not None:
        signals.update(success=status == "ok", validation_failure=status == "answer_validation_failed",
                       insufficient=status in {"insufficient_evidence", "evidence_insufficient"})
    if "final_failure_reason" in result:
        signals["failure"] = result["final_failure_reason"] is not None
    for name, key in {"partial": "partial_answer", "cold_start": "model_was_cold",
                      "dimension_coverage": "analytical_coverage_ratio",
                      "citation_alignment": "citation_alignment_used"}.items():
        put(name, debug.get(key))
    for name, key in {"model_calls": "central_model_calls", "tool_calls": "central_tool_calls",
                      "input_tokens": "central_input_tokens", "output_tokens": "central_output_tokens"}.items():
        put(name, performance.get(key))
    put("latency_ms", latency_ms if latency_ms is not None else performance.get("central_total_latency_ms"))
    if "tool_calls" in signals:
        signals["used_tools"] = signals["tool_calls"] > 0
    if status is not None and "model_calls" in signals:
        signals["one_generation_success"] = status == "ok" and signals["model_calls"] == 1
    if "tool_parse_failures" in debug:
        signals["parse_failure"] = debug["tool_parse_failures"] > 0
    if "malformed_tool_calls" in debug:
        signals["malformed_call"] = bool(debug["malformed_tool_calls"])
    for signal, key in (("unsupported_names", "unsupported_named_claims"), ("unsupported_years", "unsupported_years")):
        if key in result:
            signals[signal] = bool(result[key])
    risks = debug.get("grounding_risk_checks")
    if risks and all(key in risks[-1] for key in ("unsupported_named_claims", "unsupported_years")):
        signals["grounding_pass"] = not (risks[-1]["unsupported_named_claims"] or risks[-1]["unsupported_years"])
    traces = debug.get("tools")
    if traces is not None:
        signals["search_history"] = any(row.get("name") == "search_history" for row in traces)
        checked = [row for row in traces if "error" in row]
        signals.update(checked_tools=len(checked), successful_tools=sum(row["error"] is None for row in checked))
    legacy = [performance.get(name + "_generation_calls") for name in ("research", "evidence", "history")]
    if all(value is not None for value in legacy):
        signals["legacy_calls"] = sum(legacy)
    sources = result.get("source_chunks")
    checked_sources = [row for row in sources or [] if isinstance(row.get("target_consistent"), bool)]
    if checked_sources:
        signals.update(target_checked_sources=len(checked_sources),
                       target_consistent_sources=sum(row["target_consistent"] for row in checked_sources))
    facets, covered = debug.get("requested_facets"), debug.get("covered_facets")
    if facets and covered is not None:
        signals["facet_coverage"] = len(set(facets) & set(covered)) / len(set(facets))
    actors, coverage = debug.get("required_actor_coverage"), debug.get("selected_actor_coverage")
    if actors and len(actors) > 1 and coverage is not None:
        signals["actor_coverage"] = sum(bool(coverage.get(actor)) for actor in actors) / len(actors)
    balance = debug.get("comparison_balance")
    if balance and all(isinstance(value.get("adequate"), bool) for value in balance.values()):
        signals["comparison_balance"] = sum(value["adequate"] for value in balance.values()) / len(balance)
    if debug.get("answer_depth") in {"focused_explanation", "broad_analysis"} and "analytical_coverage_too_shallow" in debug:
        signals["depth_compliant"] = not debug["analytical_coverage_too_shallow"]
    if (analysis or {}).get("relation_requested") and "relationship_answer_issues" in debug:
        signals["relationship_valid"] = not debug["relationship_answer_issues"]
    repair_keys = ("repair_used", "citation_alignment_used", "citation_repair_used", "deterministic_claim_removal_used")
    known_repairs = [debug[key] for key in repair_keys if key in debug]
    if any(known_repairs) or all(key in debug for key in repair_keys[:3]):
        signals["repair"] = any(known_repairs)
    generations = performance.get("generation_metrics")
    if generations is not None:
        signals["repair_generation"] = any(row.get("stage") in {"repair", "quality_repair", "citation_repair"} for row in generations)
    if "citation_alignment_used" in debug and "citation_repair_used" in debug:
        signals["citation_repair"] = debug["citation_alignment_used"] or debug["citation_repair_used"]
        issues = debug.get("answer_quality_issues")
        if signals["citation_repair"] and issues is not None:
            signals["citation_repair_success"] = not any("citation" in issue or issue == "uncited_factual_paragraphs" for issue in issues)
    viewpoints = debug.get("viewpoint_attribution_issues")
    if viewpoints is not None:
        signals.update(viewpoint_flagged=bool(viewpoints), quote_issue=any(row.get("type") == "direct_quote" for row in viewpoints),
                       paraphrase_issue=any(row.get("type") != "direct_quote" for row in viewpoints))
    if "repair_viewpoint_plan" in debug:
        signals["viewpoint_repair"] = bool(debug["repair_viewpoint_plan"]) and bool(signals.get("repair"))
        if signals["viewpoint_repair"] and viewpoints is not None:
            signals["viewpoint_repair_success"] = not viewpoints
    # Optional complete statistics from future instrumentation/review. Do not
    # guess denominators from unique IDs, truncated traces, or uncited-only lists.
    citation_stats = result.get("citation_stats")
    if citation_stats:
        for name in ("valid_citations", "invalid_citations", "total_citations", "uncited_paragraphs",
                     "factual_paragraphs", "target_matched_citations", "target_checked_citations"):
            put(name, citation_stats.get(name))
    put("filter_collapse", debug.get("suspected_filter_collapse"))
    return EvaluationRecord(
        run_id=metadata.run_id, variant=metadata.variant, question_id=question.id,
        question=question.question, category=question.category, parsed_semantics=analysis,
        answer=result.get("answer"), status=status, final_failure_reason=result.get("final_failure_reason"),
        sources=sources, selected_evidence=(result.get("retrieval") or {}).get("final_context"),
        tool_trace=traces, validation_issues=debug.get("answer_quality_issues"),
        citations={key: value for key, value in debug.items() if "citation" in key or key == "uncited_factual_paragraphs"},
        repairs={key: value for key, value in {**provenance, **debug}.items() if "repair" in key},
        usage=performance, adapter_configured=debug.get("central_adapter_configured"),
        adapter_loaded=debug.get("central_adapter_loaded"),
        graph_name=result.get("graph_name"), graph_version=result.get("graph_version"),
        graph_topology_fingerprint=result.get("graph_topology_fingerprint"),
        graph_nodes_executed=result.get("graph_nodes_executed"), graph_route=result.get("graph_route"),
        node_timings=result.get("node_timings"), node_model_calls=result.get("node_model_calls"),
        node_tool_calls=result.get("node_tool_calls"), graph_trace=result.get("graph_trace"),
        signals=signals, raw_result=result,
    )
