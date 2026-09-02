"""Broad analytical quality regressions: local fixtures, CPU and fake Central only."""
from dataclasses import replace

import pytest

from app.agents.central_analytical import annotate_evidence, coverage_report
from app.agents.central_citations import check_citations
from app.agents.central_depth import answer_coverage, depth_contract, dimension_spans, evidence_plan
from app.agents.central_evidence import build_evidence_packet, select_synthesis_evidence, select_evidence
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_question import analyze_central_question, plan_analytical_queries
from app.agents.central_repair import remove_optional_viewpoint
from app.agents.central_viewpoints import viewpoint_repair_plan
from app.agents.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent
from tests.test_central_consolidated import row


EVENT = "Chiến tranh Việt Nam"
US = (
    "Vì sao Mỹ thua chiến tranh Việt Nam?",
    "Vì sao Mỹ lại thua chiến tranh Việt Nam?",
    "Nguyên nhân Mỹ thất bại trong Chiến tranh Việt Nam là gì?",
)
ALLIES = (
    "Vì sao Mỹ và VNCH thua chiến tranh Việt Nam?",
    "Vì sao Mỹ và Việt Nam Cộng hòa thất bại trong Chiến tranh Việt Nam?",
)
CONFIG = CentralAgentConfig()
FACTS = {
    "overview": "Trong Chiến tranh Việt Nam, quân đội Mỹ chịu tổn thất trên chiến trường khiến khả năng duy trì tác chiến bị hạn chế. Chiến lược không đạt mục tiêu chiến tranh khiến Mỹ gặp khó khăn trong việc duy trì can dự.",
    "institutions": "Trong Chiến tranh Việt Nam, chính quyền Việt Nam Cộng hòa gặp khó khăn về tổ chức khiến khả năng kiểm soát bị hạn chế. Sự bất ổn của chính quyền làm suy yếu khả năng phối hợp và duy trì lực lượng.",
    "domestic": "Trong Chiến tranh Việt Nam, phong trào phản chiến gây áp lực trong nước khiến Mỹ phải giảm mức độ can dự. Chi phí chiến tranh tăng gây áp lực ngân sách và hạn chế khả năng duy trì lực lượng lâu dài.",
    "opponent": "Trong Chiến tranh Việt Nam, khả năng huy động và duy trì lực lượng của đối phương khiến Mỹ khó đạt mục tiêu. Đàm phán ngoại giao tạo điều kiện giảm mức độ can dự của Mỹ và tác động đến kết quả chiến tranh.",
}
GOOD = "\n\n".join(text + f" [{key}]" for key, text in FACTS.items())
SHALLOW = "Quân đội Mỹ chịu tổn thất trên chiến trường khiến khả năng tác chiến bị hạn chế. [overview]"
QUOTE = "Tôi đã phải chấp nhận thực tế rằng người dân địa phương ủng hộ lực lượng đối phương và chống lại chính quyền"
PARAPHRASE = "Người dân địa phương chống lại chính quyền và ủng hộ lực lượng đối phương."


def sources():
    return [row(key, EVENT if key == "overview" else key, text, .8) for key, text in FACTS.items()]


def opinion():
    return row("opinion", EVENT, f'“{QUOTE}”.', .99)


def packet_plan(config=CONFIG):
    analysis = analyze_central_question(US[0])
    selected = select_synthesis_evidence(sources(), analysis, config)
    packet = build_evidence_packet(selected)
    return analysis, packet, evidence_plan(packet, analysis, config)


@pytest.mark.parametrize("question,expected", [
    *[(q, "broad_analysis") for q in US + ALLIES],
    ("Ngô Quyền lên ngôi năm nào?", "simple_fact"),
    ("Nguyễn Cao Kỳ là ai?", "biography_summary"),
    ("So sánh Hiệp định Genève và Hiệp định Paris", "comparison"),
    ("Vì sao trận Bạch Đằng năm 938 thắng lợi?", "broad_analysis"),
    ("Vì sao Mỹ thất bại về quân sự trong Chiến tranh Việt Nam?", "focused_explanation"),
])
def test_answer_depth_classification(question, expected):
    analysis = analyze_central_question(question)
    assert analysis.answer_depth == analysis.telemetry()["answer_depth"] == expected


def test_paraphrase_invariance_includes_queries_excerpts_policy_and_scope():
    analyses = [analyze_central_question(q) for q in US]
    assert {(a.question_type, a.event, a.actors, a.outcome, a.answer_depth) for a in analyses} == {
        ("cause", EVENT, ("Mỹ",), "thất bại", "broad_analysis")}
    assert all(plan_analytical_queries(a) == plan_analytical_queries(analyses[0]) for a in analyses)
    pool = sources() + [opinion()]
    selected = [select_synthesis_evidence(pool, a, CONFIG) for a in analyses]
    assert all({(r["chunk_id"], r["text"]) for r in group} == {(r["chunk_id"], r["text"]) for r in selected[0]} for group in selected)
    assert all(coverage_report(group, pool, a, CONFIG)[1]["neutral_evidence_preference"] for a, group in zip(analyses, selected))
    for question in ALLIES:
        analysis = analyze_central_question(question)
        assert analysis.actors == ("Mỹ", "Việt Nam Cộng hòa")
        assert analysis.answer_depth == "broad_analysis"


@pytest.mark.parametrize("question", [US[0], ALLIES[0]])
def test_broad_cause_one_generation_developed_answer(question):
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD, output_tokens=540)])
    result = build_agent(runtime, FakeTool("search_history", sources() + [opinion()]), config=CONFIG).chat(question)
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert len(runtime.calls) == result["answer_provenance"]["central_model_calls"] == 1
    assert not debug["repair_used"] and not debug["full_quality_repair_used"]
    assert debug["answer_depth"] == "broad_analysis"
    assert len(debug["answer_dimensions_expressed"]) >= 3
    assert not debug["analytical_coverage_too_shallow"]
    assert debug["selected_neutral_evidence_count"] == 4
    assert debug["selected_viewpoint_evidence_count"] == 0
    assert debug["viewpoint_attribution_issues"] == []
    assert len(result["source_ids"]) == 4 and "[1]" in result["answer"] and "[S1]" not in result["answer"]
    assert all(debug["selected_actor_coverage"].values())
    prompt = runtime.calls[0]["messages"][-1]["content"]
    assert "EXPECTED DEPTH: broad_analysis" in prompt and "SUPPORTED ANALYTICAL DIMENSIONS" in prompt
    assert "300–600" in prompt and "4–6" in prompt
    assert runtime.calls[0]["max_new_tokens"] == 1536  # Existing sufficient ceiling is unchanged.


def test_coverage_uses_mechanisms_not_length_or_topic_list():
    analysis, _, plan = packet_plan()
    assert {"military", "political", "domestic", "international", "opponent"} <= set(plan["strong_evidence_dimensions"])
    for answer in [SHALLOW, SHALLOW * 30, "Quân sự, chiến lược, chính trị, kinh tế, đối phương và quốc tế."]:
        assert answer_coverage(answer, plan, analysis, CONFIG)["analytical_coverage_too_shallow"]
    assert not answer_coverage(GOOD, plan, analysis, CONFIG)["analytical_coverage_too_shallow"]
    relaxed = replace(CONFIG, analytical_coverage_support_threshold=8)
    assert not answer_coverage(SHALLOW, plan, analysis, relaxed)["analytical_coverage_too_shallow"]


def test_only_two_dimensions_get_two_cause_contract_without_padding():
    analysis = analyze_central_question(US[0])
    packet = build_evidence_packet(sources()[:1])
    plan = evidence_plan(packet, analysis, CONFIG)
    assert set(plan["strong_evidence_dimensions"]) == {"military", "strategy"}
    assert not answer_coverage(SHALLOW, plan, analysis, CONFIG)["analytical_coverage_too_shallow"]
    contract = depth_contract(analysis, plan)
    assert "chỉ có hai nguyên nhân" in contract and "giới hạn" in contract
    assert "300–600" not in contract


def test_simple_fact_remains_concise_grounded_one_call():
    answer = "Ngô Quyền lên ngôi năm 939. [fact]"
    runtime = FakeCentralRuntime([CentralGeneration(content=answer)])
    result = build_agent(runtime, FakeTool("search_history", [row("fact", "Ngô Quyền", answer)]), config=CONFIG).chat("Ngô Quyền lên ngôi năm nào?")
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert result["answer"] == "Ngô Quyền lên ngôi năm 939. [1]"
    prompt = runtime.calls[0]["messages"][-1]["content"]
    assert "EXPECTED DEPTH: simple_fact" in prompt and "300–600" not in prompt


def test_neutral_preference_applies_to_low_density_single_dimension_equivalent():
    base = FACTS["institutions"]
    sensitive = row("mixed", "Chính quyền", base + ' Một diễn văn tuyên bố: “Chúng ta nhất định thắng lợi”.', 1.0)
    pool = [sensitive, *sources()]
    analysis = analyze_central_question(US[0])
    annotated = annotate_evidence(sensitive, analysis)
    assert 0 < annotated["viewpoint_cost"] < .25
    assert annotated["strong_evidence_dimensions"] == ["political"]
    selected = select_synthesis_evidence(pool, analysis, CONFIG)
    assert "mixed" not in {r["chunk_id"] for r in selected}
    assert "institutions" in {r["chunk_id"] for r in selected}


def test_unique_viewpoint_dimension_can_supplement_factual_sources():
    unique = row("social", EVENT, 'Một diễn văn tuyên bố: “Sự ủng hộ của nhân dân giúp huy động lực lượng và khiến khả năng duy trì chiến tranh tăng lên”.')
    pool = sources()[:3] + [unique]
    selected = select_synthesis_evidence(pool, analyze_central_question(US[0]), CONFIG)
    assert "social" in {r["chunk_id"] for r in selected}
    assert sum(r["viewpoint_sensitive"] for r in selected) == 1


def test_historiography_does_not_displace_neutral_core_even_with_high_rank():
    history = row("myth", "Huyền thoại đâm sau lưng Việt Nam", " ".join(FACTS.values()), 1.0)
    selected = select_synthesis_evidence([history, *sources()], analyze_central_question(US[0]), CONFIG)
    assert {r["chunk_id"] for r in selected} == set(FACTS)
    assert selected[0]["overview_anchor"] and selected[0]["source_role"] == "primary_factual"
    analysis = analyze_central_question(US[0])
    assert annotate_evidence(history, analysis)["source_role"] == "historiography"


def test_excerpt_cost_and_dimensions_are_recomputed_before_selection():
    quotation = 'Một diễn văn tuyên bố: “' + FACTS["domestic"] * 8 + '”.'
    original = quotation + "\n\n" + FACTS["overview"] + "\n\n" + "Danh mục quân sự, chính trị, kinh tế, ngoại giao quốc tế. " * 50
    analysis = analyze_central_question(US[0])
    config = replace(CONFIG, evidence_excerpt_chars=600, synthesis_char_budget=2400)
    selected = select_synthesis_evidence([row("mixed", EVENT, original)], analysis, config)
    assert len(selected) == 1 and len(selected[0]["text"]) <= 600 < len(original)
    assert not selected[0]["viewpoint_sensitive"]
    assert set(selected[0]["strong_evidence_dimensions"]) == {"military", "strategy"}
    packet = build_evidence_packet(selected)
    plan = evidence_plan(packet, analysis, config)
    debug = coverage_report(selected, [row("raw", EVENT, original)], analysis, config)[1]
    assert debug["strong_evidence_dimensions"] == plan["strong_evidence_dimensions"]
    assert set(plan["strong_evidence_dimensions"]) == {"military", "strategy"}


@pytest.mark.parametrize("answer,direct", [(f'“{QUOTE}”.', True), (QUOTE + ".", True), (PARAPHRASE, False)])
def test_direct_quote_and_paraphrase_are_distinct(answer, direct):
    issues = check_citations(answer + " [S1]", build_evidence_packet([opinion()])).viewpoint_issues
    assert issues
    assert issues[0]["type"] == ("direct_quote" if direct else "viewpoint_paraphrase")


def test_eight_word_copy_with_low_overlap_is_not_direct_quote():
    text = 'Một diễn văn tuyên bố: “Người dân địa phương ủng hộ lực lượng đối phương và chống lại chính quyền vì những lý do riêng của họ”.'
    answer = "Người dân địa phương ủng hộ lực lượng đối phương, nhưng chính quyền chịu áp lực kinh tế từ ngân sách và tổ chức xã hội trong nước. [S1]"
    issues = check_citations(answer, build_evidence_packet([row("speech", EVENT, text)])).viewpoint_issues
    assert all(issue["type"] != "direct_quote" for issue in issues)


def test_cross_supported_fact_prefers_cited_neutral_sentence():
    packet = build_evidence_packet([row("neutral", EVENT, PARAPHRASE), opinion()])
    check = check_citations(PARAPHRASE + " [S1] [S2]", packet)
    assert check.viewpoint_issues == []
    assert check.viewpoint_cross_support[0]["claim_neutral_support_sources"] == ["S1"]
    assert check_citations(PARAPHRASE + " [S1]", packet).viewpoint_issues == []
    # An uncited neutral source cannot authorize a claim citing only an opinion.
    assert check_citations(PARAPHRASE + " [S2]", packet).viewpoint_issues


def test_opposite_or_unrelated_neutral_sentence_cannot_clear_viewpoint():
    for text in [PARAPHRASE.replace("ủng hộ", "không ủng hộ"), "Người dân địa phương tham gia các hoạt động xã hội."]:
        packet = build_evidence_packet([row("neutral", EVENT, text), opinion()])
        assert check_citations(PARAPHRASE + " [S1] [S2]", packet).viewpoint_issues


def test_null_attribution_provides_concrete_remove_or_neutralize_plan():
    issues = check_citations(PARAPHRASE + " [S1]", build_evidence_packet([opinion()])).viewpoint_issues
    assert issues[0]["type"] == "viewpoint_paraphrase" and issues[0]["attribution_hint"] is None
    plan = viewpoint_repair_plan(issues)
    assert plan[0]["recommended_action"] == "remove_or_neutralize"
    assert plan[0]["claim"] == PARAPHRASE and plan[0]["matched_sensitive_span"] == QUOTE


def test_null_hint_full_repair_removes_claim_and_makes_progress():
    runtime = FakeCentralRuntime([CentralGeneration(content=PARAPHRASE + " [opinion]", output_tokens=162), CentralGeneration(content=GOOD)])
    config = replace(CONFIG, analytical_max_sources=6)
    result = build_agent(runtime, FakeTool("search_history", sources() + [opinion()]), config=config).chat(US[0])
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert len(runtime.calls) == 2 and debug["repair_progress"]
    assert debug["repair_viewpoint_action"] == "neutralize"
    assert debug["repair_issue_count_before"] > debug["repair_issue_count_after"] == 0
    assert '"recommended_action": "remove_or_neutralize"' in runtime.calls[-1]["messages"][-1]["content"]
    assert runtime.calls[-1]["max_new_tokens"] == config.repair_max_new_tokens


def test_isolated_optional_claim_can_be_removed_with_no_second_generation():
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD + "\n\n" + PARAPHRASE + " [opinion]")])
    result = build_agent(runtime, FakeTool("search_history", sources() + [opinion()]), config=replace(CONFIG, analytical_max_sources=6)).chat(US[0])
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert PARAPHRASE not in result["answer"] and len(runtime.calls) == 1
    assert debug["deterministic_claim_removal_used"] and debug["repair_progress"]
    assert debug["repair_viewpoint_action"] == "neutralize" and not debug["repair_used"]


def test_fast_removal_declines_mixed_prose_or_coherence_dependency():
    analysis, _, plan = packet_plan()
    packet = build_evidence_packet(sources() + [opinion()])
    for text in [PARAPHRASE + " Điều này có ảnh hưởng đến kết quả. [S5]", PARAPHRASE + " [S5]\n\nĐiều này khiến kết quả thay đổi. [S1]"]:
        answer = GOOD + "\n\n" + text
        citations = check_citations(answer, packet)
        assert remove_optional_viewpoint(answer, ["unattributed_viewpoint"], citations, analysis, plan, CONFIG) is None


def test_same_repair_issue_stops_after_one_attempt():
    bad = PARAPHRASE + " [opinion]"
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=bad)])
    result = build_agent(runtime, FakeTool("search_history", sources() + [opinion()]), config=replace(CONFIG, analytical_max_sources=6)).chat(US[0])
    debug = result["central_debug"]
    assert result["status"] == "answer_validation_failed" and len(runtime.calls) == 2
    assert debug["repair_progress"] is False
    assert debug["repair_issue_count_before"] == debug["repair_issue_count_after"] > 0
    assert debug["repair_viewpoint_action"] == "neutralize"


def test_shallow_answer_is_not_accepted_after_unchanged_repair():
    runtime = FakeCentralRuntime([CentralGeneration(content=SHALLOW), CentralGeneration(content=SHALLOW)])
    result = build_agent(runtime, FakeTool("search_history", sources()), config=CONFIG).chat(US[0])
    assert result["status"] == "answer_validation_failed"
    assert "analytical_coverage_too_shallow" in result["central_debug"]["answer_quality_issues"]
    assert result["central_debug"]["repair_progress"] is False and len(runtime.calls) == 2


@pytest.mark.parametrize("actor,event", [("Pháp", "Chiến tranh Đông Dương"), ("Anh", "Chiến tranh Độc lập"), ("Liên minh A và Liên minh B", "Chiến tranh Khu Vực")])
def test_cause_grammar_and_depth_generalize_to_unrelated_actor_names(actor, event):
    questions = [f"Vì sao {actor} thất bại trong {event}?", f"Tại sao {actor} lại thua trong {event}?",
                 f"Nguyên nhân dẫn tới thất bại của {actor} trong {event} là gì?"]
    analyses = [analyze_central_question(q) for q in questions]
    expected_actors = tuple(actor.split(" và "))
    assert {(a.event, a.actors, a.outcome, a.answer_depth) for a in analyses} == {(event, expected_actors, "thất bại", "broad_analysis")}
    assert all(plan_analytical_queries(a) == plan_analytical_queries(analyses[0]) for a in analyses)
    unscoped = [analyze_central_question(q.replace(" trong " + event, "")) for q in questions]
    assert {(a.subject, a.actors, a.outcome, a.answer_depth) for a in unscoped} == {(actor, expected_actors, "thất bại", "broad_analysis")}


def test_one_analysis_is_reused_through_runtime_and_compaction(monkeypatch):
    calls = []
    def analyze(question):
        calls.append(question)
        return analyze_central_question(question)
    monkeypatch.setattr("app.agents.central_agent.analyze_central_question", analyze)
    monkeypatch.setattr("app.agents.central_compaction.analyze_central_question", analyze)
    result = build_agent(FakeCentralRuntime([CentralGeneration(content=GOOD)]), FakeTool("search_history", sources()), config=CONFIG).chat(US[0])
    assert result["status"] == "ok" and calls == [US[0]]


def test_multi_actor_cause_without_named_event_accepts_separate_actor_evidence():
    question = "Vì sao Liên minh A và Liên minh B thất bại?"
    facts = [
        row("actor-a", "Liên minh A", "Liên minh A gặp khó khăn trong tổ chức chính quyền khiến khả năng kiểm soát bị hạn chế. Chi phí kinh tế tăng gây áp lực ngân sách và làm suy yếu khả năng duy trì lực lượng."),
        row("actor-b", "Liên minh B", "Liên minh B chịu tổn thất trên chiến trường khiến quân đội khó duy trì tác chiến. Chiến lược không đạt mục tiêu chiến tranh khiến khả năng duy trì can dự bị hạn chế."),
    ]
    answer = "\n\n".join(r["text"] + f' [{r["chunk_id"]}]' for r in facts)
    runtime = FakeCentralRuntime([CentralGeneration(content=answer)])
    result = build_agent(runtime, FakeTool("search_history", facts), config=CONFIG).chat(question)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert all(result["central_debug"]["selected_actor_coverage"].values())
    assert result["central_debug"]["unresolved_actor_scopes"] == []


def test_uncertain_match_is_downranked_and_explicit_wrong_event_is_rejected():
    analysis = analyze_central_question(US[0])
    uncertain = row("uncertain", "Tổng quan", FACTS["overview"].replace("Trong Chiến tranh Việt Nam, ", ""))
    wrong = row("wrong", "Chiến tranh Khác", "Quân đội chịu tổn thất trong Chiến tranh Khác khiến chiến lược không đạt mục tiêu.")
    kept, debug = select_evidence([uncertain, wrong, *sources()], analysis, CONFIG)
    assert "uncertain" in {r["chunk_id"] for r in kept} and "wrong" not in {r["chunk_id"] for r in kept}
    assert debug["retrieval_downrank_reasons"]["target_match_unconfirmed"] == 1
    assert next(r for r in kept if r["chunk_id"] == "uncertain")["target_match_uncertain"]
    assert "uncertain" not in {r["chunk_id"] for r in select_synthesis_evidence(kept, analysis, CONFIG)}


@pytest.mark.parametrize("recoverable", [True, False])
def test_filter_collapse_uses_one_canonical_retrieval_without_bypassing_filters(recoverable):
    ambiguous = [row(f"uncertain{i}", "Tổng quan", text.replace("Trong Chiến tranh Việt Nam, ", "")) for i, text in enumerate(FACTS.values())]
    tool = FakeTool("search_history", lambda args: sources() if recoverable and args["query"] == EVENT else ambiguous)
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD)] if recoverable else [])
    result = build_agent(runtime, tool, config=replace(CONFIG, max_action_rounds=0)).chat(US[0])
    assert result["central_debug"]["filter_collapse_recovery_used"]
    assert sum(call["query"] == EVENT for call in tool.calls) == 1
    assert len(tool.calls) == 3
    assert result["status"] == ("ok" if recoverable else "insufficient_evidence")
    assert len(runtime.calls) == (1 if recoverable else 0)
