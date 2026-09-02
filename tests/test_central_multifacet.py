"""Production multi-facet / quote cases, deterministic retrieval and fake Central only."""
from dataclasses import replace

import pytest

from app.agents.central_analytical import annotate_evidence
from app.agents.central_citations import check_citations
from app.agents.central_compaction import excerpt_evidence
from app.agents.central_evidence import build_evidence_packet, select_evidence, select_synthesis_evidence
from app.agents.central_facets import evidence_facets, viewpoint_cost
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_question import analyze_central_question, plan_analytical_queries
from app.agents.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent
from tests.test_central_consolidated import row


QUESTION = "Vì sao Mỹ lại thua chiến tranh Việt Nam và hệ quả lâu dài của nó?"
EVENT = "Chiến tranh Việt Nam"
CAUSE = ("Chiến tranh Việt Nam: Mỹ thất bại vì chiến lược quân sự không đạt mục tiêu chính trị. "
         "Chi phí chiến tranh kéo dài gây áp lực kinh tế, trong khi phong trào phản chiến và sự thiếu ủng hộ trong nước "
         "hạn chế khả năng duy trì lực lượng. Các nguyên nhân chính trị và quân sự tác động lẫn nhau.")
EFFECT = ("Hệ quả lâu dài của Chiến tranh Việt Nam bao gồm tổn thất xã hội, di chứng sức khỏe và thiệt hại môi trường. "
          "Sau chiến tranh, việc phục hồi kinh tế và tái thiết cơ sở hạ tầng đòi hỏi nhiều nguồn lực. "
          "Những hậu quả này kéo dài nhiều năm và ảnh hưởng đến đời sống nhân dân.")
CAUSE_ANSWER = "## Nguyên nhân\n\nMỹ thất bại trong Chiến tranh Việt Nam vì chiến lược quân sự không đạt mục tiêu chính trị, đồng thời chi phí kinh tế và phong trào phản chiến hạn chế khả năng duy trì chiến tranh. [cause]"
FULL_ANSWER = CAUSE_ANSWER + "\n\n## Hệ quả lâu dài\n\nChiến tranh Việt Nam để lại tổn thất xã hội, di chứng sức khỏe, thiệt hại môi trường và nhu cầu phục hồi kinh tế sau chiến tranh. [effect]"
CONFIG = CentralAgentConfig()


@pytest.mark.parametrize("question,event,facets", [
    (QUESTION, EVENT, {"cause", "consequence"}),
    ("Vì sao Mỹ thua Chiến tranh Việt Nam, và điều đó có ý nghĩa gì?", EVENT, {"cause", "significance"}),
    ("Nguyên nhân thắng lợi của Chiến thắng Bạch Đằng năm 938 và ý nghĩa lịch sử?", "Chiến thắng Bạch Đằng năm 938", {"cause", "significance"}),
    ("Bối cảnh, diễn biến và kết quả của Cách mạng Tháng Tám?", "Cách mạng Tháng Tám", {"context", "method", "result"}),
    ("Vì sao Chiến tranh Việt Nam kết thúc và điều đó dẫn đến hệ quả gì?", EVENT, {"cause", "consequence"}),
])
def test_event_boundaries_and_independent_facet_plans(question, event, facets):
    analysis = analyze_central_question(question)
    assert analysis.event == event
    assert set(analysis.facets) == facets
    plan = plan_analytical_queries(analysis)
    assert set(plan) == {f"facet:{facet}" for facet in facets}
    assert all(event in query for queries in plan.values() for query in queries)
    assert all("của nó" not in query for queries in plan.values() for query in queries)


def test_exact_production_two_primary_queries_one_synthesis_no_filter_collapse():
    history = FakeTool("search_history", lambda args: [row("effect", EVENT, EFFECT)] if "hệ quả" in args["query"] else [row("cause", EVENT, CAUSE)])
    runtime = FakeCentralRuntime([CentralGeneration(content=FULL_ANSWER)])
    result = build_agent(runtime, history, config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert debug["canonical_event"] == EVENT
    assert debug["raw_event_clause"].endswith("và hệ quả lâu dài của nó")
    assert debug["event_resolution_events"]
    assert debug["requested_facets"] == debug["covered_facets"] == ["cause", "consequence"]
    assert debug["unresolved_facets"] == [] and not debug["partial_answer"]
    assert len(history.calls) == 2 and len(runtime.calls) == 1
    assert len(debug["retrieval_queries_skipped"]) == 2
    assert not debug["repair_used"] and not debug["suspected_filter_collapse"]
    assert {r["retrieval_facet"] for r in result["source_chunks"]} == {"cause", "consequence"}
    assert all(r["evidence_facets"] for r in result["source_chunks"])
    prompt = runtime.calls[0]["messages"][-1]["content"]
    assert "evidence_facets: cause" in prompt and "Hệ quả lâu dài" in prompt


def test_overview_and_related_event_rows_survive_and_unrelated_event_does_not():
    analysis = analyze_central_question(QUESTION)
    candidates = [row(str(i), EVENT if i % 2 else "Việt Nam hóa chiến tranh", CAUSE) for i in range(25)]
    candidates.append(row("wrong", "Chiến tranh thế giới thứ hai", "Một sự kiện khác có nhiều hậu quả."))
    kept, debug = select_evidence(candidates, analysis, CONFIG)
    assert len(kept) == 25
    assert debug["entity_disambiguation_filtered_count"] == 1


def test_missing_consequence_gets_bounded_canonical_fallback_and_partial_answer():
    history = FakeTool("search_history", [row("cause", EVENT, CAUSE)])
    wikipedia = FakeTool("search_wikipedia", [])
    runtime = FakeCentralRuntime([CentralGeneration(content=CAUSE_ANSWER)])
    result = build_agent(runtime, history, wikipedia, FakeTool("fetch_wikipedia_page", []), config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert debug["partial_answer"] and debug["unresolved_facets"] == ["consequence"]
    assert "chưa đủ" in result["answer"] and "hệ quả lâu dài" in result["answer"]
    assert len(history.calls) == 3 and len(wikipedia.calls) == 1
    assert wikipedia.calls[0]["query"] == EVENT + " hệ quả lâu dài"
    assert len(runtime.calls) == 1 and not debug["repair_used"]


def test_missing_facet_cannot_be_filled_from_memory_and_repair_is_fresh():
    bad = FULL_ANSWER.replace("[effect]", "[cause]")
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=CAUSE_ANSWER)])
    result = build_agent(runtime, FakeTool("search_history", [row("cause", EVENT, CAUSE)]), config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    assert "unsupported_consequence_section" in debug["repair_reasons"]
    assert result["status"] == "ok" and debug["answer_quality_issues"] == []
    assert len(runtime.calls) == 2 and debug["answer_validation_stage"] == "quality_repair"


def test_no_requested_facet_means_no_generation_and_filter_collapse_is_diagnostic():
    unrelated = row("wrong", "Một thành phố", "Kinh tế quốc tế và văn hóa du lịch. " * 10)
    runtime = FakeCentralRuntime([])
    result = build_agent(runtime, FakeTool("search_history", [unrelated]), config=CONFIG).chat(QUESTION)
    assert result["status"] == "insufficient_evidence"
    assert result["central_debug"]["suspected_filter_collapse"]
    assert len(runtime.calls) == 0


def test_excerpt_facets_do_not_inherit_query_or_discarded_page_metadata():
    item = row("cause", EVENT, CAUSE)
    item.update(retrieval_facet="consequence", evidence_facets=["consequence"])
    annotated = annotate_evidence(item, analyze_central_question(QUESTION))
    assert "consequence" not in annotated["evidence_facets"]
    assert "cause" in annotated["evidence_facets"]
    assert evidence_facets({"text": "Mỹ rút quân. Kết quả là chiến tranh kết thúc."}) == ["result"]


def test_undated_title_does_not_accept_evidence_for_a_different_battle_year():
    analysis = analyze_central_question("Vì sao Chiến thắng Bạch Đằng năm 938 thành công và ý nghĩa lịch sử?")
    assert not annotate_evidence(row("wrong-year", "Bạch Đằng", "Trận Bạch Đằng năm 1288 có ý nghĩa lịch sử."), analysis)["target_consistent"]


def test_canonical_wikipedia_fallback_covers_only_missing_facet_without_action_generation():
    history = FakeTool("search_history", [row("cause", EVENT, CAUSE)])
    wikipedia = FakeTool("search_wikipedia", [{"page_id": "war-page", "title": EVENT, "snippet": "Hệ quả lâu dài."}])
    fetch = FakeTool("fetch_wikipedia_page", row("effect", EVENT, EFFECT))
    runtime = FakeCentralRuntime([CentralGeneration(content=FULL_ANSWER)])
    result = build_agent(runtime, history, wikipedia, fetch, config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok", result["central_debug"]
    assert len(wikipedia.calls) == len(fetch.calls) == len(runtime.calls) == 1
    assert "hệ quả lâu dài" in wikipedia.calls[0]["query"]
    assert result["central_debug"]["unresolved_facets"] == []
    effect = next(source for source in result["source_chunks"] if source["chunk_id"] == "effect")
    assert effect["retrieval_facet"] == "consequence" and "consequence" in effect["evidence_facets"]


QUOTES = ['một chiến lược khôn khéo đến mức nguy hiểm để đánh bại Mỹ', 'Nước ngoài không bao giờ có thể địch nổi chiến lược đó']


@pytest.mark.parametrize("quote", QUOTES)
def test_real_quotes_require_named_speaker_not_generic_scholars(quote):
    packet = build_evidence_packet([row("quote", EVENT, f'Noam Chomsky nhận định: "{quote}".')])
    for prefix in ["", "Theo một số học giả, "]:
        issues = check_citations(f'{prefix}"{quote}". [S1]', packet).viewpoint_issues
        assert issues and issues[0]["type"] == "direct_quote"
        assert issues[0]["attribution_hint"] == "Noam Chomsky"
        assert issues[0]["matched_sensitive_span"] == quote
    assert check_citations(f'Theo nhận định của Noam Chomsky, "{quote}". [S1]', packet).viewpoint_issues == []


@pytest.mark.parametrize("action", ["attribute", "neutralize"])
def test_single_quote_repair_receives_span_hint_and_revalidates(action):
    quoted = f'Noam Chomsky nhận định: "{QUOTES[0]}". '
    source = row("cause", EVENT, CAUSE + quoted)
    first = f'Theo một số học giả, "{QUOTES[0]}". [cause]'
    fixed = f'Theo Noam Chomsky, "{QUOTES[0]}". [cause]' if action == "attribute" else CAUSE_ANSWER
    runtime = FakeCentralRuntime([CentralGeneration(content=first), CentralGeneration(content=fixed)])
    result = build_agent(runtime, FakeTool("search_history", [source]), config=CONFIG).chat("Vì sao Mỹ thua chiến tranh Việt Nam?")
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert len(runtime.calls) == 2 and debug["repair_viewpoint_action"] == action
    assert debug["viewpoint_attribution_issues"] == []
    assert "Noam Chomsky" in runtime.calls[-1]["messages"][-1]["content"]
    assert QUOTES[0] in runtime.calls[-1]["messages"][-1]["content"]


def test_neutral_selection_and_compaction_preserve_simple_cause_fast_path():
    dense_quote = row("opinion", EVENT, 'Noam Chomsky nhận định: "' + CAUSE + ' ' + QUOTES[0] + '".')
    neutrals = [row(f"neutral-{i}", EVENT, CAUSE + f" Tư liệu {i}.") for i in range(4)]
    analysis = analyze_central_question("Vì sao Mỹ thua chiến tranh Việt Nam?")
    small = replace(CONFIG, analytical_max_sources=3)
    selected = select_synthesis_evidence([dense_quote, *neutrals], analysis, small)
    assert all(source["chunk_id"].startswith("neutral") for source in selected)
    long_source = dense_quote["text"] + "\n\n" + CAUSE + "\n\n" + "Phụ lục tài liệu. " * 90
    excerpt = excerpt_evidence(long_source, analysis, 430)
    assert "chiến lược quân sự" in excerpt and viewpoint_cost(excerpt) == 0
    runtime = FakeCentralRuntime([CentralGeneration(content=CAUSE_ANSWER.replace("[cause]", "[neutral-0]"))])
    result = build_agent(runtime, FakeTool("search_history", neutrals), config=CONFIG).chat(analysis.question)
    assert result["status"] == "ok"
    assert len(runtime.calls) == 1 and len(result["central_debug"]["retrieval_queries_executed"]) == 1
    assert result["central_debug"]["neutral_evidence_selected_count"] >= 3
