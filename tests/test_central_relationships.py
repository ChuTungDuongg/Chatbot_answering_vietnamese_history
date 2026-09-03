"""Two-entity biography regressions using synthetic/local evidence and fake models."""
from dataclasses import replace

import pytest

from app.agents.central.analytical import coverage_report
from app.agents.central.compaction import compact_history
from app.agents.central.evidence import build_evidence_packet, select_evidence, select_synthesis_evidence
from app.agents.central.grounding import entity_alias_matches, grounding_risks
from app.agents.central.model_runtime import CentralGeneration
from app.agents.central.question import analyze_central_question, plan_analytical_queries
from app.agents.central.relationships import RELATION_CAVEAT, relationship_answer_issues, relationship_coverage, relation_spans
from app.agents.central.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent
from tests.test_central_consolidated import row


PRIMARY, RELATED = "Võ Nguyên Giáp", "Nguyễn Văn Thiệu"
QUESTION = f"{PRIMARY} là ai, có liên hệ gì với {RELATED}?"
CONFIG = CentralAgentConfig()


def sources():
    return [
        row("p1", PRIMARY, f"{PRIMARY} là một nhà chỉ huy quân sự Việt Nam. Ông hoạt động trong quân đội và tham gia tổ chức lực lượng trong chiến tranh.", .999),
        row("p2", PRIMARY, f"{PRIMARY} giữ vai trò lãnh đạo quân sự và tham gia xây dựng lực lượng. Tư liệu giới thiệu các hoạt động của ông trong nhiều giai đoạn lịch sử.", .99),
        row("r1", RELATED, f"{RELATED} là một chính khách và sĩ quan quân đội. Ông giữ chức Tổng thống Việt Nam Cộng hòa trong thời kỳ Chiến tranh Việt Nam.", .95),
        row("r2", RELATED, f"{RELATED} hoạt động trong chính quyền Việt Nam Cộng hòa. Các tư liệu ghi lại sự nghiệp quân sự và chính trị của ông qua nhiều giai đoạn.", .9),
        row("noise1", "Vụ án phố Ôn Như Hầu", f"Tư liệu nhắc đến {PRIMARY} khi kể lại một vụ án. Nội dung chủ yếu mô tả những diễn biến của vụ án và các hoạt động điều tra.", .98),
        row("noise2", "Lê Trọng Nghĩa", f"Lê Trọng Nghĩa hoạt động trong quân đội. Trang này có nhắc đến {PRIMARY} nhưng chủ yếu viết về sự nghiệp riêng của nhân vật khác.", .97),
    ]


GOOD = f"{PRIMARY} là một nhà chỉ huy quân sự Việt Nam. [S1]\n\n{RELATED} là một chính khách và sĩ quan quân đội. [S2]\n\n" + RELATION_CAVEAT


@pytest.mark.parametrize("question", [
    QUESTION,
    f"{PRIMARY} có quan hệ gì với {RELATED}?",
    f"{PRIMARY} và {RELATED} có liên quan gì đến nhau?",
    f"Mối quan hệ giữa {PRIMARY} và {RELATED}",
    f"{PRIMARY} có từng gặp {RELATED} không?",
    f"{PRIMARY} có từng làm việc với {RELATED} không?",
    f"{PRIMARY} có từng đối đầu với {RELATED} không?",
    f"{PRIMARY} có liên hệ như thế nào với {RELATED}?",
    f"{PRIMARY} là ai và vai trò của {PRIMARY} đối với {RELATED}?",
])
def test_relation_parser_preserves_both_requested_people(question):
    analysis = analyze_central_question(question)
    assert analysis.question_type == "biography"
    assert analysis.subject == PRIMARY and analysis.related_entities == (RELATED,)
    assert analysis.relation_requested and analysis.relation_phrase
    assert analysis.facets == ("identity", "relationship") and analysis.comparison_targets == ()
    assert [q for queries in plan_analytical_queries(analysis).values() for q in queries] == [PRIMARY, RELATED, f"{PRIMARY} {RELATED}"]


def test_relation_parser_is_not_specific_to_production_names():
    analysis = analyze_central_question("Trần Minh có quan hệ gì với Lê An?")
    assert analysis.subject == "Trần Minh" and analysis.related_entities == ("Lê An",)
    assert not analyze_central_question("Nguyễn Cao Kỳ là ai?").relation_requested
    assert not analyze_central_question("So sánh Cách mạng Tháng Tám và Điện Biên Phủ.").relation_requested


def test_related_canonical_sources_survive_biography_filter_and_per_entity_selection():
    analysis = analyze_central_question(QUESTION)
    filtered, debug = select_evidence(sources(), analysis, CONFIG)
    assert {"r1", "r2"} <= {source["chunk_id"] for source in filtered}
    assert "biography_entity_collision" not in debug["retrieval_filter_reasons"]
    selected = select_synthesis_evidence(filtered, analysis, CONFIG)
    assert {source["chunk_id"] for source in selected} == {"p1", "p2", "r1", "r2"}
    assert {role for source in selected for role in source["entity_roles"]} == {"primary_subject", "related_entity"}
    sufficient, coverage = coverage_report(selected, filtered, analysis, CONFIG)
    assert sufficient and coverage["primary_subject_evidence_count"] == coverage["related_entity_evidence_count"] == 2
    assert coverage["direct_relation_evidence_count"] == 0
    assert coverage["partial_answer"] and coverage["unresolved_facets"] == ["relationship"]
    assert not relationship_coverage(selected[:1], analysis)[0]


def test_relation_absent_returns_supported_answer_in_one_generation_and_no_unrelated_history():
    history = [{"role": role, "content": f"Nội dung không liên quan {index}."} for index in range(4) for role in ("user", "assistant")]
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD)])
    tool = FakeTool("search_history", sources())
    result = build_agent(runtime, tool, config=CONFIG).chat(QUESTION, history=history)
    debug = result["central_debug"]
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert [call["query"] for call in tool.calls] == [PRIMARY, RELATED, f"{PRIMARY} {RELATED}"]
    assert debug["partial_answer"] and debug["unresolved_facets"] == ["relationship"]
    assert debug["history_input_turns"] == debug["history_turns_selected"] == 0
    assert debug["history_turns_considered"] == 8 and debug["history_relevance_mode"] == "self_contained"
    assert debug["related_entities"] == [RELATED]
    assert not debug["repair_used"] and not debug["full_quality_repair_used"]
    assert result["final_failure_reason"] is None and RELATION_CAVEAT in result["answer"]
    assert {source["title"] for source in result["source_chunks"]} == {PRIMARY, RELATED}
    assert "không chứng minh họ từng gặp" in runtime.calls[0]["messages"][-1]["content"]


def test_missing_relation_caveat_is_inserted_from_coverage_without_rewriting_facts():
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD.removesuffix(RELATION_CAVEAT).strip())])
    result = build_agent(runtime, FakeTool("search_history", sources()), config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert result["answer"].endswith(RELATION_CAVEAT)
    assert not result["central_debug"]["repair_used"]


def test_primary_subject_alone_cannot_pass_relation_sufficiency_or_trigger_llm_action():
    runtime = FakeCentralRuntime([])
    result = build_agent(runtime, FakeTool("search_history", sources()[:2]), config=CONFIG).chat(QUESTION)
    assert result["status"] == "insufficient_evidence"
    assert result["central_debug"]["related_entity_evidence_count"] == 0
    assert not result["central_debug"]["evidence_sufficient"] and not runtime.calls


@pytest.mark.parametrize("fabrication", [
    f"{PRIMARY} từng gặp {RELATED} vào năm 1970.",
    f"{PRIMARY} là cấp trên của {RELATED}.",
    f"{PRIMARY} là đối thủ cá nhân của {RELATED}.",
    f"{PRIMARY} chưa từng gặp {RELATED}.",
    f"{PRIMARY} và {RELATED} ở hai phía đối lập.",
])
def test_two_biographies_do_not_prove_a_direct_relationship(fabrication):
    analysis = analyze_central_question(QUESTION)
    packet = build_evidence_packet(select_synthesis_evidence(sources(), analysis, CONFIG))
    assert relationship_answer_issues(fabrication + " [S1] [S2]\n\n" + RELATION_CAVEAT, analysis, packet) == ["unsupported_relationship_claim"]


def test_documented_direct_relation_uses_one_synthesis_and_needs_no_fallback():
    # Synthetic people and meeting, not a historical assertion about the production pair.
    question = "Trần Minh có quan hệ gì với Lê An?"
    text = "Trần Minh gặp Lê An trong một buổi họp để trao đổi về công việc chung của cơ quan."
    rows = [row("first", "Trần Minh", "Trần Minh là một cán bộ của cơ quan. Tài liệu này giới thiệu những hoạt động và công việc của ông."),
            row("second", "Lê An", "Lê An là người làm việc tại cơ quan khác. Tài liệu ghi lại quá trình công tác và các hoạt động của ông."),
            row("meeting", "Biên bản buổi họp", text)]
    answer = f"Trần Minh là một cán bộ của cơ quan. [S1]\n\n{text} [S3]"
    runtime = FakeCentralRuntime([CentralGeneration(content=answer)])
    wiki = FakeTool("search_wikipedia", [])
    result = build_agent(runtime, FakeTool("search_history", rows), wiki, FakeTool("fetch_wikipedia_page", []), config=CONFIG).chat(question)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert result["central_debug"]["direct_relation_evidence_count"] == 1
    assert not result["central_debug"]["partial_answer"] and not wiki.calls


def test_cooccurrence_with_a_third_person_or_an_evidence_limit_is_not_direct_relation_proof():
    analysis = analyze_central_question(QUESTION)
    assert relation_spans(f"{PRIMARY} gặp một phóng viên và {RELATED} làm việc ở nơi khác.", analysis) == []
    assert relation_spans(f"Chưa có bằng chứng {PRIMARY} từng gặp {RELATED}.", analysis) == []
    assert relation_spans(f"{PRIMARY} gặp Lê An; {RELATED} cũng có mặt trong danh mục tư liệu.", analysis) == []


def test_relation_fallback_is_bounded_search_then_fetch_and_snippets_never_count_as_proof():
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD)])
    wiki = FakeTool("search_wikipedia", [{"title": PRIMARY, "snippet": sources()[0]["text"]}, {"title": RELATED, "snippet": sources()[2]["text"]}])
    fetch = FakeTool("fetch_wikipedia_page", lambda args: [source for source in sources() if source["title"] == args["page_id_or_title"]][:1])
    result = build_agent(runtime, FakeTool("search_history", sources()), wiki, fetch, config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert len(wiki.calls) == 1 and len(fetch.calls) == 2
    assert result["central_debug"]["relation_fallback_used"]
    assert result["central_debug"]["direct_relation_evidence_count"] == 0
    assert all(source["retrieval_tool"] != "search_wikipedia" for source in result["source_chunks"])


def test_real_followup_keeps_previous_entity_context():
    history = [{"role": "user", "content": f"{RELATED} là ai?"},
               {"role": "assistant", "content": f"{RELATED} từng giữ chức Tổng thống Việt Nam Cộng hòa."}]
    debug = {}
    kept = compact_history("Ông ấy giữ chức vụ đó đến năm nào?", history, max_messages=4, char_budget=2400, debug=debug)
    assert kept == history and debug["history_turns_selected"] == 2
    assert debug["history_relevance_mode"] == "bounded_context"


def test_unsupported_ho_chi_minh_is_removed_by_one_explicit_repair_and_recomputed():
    bad = GOOD + "\n\nNgoài ra, Hồ Chí Minh đã chỉ đạo hoạt động này. [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=GOOD)])
    result = build_agent(runtime, FakeTool("search_history", sources()), config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok" and len(runtime.calls) == 2
    assert result["central_debug"]["unsupported_named_claims"] == []
    assert "Hồ Chí Minh" not in result["answer"]
    prompt = runtime.calls[1]["messages"][-1]["content"]
    assert 'Tên không được hỗ trợ: ["Hồ Chí Minh"]' in prompt and "Xóa hoàn toàn" in prompt


def test_repair_cannot_repeat_an_unsupported_name():
    bad = GOOD + "\n\nHồ Chí Minh chỉ đạo hoạt động này. [S1]"
    result = build_agent(FakeCentralRuntime([CentralGeneration(content=bad)] * 2), FakeTool("search_history", sources()), config=CONFIG).chat(QUESTION)
    assert result["status"] == "answer_validation_failed"
    assert result["central_debug"]["unsupported_named_claims"] == ["Hồ Chí Minh"]


@pytest.mark.parametrize("metadata,expected", [
    ({}, ["Hồ Chí Minh"]),
    ({"entity_aliases": {"Nguyễn Ái Quốc": ["Hồ Chí Minh"]}}, []),
    ({"entity_aliases": {"Tên Không Có Trong Nguồn": ["Hồ Chí Minh"]}}, ["Hồ Chí Minh"]),
])
def test_name_alias_requires_selected_explicit_entity_metadata(metadata, expected):
    packet = build_evidence_packet([row("alias", "Tư liệu", "Nguyễn Ái Quốc hoạt động ở nước ngoài.", metadata=metadata)])
    assert grounding_risks("Hồ Chí Minh hoạt động ở nước ngoài.", "", packet)["unsupported_named_claims"] == expected
    assert bool(entity_alias_matches("Hồ Chí Minh", packet)) is (not expected)


@pytest.mark.parametrize("text", [
    "Nguyễn Ái Quốc, sau này được biết đến với tên Hồ Chí Minh, hoạt động ở nước ngoài.",
    "Nguyễn Sinh Cung (Hồ Chí Minh) được giới thiệu trong nguồn này.",
])
def test_name_alias_stated_in_visible_text_is_supported_and_truncation_removes_it(text):
    packet = build_evidence_packet([row("alias", "Tư liệu", text)])
    assert not grounding_risks("Hồ Chí Minh được giới thiệu trong nguồn.", "", packet)["unsupported_named_claims"]
    assert entity_alias_matches("Hồ Chí Minh", packet)
    trimmed = replace(packet[0], text="Nguyễn Ái Quốc hoạt động ở nước ngoài.")
    assert not trimmed.entity_aliases
    assert grounding_risks("Hồ Chí Minh hoạt động ở nước ngoài.", "", [trimmed])["unsupported_named_claims"] == ["Hồ Chí Minh"]
