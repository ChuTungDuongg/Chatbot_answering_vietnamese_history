"""Claim-local regressions from the warm CMT8 trace; CPU/fake inference only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agents.central_analytical import annotate_evidence, coverage_report
from app.agents.central_citations import check_citations
from app.agents.central_compaction import excerpt_evidence
from app.agents.central_evidence import build_evidence_packet, select_synthesis_evidence
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_question import analyze_central_question
from app.agents.central_viewpoints import annotate_viewpoints, viewpoint_attribution_issues
from app.agents.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent
from tests.test_central_consolidated import COMPARE, GOOD_COMPARE, comparison_tool, row
from tests.test_central_reliability import QUESTION, FACTS
from tests.test_central_viewpoints import WAR, production_sources


SUMMARY = (
    "Tóm lại, sự thành công của Cách mạng Tháng Tám là do sự kết hợp giữa "
    "khoảng trống quyền lực do Nhật đầu hàng và sự chuẩn bị kỹ lưỡng, lãnh đạo "
    "của Việt Minh cùng sự ủng hộ của nhân dân."
)


def cmt8_sources():
    return [
        row("preparation", "Cách mạng Tháng Tám", FACTS[0]),
        row("context", "Cách mạng Tháng Tám", FACTS[1]),
        row("mobilization", "Cách mạng Tháng Tám",
            'Việt Minh chuẩn bị tổ chức chính trị và lực lượng quân sự, huy động sự ủng hộ của nhân dân. '
            'Nhật đầu hàng tạo khoảng trống quyền lực và thời cơ cho Cách mạng Tháng Tám thành công. '
            'Một người phát biểu nói về “Cách mạng Tháng Tám”. '
            'Một diễn văn tuyên bố: “Chúng ta nhất định thắng lợi”.'),
    ]


@pytest.mark.parametrize("leadership", ["lãnh đạo", "lãnh đạo sáng suốt"])
def test_exact_cmt8_summary_is_neutral_even_when_s3_contains_speech_and_quoted_title(leadership):
    packet = build_evidence_packet(cmt8_sources())
    title_span = next(a for a in packet[2].viewpoint_annotations if a["text"] == packet[2].title)
    assert title_span["requires_attribution"]  # Reproduces the old short-span shortcut.
    answer = SUMMARY.replace("lãnh đạo", leadership) + " [S1] [S3]"
    check = check_citations(answer, packet)
    assert check.viewpoint_issues == [] and not check.unattributed_viewpoints
    assert check.source_ids and not check.invalid and not check.uncited_paragraphs


def test_cmt8_warm_first_synthesis_keeps_one_query_one_generation_and_compaction():
    sources = [{**source, "text": source["text"] + " Trang này mô tả danh mục tài liệu lưu trữ." * 80} for source in cmt8_sources()]
    runtime = FakeCentralRuntime([CentralGeneration(content=SUMMARY + " [S1] [S2] [S3]")])
    tool = FakeTool("search_history", sources)
    result = build_agent(runtime, tool, config=CentralAgentConfig()).chat(QUESTION, history=[])
    debug = result["central_debug"]
    assert result["status"] == "ok" and result["final_failure_reason"] is None
    assert result["answer_provenance"]["central_model_calls"] == len(runtime.calls) == 1
    assert debug["viewpoint_attribution_issues"] == []
    assert not debug["repair_used"] and not debug["full_quality_repair_used"]
    assert not debug["model_was_cold"] and debug["model_load_wait_ms"] == 0
    assert len(debug["retrieval_queries_planned"]) == 2
    assert len(debug["retrieval_queries_executed"]) == len(tool.calls) == 1
    assert len(debug["retrieval_queries_skipped"]) == 1
    assert debug["evidence_sufficient"]
    assert debug["evidence_chars_after_compaction"] < debug["evidence_chars_before_compaction"]
    assert all(len(source["text"]) <= 1600 for source in result["source_chunks"])
    assert "[1]" in result["answer"] and "[S1]" not in result["answer"]


@pytest.mark.parametrize("answer,expected", [
    ("Chúng ta nhất định thắng lợi. [S1]", "direct_quote"),
    ('“Chúng ta nhất định thắng lợi”. [S1]', "direct_quote"),
    ("Theo phát biểu được trích của Hồ Chí Minh, 'chúng ta nhất định thắng lợi'. [S1]", None),
    ("Theo phát biểu được trích trong nguồn, 'chúng ta nhất định thắng lợi'. [S1]", "direct_quote"),
])
def test_short_concrete_first_person_quote_and_attribution(answer, expected):
    packet = build_evidence_packet([row("speech", "Diễn văn", 'Hồ Chí Minh nói: “chúng ta nhất định thắng lợi”.')])
    issues = check_citations(answer, packet).viewpoint_issues
    assert [issue["type"] for issue in issues] == ([expected] if expected else [])
    if issues:
        issue = issues[0]
        assert issue["source_alias"] == "S1"
        assert issue["matched_sensitive_span"] == issue["source_excerpt"] == "chúng ta nhất định thắng lợi"
        assert issue["answer_claim"] and issue["overlap_score"] == 1
        assert issue["attribution_hint"] == "Hồ Chí Minh" and issue["reason"]


@pytest.mark.parametrize("answer,expected", [
    ("VNCH không có cơ sở trong nhân dân. [S1]", "viewpoint_paraphrase"),
    ("Theo nhận định của Noam Chomsky, VNCH không có cơ sở vững chắc trong nhân dân. [S1]", None),
    ("Hoa Kỳ giảm dần mức độ tham chiến trực tiếp trong quá trình Việt Nam hóa chiến tranh. [S1]", None),
    ("VNCH có cơ sở vững chắc trong nhân dân. [S1]", None),
])
def test_specific_opinion_paraphrase_is_separate_from_neutral_facts(answer, expected):
    text = ('Noam Chomsky: “VNCH không có cơ sở thành trì trong nhân dân”. '
            'Hoa Kỳ giảm dần mức độ tham chiến trực tiếp trong quá trình Việt Nam hóa chiến tranh.')
    packet = build_evidence_packet([row("opinion_and_facts", "Chiến tranh Việt Nam", text)])
    assert packet[0].viewpoint_sensitive
    issues = check_citations(answer, packet).viewpoint_issues
    assert [issue["type"] for issue in issues] == ([expected] if expected else [])
    if issues:
        assert issues[0]["matched_sensitive_span"] == "VNCH không có cơ sở thành trì trong nhân dân"
        assert issues[0]["overlap_score"] >= .86


@pytest.mark.parametrize("annotation", [
    {"type": "direct_quote", "requires_attribution": True},
    {"type": "direct_quote", "requires_attribution": True, "text": "Cách mạng Tháng Tám"},
    {"type": "direct_quote", "requires_attribution": True, "text": "chúng ta nhất định thắng lợi", "start": 0, "end": 25},
])
def test_metadata_or_title_alone_cannot_become_a_sensitive_span(annotation):
    source = SimpleNamespace(alias="S3", title="Cách mạng Tháng Tám", text=FACTS[0],
                             viewpoint_sensitive=True, viewpoint_annotations=(annotation,))
    assert viewpoint_attribution_issues(SUMMARY, [source]) == []
    assert viewpoint_attribution_issues("Chúng ta nhất định thắng lợi.", [source]) == []


def test_long_title_quote_is_still_not_a_sensitive_proposition():
    title = "Lịch sử tổ chức chính trị và lực lượng quân sự"
    source = build_evidence_packet([row("title", title, f'Một người nói về “{title}”.')])[0]
    assert viewpoint_attribution_issues(f"{title} được giới thiệu trong tài liệu.", [source]) == []


def test_long_unquoted_copy_requires_eight_contiguous_words_and_actual_source_span():
    quote = "Nhân dân cần được tham gia quyết định tương lai chính trị của đất nước"
    packet = build_evidence_packet([row("speech", "Bài phát biểu", f'Một diễn văn tuyên bố: “{quote}”.')])
    issues = check_citations(quote + ". [S1]", packet).viewpoint_issues
    assert issues[0]["type"] == "direct_quote"
    assert issues[0]["matched_sensitive_span"] == quote
    # Shared subject/topic words without the quoted proposition are insufficient.
    assert not check_citations("Nhân dân tham gia hoạt động chính trị trong nước. [S1]", packet).viewpoint_issues


def test_sensitive_claim_does_not_poison_another_claim_in_the_same_paragraph():
    packet = build_evidence_packet(cmt8_sources())
    answer = SUMMARY + " Chúng ta nhất định thắng lợi. [S1] [S3]"
    issues = check_citations(answer, packet).viewpoint_issues
    assert len(issues) == 1 and issues[0]["type"] == "direct_quote"
    assert "Tóm lại" not in issues[0]["answer_claim"]
    assert issues[0]["attribution_hint"] is None  # "văn tuyên bố" is not a named speaker.


def test_repair_recomputes_all_answer_derived_issues_without_stale_synthesis_state(monkeypatch):
    bad = "Lũ tay sai gây chiến tranh phi nghĩa vào năm 2099. [S999]\n\nNguyễn Văn A lãnh đạo cuộc chiến."
    good = "Thất bại liên quan đến nhiều yếu tố quân sự, chính trị và chiến lược. [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=good)])
    agent = build_agent(runtime, FakeTool("search_history", production_sources()), config=CentralAgentConfig())
    snapshots = []
    check_answer = agent._check_answer
    def capture(state, packet, **kwargs):
        checked = check_answer(state, packet, **kwargs)
        snapshots.append((kwargs["stage"], checked[0], dict(state.evidence_debug)))
        return checked
    monkeypatch.setattr(agent, "_check_answer", capture)
    result = agent.chat(WAR)
    assert result["status"] == "ok" and len(runtime.calls) == 2
    assert snapshots[0][2]["viewpoint_attribution_issues"]
    assert snapshots[0][2]["unsupported_years"] == ["2099"]
    assert snapshots[0][2]["unsupported_named_claims"]
    stage, issues, debug = snapshots[-1]
    assert stage == "quality_repair" and issues == []
    for key in ("viewpoint_attribution_issues", "unsupported_years", "unsupported_named_claims", "citation_target_mismatches", "invalid_citation_aliases"):
        assert debug[key] == [], key
    assert debug["uncited_factual_paragraphs"] == 0


def test_repair_clears_a_real_comparison_target_mismatch():
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD_COMPARE.replace("[S1]", "[S3]")),
                                  CentralGeneration(content=GOOD_COMPARE)])
    result = build_agent(runtime, FakeTool("search_history", comparison_tool), config=CentralAgentConfig()).chat(COMPARE)
    assert result["status"] == "ok" and len(runtime.calls) == 2
    assert result["central_debug"]["repair_reason"] == "comparison_citation_target_mismatch"
    assert result["central_debug"]["citation_target_mismatches"] == []
    assert result["central_debug"]["answer_validation_stage"] == "quality_repair"


@pytest.mark.parametrize("target", ["Cách mạng Tháng Tám", "Khởi nghĩa Lam Sơn"])
def test_cause_selection_prefers_relevant_excerpts_over_poem_and_aftermath(target):
    analysis = analyze_central_question(f"Vì sao {target} thành công?")
    good = [row(f"good{i}", target, fact.replace("Cách mạng Tháng Tám", target), .8) for i, fact in enumerate(FACTS)]
    poem = row("poem", target, f'Một bài thơ tiên đoán sự thành công của {target}, có từ ngữ quân sự, kinh tế, chính trị, chiến lược và ngoại giao quốc tế.', .9999,
               evidence_dimensions=["military", "economic", "political", "strategy", "international"])
    aftermath = row("aftermath", target, f"Sau thắng lợi của {target}, chính quyền tổ chức lực lượng quân sự cho giai đoạn sau và xây dựng kinh tế trong những năm tiếp theo.", .999)
    selected = select_synthesis_evidence([poem, aftermath, *good], analysis, CentralAgentConfig())
    assert {item["chunk_id"] for item in selected} == {item["chunk_id"] for item in good}
    assert coverage_report(selected, [poem, aftermath, *good], analysis, CentralAgentConfig())[0]
    annotated_poem = annotate_evidence(poem, analysis)
    assert annotated_poem["cause_focus_downranked"] and annotated_poem["evidence_dimensions"] == []
    assert not coverage_report([annotated_poem], [poem], analysis, CentralAgentConfig())[0]


def test_cause_selection_and_sufficiency_do_not_inherit_discarded_page_dimensions():
    analysis = analyze_central_question(QUESTION)
    cause = FACTS[0]
    irrelevant = "Bài thơ kể chuyện quân sự, kinh tế, chiến lược, hậu cần và ngoại giao quốc tế. "
    original = irrelevant * 35 + cause + " " + irrelevant * 35
    config = CentralAgentConfig(evidence_excerpt_chars=600)
    selected = select_synthesis_evidence([row("mixed", "Cách mạng Tháng Tám", original)], analysis, config)
    assert len(selected) == 1 and cause in selected[0]["text"]
    assert len(selected[0]["text"]) <= 600 < len(original)
    assert not {"economic", "international", "strategy", "military"} & set(selected[0]["evidence_dimensions"])
    assert "political" in selected[0]["evidence_dimensions"]
    assert not coverage_report(selected, [row("raw", "Cách mạng Tháng Tám", original)], analysis, config)[0]
    assert excerpt_evidence(original, analysis, 600) == selected[0]["text"]


def test_requested_consequence_facet_can_retain_aftermath_evidence():
    analysis = replace(analyze_central_question(QUESTION), facets=("cause", "result"))
    text = "Sau thắng lợi của cuộc khởi nghĩa, chính quyền tổ chức lực lượng quân sự và khôi phục kinh tế."
    annotated = annotate_evidence(row("after", "Cách mạng Tháng Tám", text), analysis)
    assert not annotated["cause_focus_downranked"]
    assert {"political", "military", "economic"} <= set(annotated["evidence_dimensions"])


def test_ordinary_analytical_language_is_not_evaluative_language():
    assert annotate_viewpoints("Thành công, thất bại, lãnh đạo, ủng hộ, thời cơ, chính trị, quân sự, độc lập, suy yếu và chiến thắng.") == []
