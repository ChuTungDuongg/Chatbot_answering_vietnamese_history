from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, replace

import pytest

from app.agents.central_agent import INSUFFICIENT_EVIDENCE_ANSWER
from app.agents.central_citations import check_citations, expand_citations
from app.agents.central_evidence import build_evidence_packet, render_evidence_packet, select_evidence
from app.agents.central_grounding import grounding_risks
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_question import analyze_central_question
from app.agents.config import CentralAgentConfig
from app.api.routes import _build_debug
from app.telemetry import RequestTelemetry, reset_request_telemetry, set_request_telemetry
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent as _build_agent


BIO_QUESTION = "Nguyễn Cao Kỳ là ai, tóm tắt cuộc đời, lịch sử và hoạt động của ông ta"
KY_2 = "hf_wikipedia_nguyễn_cao_kỳ_0002_c24f50461e3f"
KY_0 = "hf_wikipedia_nguyễn_cao_kỳ_0000_52bf6d692d0e"
BACH_DANG_QUESTION = "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?"


def build_agent(runtime, *tools, config=None):
    return _build_agent(runtime, *tools, config=config or CentralAgentConfig())


def biography_rows():
    return [
        {"chunk_id": KY_2, "title": "Nguyễn Cao Kỳ", "reranker_score": 0.86,
         "text": "Nguyễn Cao Kỳ từng giữ chức Thủ tướng và Phó Tổng thống Việt Nam Cộng hòa."},
        {"chunk_id": KY_0, "title": "Nguyễn Cao Kỳ", "reranker_score": 0.999,
         "text": "Nguyễn Cao Kỳ (1930–2011) là một sĩ quan không quân và chính khách."},
        {"chunk_id": "noise_tran", "title": "Trần Cao Vân", "reranker_score": 0.0033, "text": "Trần Cao Vân."},
        {"chunk_id": "noise_nguyen", "title": "Nguyễn Cao", "reranker_score": 0.055, "text": "Nguyễn Cao."},
    ]


def bach_dang_rows():
    return [
        {"chunk_id": "hf_trận_bạch_đằng_938_1", "title": "Bạch Đằng 938",
         "text": "Ngô Quyền lãnh đạo chiến thắng Bạch Đằng năm 938."},
        {"chunk_id": "hf_trận_bạch_đằng_938_2", "title": "Ý nghĩa Bạch Đằng 938",
         "text": "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc kéo dài, khôi phục quyền tự chủ."},
    ]


@pytest.mark.parametrize("question,subject", [
    (BIO_QUESTION, "Nguyễn Cao Kỳ"),
    ("Trương Định là ai?", "Trương Định"),
    ("Hãy tóm tắt tiểu sử Phan Bội Châu", "Phan Bội Châu"),
    ("Cuộc đời và sự nghiệp của Võ Nguyên Giáp", "Võ Nguyên Giáp"),
    ("Sự nghiệp Nguyễn Cao Kỳ", "Nguyễn Cao Kỳ"),
    ("Hoạt động của Phan Bội Châu", "Phan Bội Châu"),
    ("Vai trò của Võ Nguyên Giáp trong chiến dịch", "Võ Nguyên Giáp"),
    ("Những chức vụ của Nguyễn Cao Kỳ", "Nguyễn Cao Kỳ"),
    ("Nguyễn Cao Kỳ từng giữ chức vụ gì?", "Nguyễn Cao Kỳ"),
    ("Hay tom tat tieu su Phan Boi Chau", "Phan Boi Chau"),
    ("nguyen cao ky la ai?", "nguyen cao ky"),
    ("Trần Cao Vân là ai?", "Trần Cao Vân"),
    ("Hồ Chí Minh là ai?", "Hồ Chí Minh"),
    ("Hoạt động của Đặng Tiểu Bình", "Đặng Tiểu Bình"),
    ("Tiểu sử Triệu Quang Phục", "Triệu Quang Phục"),
    (unicodedata.normalize("NFD", "Tiểu sử Nguyễn Cao Kỳ"), "Nguyễn Cao Kỳ"),
])
def test_biography_patterns_and_original_subject(question, subject):
    analysis = analyze_central_question(question)
    assert analysis.question_type == "biography"
    assert analysis.analytical is True
    assert analysis.subject == subject


@pytest.mark.parametrize("question,kind", [
    (BACH_DANG_QUESTION, "significance"),
    ("Vai trò của chiến thắng Bạch Đằng năm 938", "significance"),
    ("Vai trò của trận Bạch Đằng năm 938", "significance"),
    ("Vì sao chiến thắng Bạch Đằng thành công?", "cause"),
    ("So sánh Ngô Quyền và Trương Định", "comparison"),
])
def test_other_question_types_are_preserved(question, kind):
    analysis = analyze_central_question(question)
    assert analysis.question_type == kind
    assert analysis.subject is None


def test_nguyen_cao_ky_end_to_end_one_generation_and_telemetry():
    answer = (
        "Nguyễn Cao Kỳ (1930–2011) là sĩ quan không quân và chính khách. [S1]\n\n"
        "Ông từng giữ chức Thủ tướng và Phó Tổng thống Việt Nam Cộng hòa. [S2]"
    )
    runtime = FakeCentralRuntime([CentralGeneration(content=answer, output_tokens=96)])
    history = FakeTool("search_history", biography_rows())
    telemetry = RequestTelemetry(request_id="bio", inference_mode="central")
    token = set_request_telemetry(telemetry)
    try:
        result = build_agent(runtime, history).chat(BIO_QUESTION)
    finally:
        reset_request_telemetry(token)
    assert result["status"] == "ok"
    assert result["analysis"]["question_type"] == "biography"
    assert result["analysis"]["subject"] == "Nguyễn Cao Kỳ"
    assert {row["title"] for row in result["retrieval"]["final_context"]} == {"Nguyễn Cao Kỳ"}
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["stage"] == "synthesis"
    assert runtime.calls[0]["tools"] == []
    prompt = json.dumps(runtime.calls[0]["messages"], ensure_ascii=False)
    assert "[S1]" in prompt and "[S2]" in prompt
    assert KY_0 not in prompt and KY_2 not in prompt
    assert "Trần Cao Vân" not in prompt
    assert "reranker_score" not in prompt and "rrf" not in prompt
    assert result["source_ids"] == [KY_0, KY_2]
    assert f"[{KY_0}]" in result["answer"] and "[S1]" not in result["answer"]
    provenance = result["answer_provenance"]
    assert provenance["central_model_calls"] == 1
    assert provenance["repair_generation_used"] is False
    assert provenance["central_adapter_loaded"] is False
    for role in ("research", "evidence", "history"):
        assert provenance[f"{role}_generation_calls"] == 0
    debug = result["central_debug"]
    assert debug["retrieval_candidates_before_filter"] == 4
    assert debug["retrieval_candidates_after_filter"] == 2
    assert debug["retrieval_filtered_count"] == 2
    assert sum(debug["retrieval_filter_reasons"].values()) == 2
    assert debug["biography_exact_title_hits"] == 2
    assert debug["evidence_source_count"] == 2
    assert debug["evidence_input_chars"] > 0
    assert debug["repair_avoided_reason"] == "valid_first_synthesis"
    assert telemetry.central_quality["evidence_source_count"] == 2
    assert _build_debug(result)["request"]["subject"] == "Nguyễn Cao Kỳ"
    assert debug["phase_trace"] == ["prepare", "initial_grounding", "synthesis", "final"]


def test_entity_filter_does_not_depend_on_score_calibration():
    rows = biography_rows()
    for row in rows:
        row.pop("reranker_score")
    selected, debug = select_evidence(rows, analyze_central_question(BIO_QUESTION), CentralAgentConfig())
    assert [row["chunk_id"] for row in selected] == [KY_2, KY_0]
    assert debug["retrieval_filter_reasons"] == {"biography_entity_collision": 2}


def test_different_title_referring_to_same_person_and_page_is_retained():
    rows = biography_rows()[:2]
    rows[0]["page_id"] = "person-123"
    rows += [
        {"chunk_id": "career", "title": "Không quân", "text": "Nguyễn Cao Kỳ hoạt động trong không quân.", "reranker_score": 0.9},
        {"chunk_id": "later", "title": "Cuối đời", "page_id": "person-123", "text": "Ông qua đời.", "reranker_score": 0.8},
    ]
    selected, _ = select_evidence(rows, analyze_central_question(BIO_QUESTION), CentralAgentConfig())
    assert {row["chunk_id"] for row in selected} == {KY_0, KY_2, "career", "later"}


def test_no_exact_title_falls_back_and_non_biography_has_no_title_gate():
    rows = biography_rows()[2:]
    for question in (BIO_QUESTION, BACH_DANG_QUESTION):
        selected, _ = select_evidence(rows, analyze_central_question(question), CentralAgentConfig())
        assert selected == rows


@pytest.mark.parametrize("scores,expected", [
    ([0.999, 0.86, 0.055, 0.0033], 2),
    ([9.99, 8.60, 0.55, 0.033], 2),
    ([-0.01, -0.14, -0.945, -0.9967], 2),
    ([0.05, 0.04, 0.03, 0.02], 4),
    ([0.1, 0.1, 0.1], 3),
    ([0.9, 0.01], 2),
    ([None, float("nan"), float("inf")], 3),
])
def test_scale_safe_tail_filter(scores, expected):
    rows = [{"chunk_id": str(i), "text": "evidence", "reranker_score": score} for i, score in enumerate(scores)]
    selected, _ = select_evidence(rows, analyze_central_question("Câu hỏi"), CentralAgentConfig())
    assert len(selected) == expected
    assert rows[0] in selected


def test_probability_floor_is_opt_in_and_preserves_top_candidate():
    rows = [{"reranker_score": score} for score in (0.9, 0.055)]
    analysis = analyze_central_question("Câu hỏi")
    raw = CentralAgentConfig(reranker_score_floor=0.08)
    assert len(select_evidence(rows, analysis, raw)[0]) == 2
    calibrated = replace(raw, reranker_score_mode="probability")
    assert select_evidence(rows, analysis, calibrated)[0] == rows[:1]
    assert len(select_evidence([{"reranker_score": 0.05}, {"reranker_score": 0.01}], analysis, calibrated)[0]) == 2


def test_biography_context_cap_packet_fields_and_duplicate_ids():
    rows = biography_rows()[:2] + [
        {"chunk_id": f"ky_{i}", "title": "Nguyễn Cao Kỳ", "text": "Nguyễn Cao Kỳ", "reranker_score": 0.9}
        for i in range(4)
    ]
    selected, debug = select_evidence(rows, analyze_central_question(BIO_QUESTION), CentralAgentConfig(biography_max_sources=3))
    assert len(selected) == 3
    assert debug["retrieval_filter_reasons"] == {"biography_context_limit": 3}
    packet = build_evidence_packet(selected + selected)
    assert [item.alias for item in packet] == ["S1", "S2", "S3"]
    assert set(asdict(packet[0])) == {"alias", "real_source_id", "title", "source_kind", "text"}
    assert "reranker" not in render_evidence_packet(packet)


@pytest.mark.parametrize("citation,valid,invalid,normalized", [
    ("[S1]", True, [], False),
    ("[S2]", True, [], False),
    ("[S99]", False, ["S99"], False),
    ("[938]", False, [], False),
    ("[1945]", False, [], False),
    ("[source_1]", False, ["source_1"], False),
    ("[source]", False, ["source"], False),
    ("[1]", False, ["1"], False),
    ("[ s 1 ]", True, [], True),
    ("[[S1]]", True, [], True),
    ("[S1, S2]", True, [], True),
    ("[S1, S99]", False, ["S1, S99"], False),
    (f"[{KY_2}]", True, [], True),
    ("[hf_không_được_chọn]", False, ["hf_không_được_chọn"], False),
])
def test_citation_validation_contract(citation, valid, invalid, normalized):
    packet = build_evidence_packet(biography_rows()[:2])
    checked = check_citations("Thông tin. " + citation, packet)
    assert bool(checked.source_ids) is valid
    assert checked.invalid == invalid
    assert checked.normalized is normalized


def test_expansion_is_single_pass_idempotent_and_aliases_request_scoped():
    packet = build_evidence_packet(biography_rows()[:2])
    answer = "Một. [S1] Hai. [S2] Một nữa. [S1]"
    expanded = expand_citations(answer, packet)
    assert expanded == f"Một. [{KY_2}] Hai. [{KY_0}] Một nữa. [{KY_2}]"
    assert expand_citations(expanded, packet) == expanded
    assert check_citations(expanded, packet).source_ids == [KY_2, KY_0]
    other = build_evidence_packet(bach_dang_rows()[:1])
    assert expand_citations("[S1]", other) == "[hf_trận_bạch_đằng_938_1]"
    assert check_citations("[S2]", other).invalid == ["S2"]


def test_frontend_source_ids_preserve_decomposed_unicode_without_reparsing():
    source_id = unicodedata.normalize("NFD", KY_0)
    rows = [{"chunk_id": source_id, "title": "Nguyễn Cao Kỳ", "text": "Nguyễn Cao Kỳ là sĩ quan."}]
    runtime = FakeCentralRuntime([CentralGeneration(content="Nguyễn Cao Kỳ là sĩ quan. [S1]")])
    result = build_agent(runtime, FakeTool("search_history", rows)).chat(BIO_QUESTION)
    assert result["source_ids"] == [source_id]
    assert result["answer"] == f"Nguyễn Cao Kỳ là sĩ quan. [{source_id}]"


@pytest.mark.parametrize("answer", [
    "Ngô Quyền lãnh đạo chiến thắng Bạch Đằng năm 938. [S1]\n\nChiến thắng khôi phục quyền tự chủ.",
    "- Ngô Quyền lãnh đạo chiến thắng Bạch Đằng năm 938. [S1]\n- Chiến thắng khôi phục quyền tự chủ.",
])
def test_each_factual_paragraph_needs_its_own_citation(answer):
    assert check_citations(answer, build_evidence_packet(bach_dang_rows())).uncited_paragraphs == 1


def test_bach_dang_contamination_is_flagged_and_corrected_once():
    bad = "Lê Đại Hành giành chiến thắng Bạch Đằng năm 938, lập Đại Cồ Việt. [S1]"
    good = (
        "Ngô Quyền lãnh đạo chiến thắng Bạch Đằng năm 938. [S1]\n\n"
        "Chiến thắng chấm dứt thời kỳ Bắc thuộc kéo dài, khôi phục quyền tự chủ. [S2]"
    )
    packet = build_evidence_packet(bach_dang_rows())
    risks = grounding_risks(bad, BACH_DANG_QUESTION, packet)
    assert {"Lê Đại Hành", "Đại Cồ Việt"} <= set(risks["unsupported_named_claims"])
    assert grounding_risks(good, BACH_DANG_QUESTION, packet) == {"unsupported_named_claims": [], "unsupported_years": []}
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=good)])
    result = build_agent(runtime, FakeTool("search_history", bach_dang_rows())).chat(BACH_DANG_QUESTION)
    assert result["answer_provenance"]["repair_reason"] == "unsupported_evidence_claim"
    assert result["analysis"]["answer_quality_issues"] == []
    assert "Lê Đại Hành" not in result["answer"]
    assert result["status"] == "ok"
    assert len(runtime.calls) == 2
    assert runtime.calls[1]["tools"] == []
    assert "tool_calls" not in json.dumps(runtime.calls[1]["messages"])


def test_correct_bach_dang_answer_does_not_repair_for_short_length():
    runtime = FakeCentralRuntime([CentralGeneration(content="Chiến thắng Bạch Đằng năm 938 khôi phục quyền tự chủ. [S2]")])
    result = build_agent(runtime, FakeTool("search_history", bach_dang_rows())).chat(BACH_DANG_QUESTION)
    assert result["status"] == "ok"
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("claim,expected", [
    ("Ông lên ngôi năm 939. [S1]", "939"),
    ("Ông giữ chức Thủ tướng. [S1]", "Thủ tướng"),
    ("Ông lập nhà Nguyễn. [S1]", "nhà Nguyễn"),
    ("Ông lập triều đại Nguyễn. [S1]", "triều đại Nguyễn"),
    ("Ông tham gia chiến dịch Điện Biên Phủ. [S1]", "chiến dịch Điện Biên Phủ"),
])
def test_guard_detects_years_offices_dynasties_and_events(claim, expected):
    risks = grounding_risks(claim, BACH_DANG_QUESTION, build_evidence_packet(bach_dang_rows()))
    assert expected in risks["unsupported_years"] + risks["unsupported_named_claims"]


def test_guard_ignores_generic_words_citations_and_duration():
    risks = grounding_risks(
        "Nhà nước và quyền tự chủ có ý nghĩa lịch sử sau hơn 1000 năm. [S1]",
        BACH_DANG_QUESTION, build_evidence_packet(bach_dang_rows()),
    )
    assert risks == {"unsupported_named_claims": [], "unsupported_years": []}


@pytest.mark.parametrize("first,second,expected_status,calls", [
    ("Nguyễn Cao Kỳ là sĩ quan. [ s1 ]", None, "ok", 1),
    ("Nguyễn Cao Kỳ là sĩ quan.", "Nguyễn Cao Kỳ là sĩ quan. [S1]", "ok", 2),
    ("Nguyễn Cao Kỳ là sĩ quan.", "Nguyễn Cao Kỳ là sĩ quan.", "insufficient_evidence", 2),
    ("Nguyễn Cao Kỳ là sĩ quan. [S99]", "Nguyễn Cao Kỳ là sĩ quan. [source_1]", "insufficient_evidence", 2),
])
def test_repair_is_bounded_and_normalization_never_invents_a_citation(first, second, expected_status, calls):
    outputs = [CentralGeneration(content=first, output_tokens=100)]
    if second is not None:
        outputs.append(CentralGeneration(content=second))
    runtime = FakeCentralRuntime(outputs)
    result = build_agent(runtime, FakeTool("search_history", biography_rows())).chat(BIO_QUESTION)
    assert len(runtime.calls) == calls
    assert result["status"] == expected_status
    assert all(call["tools"] == [] for call in runtime.calls)
    if calls == 2:
        assert runtime.calls[1]["stage"] == "quality_repair"
        assert runtime.calls[1]["max_new_tokens"] == 196
    else:
        assert result["central_debug"]["repair_avoided_reason"] == "citation_normalized"
    if expected_status == "insufficient_evidence":
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
        assert result["source_ids"] == []


@pytest.mark.parametrize("tokens,cap,expected", [(10, 1024, 192), (400, 1024, 496), (1000, 512, 512), (10, 128, 128)])
def test_dynamic_repair_budget_is_bounded(tokens, cap, expected):
    agent = build_agent(FakeCentralRuntime([]), config=CentralAgentConfig(repair_max_new_tokens=cap))
    assert agent._repair_budget(CentralGeneration(output_tokens=tokens), "Answer") == expected


def test_disabled_repair_keeps_risk_visible_without_returning_contamination():
    runtime = FakeCentralRuntime([CentralGeneration(content="Lê Đại Hành lập Đại Cồ Việt năm 968. [S1]")])
    result = build_agent(runtime, FakeTool("search_history", bach_dang_rows()), config=CentralAgentConfig(repair_max_generations=0)).chat(BACH_DANG_QUESTION)
    assert result["status"] == "insufficient_evidence"
    assert "unsupported_evidence_claim" in result["analysis"]["answer_quality_issues"]
    assert result["unsupported_years"] == ["968"]
    assert "Lê Đại Hành" in result["unsupported_named_claims"]
    assert len(runtime.calls) == 1
