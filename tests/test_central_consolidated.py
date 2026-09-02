"""CPU-only production regressions: no live retriever, network, model or GPU."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agents.central_analytical import annotate_evidence, coverage_report, evidence_targets
from app.agents.central_citations import check_citations, citation_display_map, expand_citations
from app.agents.central_evidence import build_evidence_packet, select_evidence, select_synthesis_evidence
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_question import analyze_central_question, plan_analytical_queries
from app.agents.config import CentralAgentConfig
from app.api.routes import _build_debug, _context_to_api, _result_sources
from app.tools.local_search import SearchHistoryInput, SearchHistoryTool
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent, generation_call
from tests.test_central_quality_hardening import BIO_QUESTION, biography_rows


TRAN = "Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?"
WAR = "Vì sao Mỹ và VNCH lại thua chiến tranh Việt Nam?"
COMPARE = "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ."
A, B = analyze_central_question(COMPARE).comparison_targets
CONFIG = CentralAgentConfig()


def row(source_id, title, text, score=0.99, **extra):
    return dict(chunk_id=source_id, title=title, text=text, reranker_score=score, **extra)


def comparison_rows(target):
    if target == A:
        return [
            row("A1", A, "Cách mạng Tháng Tám diễn ra trong bối cảnh chiến tranh thế giới. Mục tiêu của nhân dân là giành chính quyền và độc lập thông qua khởi nghĩa trên cả nước."),
            row("A2", A, "Cách mạng Tháng Tám có lực lượng nhân dân tham gia đấu tranh giành chính quyền. Kết quả thành công có ý nghĩa chính trị, mở ra một bước ngoặt lịch sử."),
            row("month", "Tháng tám", "Tháng tám là một tháng trong năm. Cách mạng Tháng Tám được kỷ niệm trong tháng này."),
            row("square", "Quảng trường Cách mạng Tháng Tám", "Quảng trường Cách mạng Tháng Tám là một địa điểm du lịch."),
            row("street", "Đường Cách Mạng Tháng Tám, Thành phố Hồ Chí Minh", "Đường Cách Mạng Tháng Tám là một con đường."),
        ]
    return [
        row("B1", "Chiến dịch Điện Biên Phủ", "Chiến dịch Điện Biên Phủ là một chiến dịch quân sự. Quân đội thực hiện mục tiêu đánh bại tập đoàn cứ điểm bằng phương pháp tiến công và chuẩn bị hậu cần.", 0.9999),
        row("B2", "Trận Điện Biên Phủ", "Trận Điện Biên Phủ kết thúc với sự đầu hàng của đối phương. Kết quả thắng lợi có ý nghĩa quân sự và tác động đến đấu tranh ngoại giao, tạo bước ngoặt lịch sử."),
        row("monument", "Tượng đài chiến thắng Điện Biên Phủ", "Tượng đài chiến thắng Điện Biên Phủ được xây để kỷ niệm chiến thắng."),
        row("city", "Điện Biên Phủ (thành phố)", "Điện Biên Phủ là thành phố phát triển du lịch."),
    ]


def comparison_tool(args):
    return comparison_rows(A if args["query"].startswith(A) else B)


GOOD_COMPARE = (
    "Cách mạng Tháng Tám có mục tiêu giành chính quyền bằng khởi nghĩa của nhân dân. [S1]\n\n"
    "Điện Biên Phủ là chiến dịch quân sự đánh bại tập đoàn cứ điểm. [S3]\n\n"
    "Điểm giống là kết quả tạo bước ngoặt lịch sử; điểm khác nằm ở phương pháp đấu tranh, "
    "vì bối cảnh và mục tiêu riêng dẫn đến ý nghĩa chính trị, quân sự khác nhau. [S2] [S4]"
)


@pytest.mark.parametrize("question,subject,event,actors,outcome", [
    (TRAN, "Nhà Trần", None, (), "suy yếu"),
    (WAR, None, "Chiến tranh Việt Nam", ("Mỹ", "Việt Nam Cộng hòa"), "thất bại"),
    ("Vì sao Cách mạng Tháng Tám thành công?", None, A, (), "thành công"),
    ("Tại sao nhà Lê suy yếu?", "Nhà Lê", None, (), "suy yếu"),
    ("Vì sao nhà Trần suy yếu?", "Nhà Trần", None, (), "suy yếu"),
    ("Nguyên nhân suy yếu của triều đại nhà Trần?", "Nhà Trần", None, (), "suy yếu"),
])
def test_analytical_targets(question, subject, event, actors, outcome):
    analysis = analyze_central_question(question)
    assert analysis.question_type == "cause"
    assert (analysis.subject, analysis.event, analysis.actors, analysis.outcome) == (subject, event, actors, outcome)
    assert analysis.telemetry()["outcome"] == outcome


def test_bounded_queries_keep_targets_independent_and_prioritize_requested_facets():
    analysis = analyze_central_question(COMPARE[:-1] + " về mục tiêu và kết quả.")
    assert analysis.comparison_targets == (A, B)
    assert analysis.facets == ("objective", "result")
    plan = plan_analytical_queries(analysis)
    assert len(plan[A]) == len(plan[B]) == 2
    assert all(B not in q for q in plan[A])
    assert all(A not in q for q in plan[B])
    assert "mục tiêu kết quả" in plan[A][1]
    assert "nguyên nhân suy yếu" in plan_analytical_queries(analyze_central_question(TRAN))["Nhà Trần"][0]
    assert "nguyên nhân thất bại" in plan_analytical_queries(analyze_central_question(WAR))["Chiến tranh Việt Nam"][0]
    assert len(plan_analytical_queries(analyze_central_question(BIO_QUESTION))[BIO_QUESTION]) == 1


def test_nguyen_cao_ky_six_to_two_one_call_numeric_display():
    rows = biography_rows() + [row("noise3", "Nguyễn Cao", "Nguyễn Cao.", 0.003), row("noise4", "Trần Cao Vân", "Trần Cao Vân.", 0.002)]
    runtime = FakeCentralRuntime([CentralGeneration(content="Nguyễn Cao Kỳ là sĩ quan không quân và chính khách. [S1]\n\nÔng giữ chức Thủ tướng và Phó Tổng thống Việt Nam Cộng hòa. [S2]")])
    result = build_agent(runtime, FakeTool("search_history", rows), config=CONFIG).chat(BIO_QUESTION)
    assert result["status"] == "ok"
    debug = result["central_debug"]
    assert (debug["retrieval_candidates_before_filter"], debug["retrieval_candidates_after_filter"]) == (6, 2)
    assert len(runtime.calls) == 1 and debug["repair_used"] is False
    assert "[1]" in result["answer"] and "[2]" in result["answer"]
    assert "hf_wikipedia" not in result["answer"]
    assert all(source_id.startswith("hf_wikipedia") for source_id in result["source_ids"])


def tran_rows():
    return [
        row("china", "Nhà Trần (Trung Quốc)", "Nhà Trần suy yếu do những nguyên nhân kinh tế và chính trị ở Trung Quốc.", 0.9999),
        row("after", "Nhà Trần", "Sau khi nhà Trần sụp đổ, Hậu Trần tiếp tục đấu tranh. Hậu Trần tổ chức lực lượng quân sự nhằm phục hưng triều đại.", 0.998),
        row("late", "Nhà Trần", "Cuối thời nhà Trần, triều đình và quan lại tham nhũng khiến chính quyền suy yếu. Ruộng đất tập trung, thuế nặng và nông dân bất bình là các nguyên nhân kinh tế xã hội.", 0.93),
    ]


def test_tran_decline_scope_temporal_selection_and_sufficiency():
    analysis = analyze_central_question(TRAN)
    filtered, debug = select_evidence(tran_rows(), analysis, CONFIG)
    assert "china" not in {r["chunk_id"] for r in filtered}
    assert debug["entity_disambiguation_filter_reasons"] == {"foreign_entity_scope": 1}
    assert next(r for r in filtered if r["chunk_id"] == "after")["chronology_downranked"]
    selected = select_synthesis_evidence(filtered, analysis, CONFIG)
    assert selected[0]["chunk_id"] == "late"
    assert coverage_report(selected, filtered, analysis, CONFIG)[0]
    assert not coverage_report([r for r in selected if r["chunk_id"] == "after"], filtered, analysis, CONFIG)[0]
    runtime = FakeCentralRuntime([CentralGeneration(content="Nhà Trần suy yếu do triều đình và quan lại tham nhũng, thuế nặng và nông dân bất bình. [S1]")])
    result = build_agent(runtime, FakeTool("search_history", tran_rows()), config=CONFIG).chat(TRAN)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert result["central_debug"]["chronology_downranked_count"] == 1


def war_rows():
    return [
        row("campaign72", "Chiến cục năm 1972", "Trong Chiến tranh Việt Nam, chiến cục năm 1972 có nhiều hoạt động quân sự. Quân đội tham gia chiến dịch tiến công ở chiến trường.", 0.9999),
        row("intervention", "Can thiệp của Hoa Kỳ", "Trong Chiến tranh Việt Nam, quân đội Mỹ can thiệp bằng quân sự và chiến lược tác chiến. Các chiến dịch quân sự gây tổn thất.", 0.999),
        row("lamson", "Lam Sơn 719", "Trong Chiến tranh Việt Nam, Lam Sơn 719 là chiến dịch quân sự. Quân đội thực hiện một cuộc tiến công quân sự trên chiến trường.", 0.997),
        row("overview", "Chiến tranh Việt Nam", "Chiến tranh Việt Nam: Mỹ và Việt Nam Cộng hòa thất bại do những vấn đề chiến lược và chính trị. Phong trào phản chiến, khó khăn viện trợ kinh tế và sức chiến đấu của đối phương góp phần vào kết quả.", 0.993),
        row("limited", "Chiến tranh cục bộ (Chiến tranh Việt Nam)", "Trong Chiến tranh Việt Nam, chiến tranh cục bộ liên quan đến chiến lược quân sự. Phong trào phản chiến trong nước và đấu tranh ngoại giao quốc tế tác động đến chính phủ.", 0.908),
    ]


def test_us_vnch_overview_and_dimension_breadth_one_generation():
    runtime = FakeCentralRuntime([CentralGeneration(content="Mỹ và Việt Nam Cộng hòa thất bại trong Chiến tranh Việt Nam do vấn đề chiến lược và chính trị, phong trào phản chiến, khó khăn viện trợ và sức chiến đấu của đối phương. [S1]")])
    result = build_agent(runtime, FakeTool("search_history", war_rows()), config=replace(CONFIG, analytical_max_sources=3)).chat(WAR)
    assert result["status"] == "ok"
    assert result["source_chunks"][0]["chunk_id"] == "overview"
    assert len(result["source_chunks"]) == 3
    debug = result["central_debug"]
    assert {"strategy", "political", "economic", "domestic", "opponent"} <= set(debug["evidence_dimensions_covered"])
    assert debug["overview_anchor_selected"] == ["overview"]
    assert len(runtime.calls) == 1 and not debug["repair_used"]
    assert _build_debug(result)["request"]["actors"] == ["Mỹ", "Việt Nam Cộng hòa"]


def test_comparison_balanced_before_character_budget_and_one_synthesis():
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD_COMPARE)])
    history = FakeTool("search_history", comparison_tool)
    result = build_agent(runtime, history, config=replace(CONFIG, observation_char_budget=1000)).chat(COMPARE)
    assert result["status"] == "ok"
    assert len(runtime.calls) == 1 and not result["central_debug"]["repair_used"]
    assert {row["chunk_id"] for row in result["source_chunks"]} == {"A1", "A2", "B1", "B2"}
    assert all(call["top_k"] == 10 for call in history.calls)
    for candidate in result["retrieval"]["candidates20"]:
        assert candidate["comparison_target"] == (A if candidate["chunk_id"] in {"A1", "A2", "month", "square", "street"} else B)
    debug = result["central_debug"]
    assert set(debug["target_rankings"]) == {A, B}
    assert all(item["selected_count"] == item["strong_evidence_count"] == 2 for item in debug["comparison_balance"].values())
    assert debug["entity_disambiguation_filtered_count"] == 10  # five noise rows, two variants
    prompt = runtime.calls[0]["messages"][-1]["content"]
    assert f"TARGET A — {A}" in prompt and f"TARGET B — {B}" in prompt
    trace = _build_debug(result)
    assert trace["retrieval"]["comparison_balance"][B]["adequate"]
    assert trace["retrieval"]["target_rankings"][B][0]["comparison_target"] == B
    for item in result["source_chunks"]:
        assert _context_to_api(item).model_dump()["display_index"] == item["display_index"]
    sources = _result_sources(SimpleNamespace(chunk_by_id={}), result)
    assert sources[2].comparison_target == B and sources[2].source_id == result["source_chunks"][2]["chunk_id"]


@pytest.mark.parametrize("bad_b", [[], [row("short", "Chiến dịch Điện Biên Phủ", "Chiến dịch Điện Biên Phủ.")]])
def test_one_sided_or_short_comparison_never_enters_synthesis(bad_b):
    history = FakeTool("search_history", lambda args: comparison_rows(A) if args["query"].startswith(A) else bad_b)
    runtime = FakeCentralRuntime([])
    result = build_agent(runtime, history, config=CONFIG).chat(COMPARE)
    assert result["status"] == "insufficient_evidence" and not runtime.calls
    assert not result["central_debug"]["comparison_balance"][B]["adequate"]
    assert "synthesis" not in result["central_debug"]["phase_trace"]


def test_wikipedia_fallback_ranks_event_fetches_and_preserves_target_without_actions():
    history = FakeTool("search_history", lambda args: comparison_rows(A) if args["query"].startswith(A) else [])
    search = FakeTool("search_wikipedia", [
        row("monument", "Khu di tích Điện Biên Phủ", "Địa điểm."),
        row("wikiB", "Chiến dịch Điện Biên Phủ", "Chiến dịch quân sự.", metadata={"page_id": 123, "language": "vi"}),
        row("city", "Điện Biên Phủ (thành phố)", "Thành phố."),
    ])
    fetch = FakeTool("fetch_wikipedia_page", row("wikiB", "Chiến dịch Điện Biên Phủ", comparison_rows(B)[0]["text"] + comparison_rows(B)[1]["text"]))
    answer = GOOD_COMPARE.replace("[S4]", "[S3]")
    runtime = FakeCentralRuntime([CentralGeneration(content=answer)])
    result = build_agent(runtime, history, search, fetch, config=CONFIG).chat(COMPARE)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert len(search.calls) == len(fetch.calls) == 1
    assert search.calls[0]["query"] == B
    assert fetch.calls[0]["page_id_or_title"] == "123"
    assert next(s for s in result["source_chunks"] if s["chunk_id"] == "wikiB")["comparison_target"] == B
    assert result["central_debug"]["comparison_balance"][B]["adequate"]
    assert not any(t["error"] == "duplicate_tool_call_prevented" for t in result["central_debug"]["tools"])


def test_search_snippets_alone_never_pass_and_no_second_search_generation():
    search = FakeTool("search_wikipedia", [row("wikiB", "Chiến dịch Điện Biên Phủ", comparison_rows(B)[0]["text"])])
    runtime = FakeCentralRuntime([generation_call("search_wikipedia", {"query": "chiến dịch Điện Biên Phủ"})])
    result = build_agent(runtime, FakeTool("search_history", []), search, config=CONFIG).chat("Một câu hỏi lịch sử")
    assert result["status"] == "insufficient_evidence" and len(runtime.calls) == 1
    assert not result["source_ids"] and not result["source_chunks"]
    assert result["answer_provenance"]["central_external_results_count"] == 0


def comparison_packet():
    rows = [{**r, "comparison_target": t} for t in (A, B) for r in comparison_rows(t)[:2]]
    return build_evidence_packet(rows)


@pytest.mark.parametrize("answer,mismatch", [
    (f"{A} có lực lượng nhân dân. [S1]", False),
    ("Điện Biên Phủ là chiến dịch quân sự. [S1]", True),
    ("Điện Biên Phủ là chiến dịch quân sự. [S3]", False),
    ("## Điện Biên Phủ\n\nQuân đội tiến công. [S1]", True),
    ("**Điện Biên Phủ**\n\nQuân đội tiến công. [S3]", False),
    (f"{A} và Điện Biên Phủ có kết quả khác nhau. [S1]", True),
    (f"{A} và Điện Biên Phủ có kết quả khác nhau. [S1] [S3]", False),
    ("| Điện Biên Phủ | Tiến công. [S1] |", True),
])
def test_target_aware_paragraph_section_and_table_citations(answer, mismatch):
    assert bool(check_citations(answer, comparison_packet()).target_mismatches) is mismatch


def test_table_column_targets_require_corresponding_citations():
    table = f"| Phương diện | {A} | Điện Biên Phủ |\n|---|---|---|\n| Phương pháp | Khởi nghĩa [S1] | Chiến dịch [S1] |"
    assert check_citations(table, comparison_packet()).target_mismatches == [B]
    correct = table.replace("Chiến dịch [S1]", "Chiến dịch [S3]")
    assert not check_citations(correct, comparison_packet()).target_mismatches
    assert not check_citations(correct, comparison_packet()).uncited_paragraphs


def test_duplicate_text_does_not_inflate_strong_evidence_count():
    analysis = analyze_central_question(COMPARE)
    rows = [{**comparison_rows(t)[0], "chunk_id": f"{t}_{i}", "comparison_target": t} for t in (A, B) for i in (1, 2)]
    selected = select_synthesis_evidence(rows, analysis, CONFIG)
    adequate, debug = coverage_report(selected, rows, analysis, replace(CONFIG, comparison_min_strong_sources=2))
    assert not adequate
    assert debug["comparison_balance"][A]["strong_evidence_count"] == 1


def test_shared_source_keeps_both_origins_but_truncation_cannot_transfer_support():
    analysis = analyze_central_question(COMPARE)
    shared = row("shared", A, comparison_rows(A)[0]["text"] + " " + comparison_rows(B)[0]["text"], comparison_target=A, comparison_targets=[A, B])
    selected = select_synthesis_evidence([shared], analysis, CONFIG)
    assert evidence_targets(selected[0]) == [A, B]
    assert coverage_report(selected, [shared], analysis, CONFIG)[0]
    shared["text"] = comparison_rows(A)[0]["text"] + " Diễn giải bối cảnh." * 200 + comparison_rows(B)[0]["text"]
    selected = select_synthesis_evidence([shared], analysis, replace(CONFIG, synthesis_char_budget=1000))
    assert evidence_targets(selected[0]) == [A]
    assert not coverage_report(selected, [shared], analysis, CONFIG)[0]


def test_shared_wikipedia_page_is_fetched_once_without_losing_second_target():
    text = comparison_rows(A)[0]["text"] + " " + comparison_rows(B)[0]["text"]
    search = FakeTool("search_wikipedia", [row("shared", "Lịch sử Việt Nam", text, metadata={"page_id": 777})])
    fetch = FakeTool("fetch_wikipedia_page", row("shared", "Lịch sử Việt Nam", text))
    answer = f"{A} và Điện Biên Phủ có điểm giống và điểm khác về mục tiêu vì bối cảnh đấu tranh; ý nghĩa mỗi sự kiện cần xem riêng. [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=answer)])
    result = build_agent(runtime, FakeTool("search_history", []), search, fetch, config=CONFIG).chat(COMPARE)
    assert len(fetch.calls) == 1 and len(runtime.calls) == 1
    assert result["status"] == "ok"
    assert result["source_chunks"][0]["comparison_targets"] == [A, B]
    assert result["central_debug"]["comparison_balance"][B]["adequate"]


def test_failed_target_citation_repair_stays_bounded_and_returns_no_comparison():
    bad = GOOD_COMPARE.replace("[S3]", "[S1]")
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=bad)])
    result = build_agent(runtime, FakeTool("search_history", comparison_tool), config=CONFIG).chat(COMPARE)
    assert result["status"] == "insufficient_evidence" and len(runtime.calls) == 2
    assert result["source_ids"] == []


def test_numeric_brackets_are_never_resolved_and_display_indices_do_not_follow_citation_order():
    packet = comparison_packet()
    answer = "Năm [938], [1945], [1954]. [S3] [S1] [S3]"
    checked = check_citations(answer, packet)
    assert checked.source_ids == ["B1", "A1"]
    visible = expand_citations(checked.answer, packet)
    assert visible == "Năm [938], [1945], [1954]. [3] [1] [3]"
    assert expand_citations(visible, packet) == visible
    assert citation_display_map(packet)["S3"]["display_index"] == 3
    numeric_packet = build_evidence_packet([row("1954", "Nguồn", "Nội dung")])
    assert not check_citations("[1954]", numeric_packet).source_ids


def test_viewpoint_annotation_attribution_and_single_repair():
    source = row("quote", "Chiến tranh Việt Nam", 'Một nguồn tuyên bố: “chúng ta nhất định thắng”, gọi đối phương là “lũ tay sai”. Đây là quan điểm chính trị trong chiến tranh, không phải nhận định trung lập.')
    annotated = annotate_evidence(source, analyze_central_question(WAR))
    assert annotated["viewpoint_sensitive"]
    packet = build_evidence_packet([annotated])
    assert check_citations("Chúng ta nhất định thắng. [S1]", packet).unattributed_viewpoints == 1
    assert not check_citations('Theo nguồn, “chúng ta nhất định thắng” là lời tuyên bố. [S1]', packet).unattributed_viewpoints
    runtime = FakeCentralRuntime([CentralGeneration(content=GOOD_COMPARE.replace("[S3]", "[S1]")), CentralGeneration(content=GOOD_COMPARE)])
    result = build_agent(runtime, FakeTool("search_history", comparison_tool), config=CONFIG).chat(COMPARE)
    assert result["status"] == "ok" and len(runtime.calls) == 2
    assert result["central_debug"]["repair_reason"] == "comparison_citation_target_mismatch"
    assert runtime.calls[1]["stage"] == "quality_repair"
    assert "khẳng định về A" in runtime.calls[1]["messages"][-1]["content"]


def test_stricter_comparison_minimum_and_small_budget_cannot_pass_with_one_source():
    analysis = analyze_central_question(COMPARE)
    rows = [{**comparison_rows(t)[0], "comparison_target": t} for t in (A, B)]
    selected = select_synthesis_evidence(rows, analysis, CONFIG)
    assert coverage_report(selected, rows, analysis, CONFIG)[0]
    assert not coverage_report(selected, rows, analysis, replace(CONFIG, comparison_min_strong_sources=2))[0]


def test_local_candidate_pool_bypasses_final_diversity_only_when_requested():
    class Retriever:
        def retrieve(self, question, final_k):
            return {"candidates20": war_rows(), "final_context": war_rows()[:3]}
    tool = SearchHistoryTool(Retriever())
    assert "overview" not in {r["chunk_id"] for r in tool.run(SearchHistoryInput(query=WAR))}
    assert "overview" in {r["chunk_id"] for r in tool.run(SearchHistoryInput(query=WAR, candidate_pool=True, top_k=10))}
