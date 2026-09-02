"""Deterministic, CPU/fake regressions for elliptical agreement comparisons."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agents.central_analytical import annotate_evidence, coverage_report
from app.agents.central_evidence import select_synthesis_evidence
from app.agents.central_model_runtime import CentralGeneration
from app.agents.central_question import analyze_central_question, plan_analytical_queries
from app.agents.central_targets import EntityTitleIndex, resolve_comparison_targets
from app.agents.config import CentralAgentConfig
from app.tools.local_search import SearchHistoryTool
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent

QUESTION = "So sánh Hiệp định Genève và Paris"
A, B = "Hiệp định Genève", "Hiệp định Paris"
TITLES = [A + " 1954", B + " 1973", "Nhà Trần", "Nhà Hồ", "Chiến dịch Điện Biên Phủ", "Chiến dịch Hồ Chí Minh"]
CONFIG = CentralAgentConfig()


def analysis(question=QUESTION, titles=TITLES):
    return resolve_comparison_targets(analyze_central_question(question), EntityTitleIndex(titles).resolve)


def rows(target):
    year = "1954" if target == A else "1973"
    return [{"chunk_id": target + str(i), "title": target + " " + year, "text": text,
             "comparison_target": target, "comparison_targets": [target]} for i, text in enumerate([
        f"{target} năm {year} diễn ra trong bối cảnh đấu tranh ngoại giao. Mục tiêu là chấm dứt chiến tranh, chính phủ và lực lượng nhân dân tham gia bằng phương pháp đàm phán.",
        f"{target} năm {year} có nội dung chính trị và quân sự. Kết quả có ý nghĩa quốc tế, tác động đến đấu tranh của nhân dân và tạo bước ngoặt lịch sử.",
    ])]


ANSWER = ("Hiệp định Genève năm 1954 có mục tiêu chấm dứt chiến tranh trong bối cảnh đấu tranh ngoại giao. [S1]\n\n"
          "Hiệp định Paris năm 1973 có mục tiêu chấm dứt chiến tranh bằng phương pháp đàm phán. [S3]\n\n"
          "Điểm giống nhau: cả hai có ý nghĩa chính trị, quân sự và quốc tế. [S2] [S4]\n\n"
          "Điểm khác nhau: bối cảnh của hai hiệp định khác nhau, vì vậy kết quả và tác động cần đặt trong tiến trình riêng. [S1] [S3]")


def history_tool(result=None):
    tool = FakeTool("search_history", result or (lambda args: rows(A if "Genève" in args["query"] else B)))
    tool.resolve_entity_title = EntityTitleIndex(TITLES).resolve
    return tool


@pytest.mark.parametrize("question,expected", [
    (QUESTION, (A, B)),
    ("So sánh nhà Trần và Hồ", ("nhà Trần", "Nhà Hồ")),
    ("So sánh chiến dịch Điện Biên Phủ và Hồ Chí Minh", ("chiến dịch Điện Biên Phủ", "Chiến dịch Hồ Chí Minh")),
    ("So sánh Nguyễn Huệ và Gia Long", ("Nguyễn Huệ", "Gia Long")),
    ("So sánh Paris và London", ("Paris", "London")),
])
def test_shared_head_and_no_over_propagation(question, expected):
    assert analysis(question).comparison_targets == expected


def test_campaign_requires_corpus_confirmation_and_ambiguous_dates_are_not_guessed():
    value = analysis("So sánh chiến dịch Điện Biên Phủ và Hồ Chí Minh", [])
    assert value.comparison_targets[1] == "Hồ Chí Minh"
    assert EntityTitleIndex(["Hiệp định Ví dụ 1954", "Hiệp định Ví dụ 1973"]).resolve("Hiệp định Ví dụ", "agreement") is None


def test_corpus_title_index_is_cached_and_rebuilt_only_on_snapshot_replacement():
    service = SimpleNamespace(chunks=[{"title": title} for title in TITLES])
    tool = SearchHistoryTool(SimpleNamespace(service=service))
    assert tool.resolve_entity_title(B, "agreement") == B + " 1973"
    index = tool._entity_title_index
    assert tool.resolve_entity_title(A, "agreement") == A + " 1954"
    assert tool._entity_title_index is index
    service.chunks = [{"title": "Nhà Hồ"}]
    assert tool.resolve_entity_title(B, "agreement") is None


def test_exact_production_query_two_local_calls_one_generation_no_repair():
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    history = history_tool()
    wiki = FakeTool("search_wikipedia")
    result = build_agent(runtime, history, wiki, config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    assert result["status"] == "ok", debug
    assert debug["comparison_targets_raw"] == [A, "Paris"]
    assert debug["comparison_targets_normalized"] == [A, B]
    assert debug["comparison_canonical_targets"] == [A + " 1954", B + " 1973"]
    assert debug["comparison_target_entity_types"] == ["agreement", "agreement"]
    assert all(event["status"] == "corpus_confirmed" for event in debug["target_resolution_events"])
    assert [call["query"] for call in history.calls] == [A + " 1954", B + " 1973"]
    assert not wiki.calls and len(runtime.calls) == 1
    assert not debug["repair_used"] and not debug["full_quality_repair_used"]
    assert len(debug["retrieval_queries_skipped"]) == 2
    assert all(value["adequate"] for value in debug["comparison_balance"].values())
    assert debug["final_failure_reason"] is None


def test_city_cannot_satisfy_agreement_even_with_incidental_mentions_and_dimensions():
    city = {"chunk_id": "city", "title": "Paris", "text":
        "Paris là thủ đô của Pháp. Kinh tế và chính trị quốc tế có ý nghĩa và mục tiêu riêng. Hiệp định Paris 1973 được nhắc đến ở đây.",
        "comparison_target": B, "comparison_targets": [B]}
    marked = annotate_evidence(city, analysis(), B)
    assert not marked["entity_type_consistent"] and not marked["target_consistent"]
    sufficient, debug = coverage_report(rows(A) + [city], rows(A) + [city], analysis(), CONFIG)
    assert not sufficient and not debug["comparison_balance"][B]["adequate"]
    assert "city" not in {r["chunk_id"] for r in select_synthesis_evidence(rows(A) + [city], analysis(), CONFIG)}


def test_wrong_canonical_year_does_not_count():
    wrong = {**rows(B)[0], "title": B + " 1999", "text": rows(B)[0]["text"].replace("1973", "1999")}
    assert not annotate_evidence(wrong, analysis(), B)["canonical_target_consistent"]


def test_wikipedia_prefers_agreement_over_city_club_show_and_commune():
    history = history_tool(lambda args: rows(A) if "Genève" in args["query"] else [])
    search = FakeTool("search_wikipedia", [{"title": name, "page_id": page, "text": text} for name, page, text in [
        ("Paris", 2584, "Paris là thủ đô của Pháp. Hiệp định Paris 1973 được ký tại đây."),
        ("Paris Saint-Germain F.C.", 2, "Paris Saint-Germain là câu lạc bộ bóng đá."),
        ("Paris By Night", 3, "Paris By Night là chương trình ca nhạc."),
        ("Công xã Paris", 4, "Công xã Paris có ý nghĩa chính trị và quốc tế."),
        (B + " 1973", 5, rows(B)[0]["text"]),
    ]])
    fetch = FakeTool("fetch_wikipedia_page", rows(B))
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, history, search, fetch, config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok", result["central_debug"]
    assert fetch.calls[0]["page_id_or_title"] == "5"
    assert all(call["query"] != "Paris" for tool in (history, search) for call in tool.calls)


def test_one_target_only_returns_qualified_limitation_without_generation():
    runtime = FakeCentralRuntime([])
    result = build_agent(runtime, history_tool(lambda args: rows(A) if "Genève" in args["query"] else []), config=CONFIG).chat(QUESTION)
    assert result["status"] == "insufficient_evidence" and not runtime.calls
    assert A in result["answer"] and B in result["answer"]


def test_repair_receives_all_issues_and_revalidates_from_scratch():
    broken = f"{A} năm 1954 và {B} năm 1973 có bối cảnh khác nhau."
    runtime = FakeCentralRuntime([CentralGeneration(content=broken), CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, history_tool(), config=CONFIG).chat(QUESTION)
    debug = result["central_debug"]
    expected = {"uncited_factual_paragraphs", "comparison_similarity_missing", "historical_significance_missing"}
    assert expected <= set(debug["repair_reasons"])
    prompt = runtime.calls[-1]["messages"][-1]["content"]
    assert all(reason in prompt for reason in expected)
    assert len(runtime.calls) == 2 and debug["full_quality_repair_used"]
    assert result["status"] == "ok" and debug["answer_quality_issues"] == []
    assert debug["viewpoint_attribution_issues"] == [] and debug["uncited_factual_paragraphs"] == 0


def test_persistent_missing_significance_fails_closed():
    answer = ANSWER.replace("ý nghĩa", "nội dung").replace("tác động", "kết quả")
    result = build_agent(FakeCentralRuntime([CentralGeneration(content=answer)]), history_tool(),
                         config=replace(CONFIG, repair_max_generations=0)).chat(QUESTION)
    assert result["status"] == "answer_validation_failed"
    assert "historical_significance_missing" in result["central_debug"]["answer_quality_issues"]
