"""Synthetic administrative policies only; no live/current policy assertions."""
from dataclasses import replace
from datetime import date

import pytest

from app.agents.central.administration import administrative_question, administrative_levels
from app.agents.central.analytical import coverage_report
from app.agents.central.compaction import compact_history
from app.agents.central.evidence import select_synthesis_evidence
from app.agents.central.model_runtime import CentralGeneration
from app.agents.central.question import analyze_central_question, plan_analytical_queries
from app.agents.central.config import CentralAgentConfig
from tests.test_central_agent import FakeCentralRuntime, FakeTool, build_agent

QUESTION = "Vì sao Việt Nam lại bắt đầu sáp nhập tỉnh năm 2025?"
CONFIG = CentralAgentConfig()


@pytest.fixture(autouse=True)
def fixed_policy_clock(monkeypatch):
    class FixtureDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 1)
    monkeypatch.setattr("app.agents.central.administration.date", FixtureDate)


def row(key, text, title="Sắp xếp đơn vị hành chính", updated="2025-07-01", **extra):
    return {"chunk_id": key, "title": title, "text": text, "metadata": {"updated_at": updated}, **extra}


A = row("A", "Việt Nam sắp xếp đơn vị hành chính cấp tỉnh năm 2025 nhằm tinh gọn tổ chức bộ máy, nâng cao hiệu lực quản lý và sử dụng nguồn lực hiệu quả. Chủ trương có mục tiêu quản lý phù hợp quy mô mới.")
B = row("B", "Việt Nam sáp nhập tỉnh năm 2025: tỉnh A nhập tỉnh B và tỉnh C nhập tỉnh D. Các đơn vị hành chính được tổ chức lại theo danh sách triển khai của cơ quan quản lý.")
C = row("C", "Sắp xếp đơn vị hành chính cấp xã năm 2025: xã A nhập xã B, phường C nhập phường D, thị trấn E nhập thị trấn F. Danh sách quy định chi tiết từng đơn vị.")
D = row("D", "Sắp xếp đơn vị hành chính cấp huyện năm 2025: huyện A nhập huyện B, quận C nhập quận D. Danh sách quy định chi tiết từng đơn vị để thực hiện chính sách.")
E = row("E", "Chương trình sắp xếp đơn vị hành chính giai đoạn 2023–2030 có mục tiêu tinh gọn tổ chức bộ máy. Sắp xếp cấp xã và cấp huyện nhằm nâng cao hiệu quả quản lý.", updated="2023-08-01")
ANSWER = "Việt Nam sắp xếp đơn vị hành chính cấp tỉnh năm 2025 nhằm tinh gọn tổ chức bộ máy, nâng cao hiệu lực quản lý và sử dụng nguồn lực hiệu quả. Nguồn xác lập mục tiêu của đợt này, chưa xác lập đây là lần đầu có sắp xếp hành chính. [S1]"


def sufficient(rows):
    analysis = analyze_central_question(QUESTION)
    selected = select_synthesis_evidence(rows, analysis, CONFIG)
    return selected, coverage_report(selected, rows, analysis, CONFIG)


@pytest.mark.parametrize("phrase,level", [
    ("tỉnh", "province"), ("thành phố trực thuộc trung ương", "province"), ("cấp tỉnh", "province"),
    ("huyện", "district"), ("quận", "district"), ("thị xã", "district"), ("thành phố thuộc tỉnh", "district"),
    ("xã", "commune"), ("phường", "commune"), ("thị trấn", "commune"), ("cấp xã", "commune"),
])
def test_explicit_administrative_levels(phrase, level):
    assert administrative_levels(phrase) == [level]
    assert administrative_question(f"Vì sao sáp nhập {phrase} năm 2025?", today=date(2026, 9, 1))["administrative_level"] == level


def test_exact_question_semantics_and_no_unrelated_history():
    analysis = analyze_central_question(QUESTION)
    assert analysis.question_type == "cause" and analysis.administrative_level == "province"
    assert analysis.subject == "Việt Nam" and analysis.time_scope == (2025,)
    assert analysis.freshness_required and analysis.premise_requires_validation
    queries = next(iter(plan_analytical_queries(analysis).values()))
    assert len(queries) == 2 and all("cấp tỉnh" in q and "2025" in q for q in queries)
    assert compact_history(QUESTION, [{"role": "user", "content": "Nguyễn Cao Kỳ là ai?"}], max_messages=4, char_budget=2400) == []


def test_province_cause_excerpt_outranks_lists_and_stale_program():
    selected, (ok, debug) = sufficient([C, D, E, B, A])
    assert ok and selected[0]["chunk_id"] == "A"
    assert debug["cause_evidence_count"] == 1
    assert debug["administrative_level_match_count"] == 2
    assert debug["administrative_level_mismatch_count"] == 3
    assert debug["evidence_administrative_levels"] == ["commune", "district", "province"]


@pytest.mark.parametrize("rows", [[C, D, E], [C, C, C, C], [B], [dict(A, metadata={})], [dict(A, metadata={"updated_at": "2023-01-01"})]])
def test_lower_levels_lists_or_unverified_freshness_are_insufficient(rows):
    assert sufficient(rows)[1][0] is False


def test_fresh_local_core_keeps_one_retrieval_one_synthesis_and_no_repair():
    history = FakeTool("search_history", [C, D, E, B, A])
    wiki = FakeTool("search_wikipedia")
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, history, wiki, config=CONFIG).chat(QUESTION, history=[{"role": "user", "content": "Vì sao nhà Trần suy yếu?"}])
    assert result["status"] == "ok", result["central_debug"]
    assert len(runtime.calls) == len(history.calls) == 1 and not wiki.calls
    debug = result["central_debug"]
    assert not debug["repair_used"] and not debug["current_source_fallback_used"]
    assert debug["history_input_turns"] == 0 and debug["freshness_required"]


def test_stale_local_tries_wikipedia_and_fetches_current_provincial_cause():
    history = FakeTool("search_history", [C, D, E])
    search = FakeTool("search_wikipedia", [{**A, "page_id": 123}])
    fetch = FakeTool("fetch_wikipedia_page", [{**A, "metadata": {}}])
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, history, search, fetch, config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok", result["central_debug"]
    assert len(search.calls) == len(fetch.calls) == len(runtime.calls) == 1
    assert result["central_debug"]["current_source_fallback_used"]
    assert not result["central_debug"]["repair_used"]


def test_unavailable_current_evidence_returns_limitation_without_generation_or_web():
    runtime = FakeCentralRuntime([])
    wiki = FakeTool("search_wikipedia", [C])
    fetch = FakeTool("fetch_wikipedia_page")
    web = FakeTool("search_web", [A])
    result = build_agent(runtime, FakeTool("search_history", [C, D, E]), wiki, fetch, web, config=CONFIG).chat(QUESTION)
    assert result["status"] == "insufficient_evidence" and not runtime.calls
    assert wiki.calls and not web.calls and not fetch.calls
    assert "cấp tỉnh" in result["answer"] and "chưa đủ nguồn cập nhật" in result["answer"]
    assert result["central_debug"]["evidence_sufficient"] is False


def test_usable_web_is_only_tried_after_wikipedia():
    web = FakeTool("search_web", [A])
    wiki = FakeTool("search_wikipedia")
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, FakeTool("search_history", [C]), wiki, FakeTool("fetch_wikipedia_page"), web,
                         config=replace(CONFIG, web_search_provider="tavily")).chat(QUESTION)
    assert result["status"] == "ok"
    trace = [r["name"] for r in result["central_debug"]["tool_trace"]] if "tool_trace" in result["central_debug"] else []
    assert wiki.calls and web.calls and len(runtime.calls) == 1
    if trace:
        assert trace.index("search_wikipedia") < trace.index("search_web")


@pytest.mark.parametrize("question", ["Chiến thắng Bạch Đằng năm 938", "Vì sao Cách mạng Tháng Tám năm 1945 thành công?", "Vì sao nhà Trần suy yếu?"])
def test_stable_history_keeps_original_path(question):
    analysis = analyze_central_question(question)
    assert not analysis.freshness_required and analysis.administrative_level is None


def test_contemporary_wording_requires_policy_context_not_every_year():
    for cue in ("hiện nay", "gần đây", "mới đây", "đang", "hiện tại", "bây giờ"):
        assert administrative_question(f"Vì sao {cue} sáp nhập tỉnh?", today=date(2026, 9, 1))["freshness_required"]
    assert not administrative_question("Vì sao sáp nhập tỉnh năm 1945?", today=date(2026, 9, 1))["freshness_required"]
    assert administrative_levels("hiệu quả, tình hình xã hội") == []
    assert not administrative_question("Vì sao Đảng chủ trương sáp nhập tỉnh năm 1945?", today=date(2026, 9, 1))["freshness_required"]


def test_a_starting_year_is_not_inferred_from_objectives_and_is_rechecked_after_repair():
    bad = "Việt Nam lần đầu bắt đầu sáp nhập tỉnh năm 2025 nhằm tinh gọn tổ chức bộ máy và nâng cao hiệu lực quản lý. [S1]"
    runtime = FakeCentralRuntime([CentralGeneration(content=bad), CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, FakeTool("search_history", [A]), config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok", result["central_debug"]
    assert "unsupported_administrative_premise" in result["central_debug"]["repair_reasons"]
    assert result["central_debug"]["premise_validation_status"] == "evidence_qualified"
    assert result["central_debug"]["answer_quality_issues"] == []


def test_distinguishes_earlier_program_from_specific_phase_only_when_source_says_so():
    overview = row("phase", "Việt Nam bắt đầu chương trình sắp xếp đơn vị hành chính cấp huyện năm 2023. "
                   + A["text"])
    answer = "Chương trình sắp xếp cấp huyện bắt đầu năm 2023. Đợt sắp xếp cấp tỉnh năm 2025 có mục tiêu tinh gọn tổ chức bộ máy và nâng cao hiệu lực quản lý. [S1]"
    result = build_agent(FakeCentralRuntime([CentralGeneration(content=answer)]), FakeTool("search_history", [overview]), config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok", result["central_debug"]


def test_relevant_excerpt_must_retain_current_level_and_causal_material():
    crowded = row("crowded", C["text"] * 12 + " " + A["text"] + " " + D["text"] * 12)
    value = analyze_central_question(QUESTION)
    selected = select_synthesis_evidence([crowded], value, replace(CONFIG, evidence_excerpt_chars=600))
    assert A["text"].split(".")[0] in selected[0]["text"]
    assert coverage_report(selected, [crowded], value, CONFIG)[0]


def test_old_objectives_cannot_be_borrowed_for_a_recent_implementation_list():
    stale_causes = row("mixed", "Việt Nam sắp xếp đơn vị hành chính cấp tỉnh năm 2023 nhằm tinh gọn bộ máy và nâng cao hiệu lực quản lý. " + B["text"])
    assert not sufficient([stale_causes])[1][0]


def test_adjacent_objective_inherits_explicit_level_and_phase_context():
    overview = row("adjacent", "Việt Nam sắp xếp đơn vị hành chính cấp tỉnh năm 2025. Mục tiêu là tinh gọn tổ chức bộ máy, nâng cao hiệu lực quản lý và sử dụng nguồn lực hiệu quả.")
    assert sufficient([overview])[1][0]


def test_wikipedia_title_can_route_fetch_but_cannot_establish_core_evidence():
    hit = {"title": "Sắp xếp đơn vị hành chính cấp tỉnh Việt Nam 2025", "text": "Một chủ trương được triển khai trong năm 2025.", "page_id": 321}
    search = FakeTool("search_wikipedia", [hit])
    fetch = FakeTool("fetch_wikipedia_page", [{**A, "metadata": {}}])
    runtime = FakeCentralRuntime([CentralGeneration(content=ANSWER)])
    result = build_agent(runtime, FakeTool("search_history", [C]), search, fetch, config=CONFIG).chat(QUESTION)
    assert result["status"] == "ok" and len(runtime.calls) == 1
    assert fetch.calls[0]["page_id_or_title"] == "321"
    assert {r["chunk_id"] for r in result["source_chunks"]} == {"A"}
    # The same title with a lower-level body is insufficient after actual fetch.
    result = build_agent(FakeCentralRuntime([]), FakeTool("search_history", [C]), FakeTool("search_wikipedia", [hit]),
                         FakeTool("fetch_wikipedia_page", [{**C, "title": hit["title"]}]), config=CONFIG).chat(QUESTION)
    assert result["status"] == "insufficient_evidence"
