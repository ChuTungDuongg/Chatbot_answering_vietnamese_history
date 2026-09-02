from __future__ import annotations

from types import SimpleNamespace

from app.agents.common.comparison import (
    comparison_dimension_coverage,
    comparison_target_relevance,
)
from app.agents.evidence.agent import _best_candidate_claims, _claim_noise_reason
from app.agents.history_answerer.agent import HistoryAnswererAgent, _deep_answer_quality_issues
from app.api.routes import _build_debug, _failure_debug_trace
from app.chat.store import ConversationStore
from app.schemas import ChatRequest
from app.tools.local_search import SearchHistoryInput, SearchHistoryTool


QUESTION = "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ."


class FakeTargetRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, query, final_k):
        self.queries.append(query)
        if "Cách mạng Tháng Tám" in query:
            rows = [
                {
                    "chunk_id": "cmt8",
                    "title": "Cách mạng Tháng Tám",
                    "text": "Cách mạng Tháng Tám giành chính quyền và mở ra nền độc lập.",
                    "final_retrieval_score": 0.8,
                },
                {
                    "chunk_id": "30-4",
                    "title": "Sự kiện 30 tháng 4 năm 1975",
                    "text": "Sự kiện này có nhắc đến Cách mạng Tháng Tám.",
                    "final_retrieval_score": 0.99,
                },
            ]
        else:
            rows = [
                {
                    "chunk_id": "dbp",
                    "title": "Chiến dịch Điện Biên Phủ",
                    "text": "Chiến dịch Điện Biên Phủ là một thắng lợi quân sự quan trọng.",
                    "final_retrieval_score": 0.78,
                },
                {
                    "chunk_id": "khe-sanh",
                    "title": "Khe Sanh",
                    "text": "Một số người gọi Khe Sanh là Điện Biên Phủ thứ hai.",
                    "final_retrieval_score": 1.0,
                },
            ]
        return {"final_context": rows[:final_k]}


class FakeHistoryRuntime:
    def __init__(self, outputs):
        self.outputs = list(outputs if isinstance(outputs, list) else [outputs])
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def _compare_contexts():
    return [
        {
            "chunk_id": "cmt8",
            "title": "Cách mạng Tháng Tám",
            "text": "Cách mạng Tháng Tám diễn ra trong bối cảnh Nhật đầu hàng, giành chính quyền và mở ra nền độc lập.",
            "claims": [
                "Cách mạng Tháng Tám diễn ra trong bối cảnh Nhật đầu hàng.",
                "Cách mạng Tháng Tám giành chính quyền và mở ra nền độc lập.",
            ],
            "comparison_target": "target_a",
        },
        {
            "chunk_id": "dbp",
            "title": "Chiến dịch Điện Biên Phủ",
            "text": "Điện Biên Phủ diễn ra trong bối cảnh kháng chiến chống Pháp, giành thắng lợi và góp phần chấm dứt chiến tranh.",
            "claims": [
                "Điện Biên Phủ diễn ra trong bối cảnh kháng chiến chống Pháp.",
                "Điện Biên Phủ giành thắng lợi và góp phần chấm dứt chiến tranh.",
            ],
            "comparison_target": "target_b",
        },
    ]


def test_fake_retriever_uses_two_deterministic_paths_and_balances_exact_targets():
    retriever = FakeTargetRetriever()
    rows = SearchHistoryTool(retriever).run(SearchHistoryInput(query=QUESTION, top_k=4))

    assert len(retriever.queries) == 2
    assert retriever.queries[0].startswith("Cách mạng Tháng Tám ")
    assert retriever.queries[1].startswith("chiến thắng Điện Biên Phủ ")
    assert {row["chunk_id"] for row in rows} >= {"cmt8", "dbp"}
    assert rows[0]["target_specific_queries"]["strategy"].startswith("target_entity")


def test_direct_target_page_outranks_incidental_reference():
    direct = comparison_target_relevance(
        "chiến thắng Điện Biên Phủ",
        {"title": "Chiến dịch Điện Biên Phủ", "text": "Điện Biên Phủ giành thắng lợi."},
    )
    incidental = comparison_target_relevance(
        "chiến thắng Điện Biên Phủ",
        {"title": "Khe Sanh", "text": "Điện Biên Phủ thứ hai."},
    )
    assert direct["score"] > incidental["score"]
    assert incidental["incidental_penalty"] > 0


def test_comparison_matrix_reports_only_two_sided_dimensions():
    contexts = _compare_contexts()
    contexts[0]["claims"].append("Cách mạng Tháng Tám tạo ra hệ quả kinh tế riêng.")
    coverage = comparison_dimension_coverage(QUESTION, contexts)

    assert {"context", "result"} <= set(coverage["two_sided_dimensions"])
    assert "consequence" in coverage["one_sided_dimensions"]["target_a"]
    assert "consequence" not in coverage["two_sided_dimensions"]


def test_navigation_tag_claim_is_dropped_but_explanatory_claim_survives():
    noisy = "Việt Minh Khởi nghĩa Việt Nam Phong trào độc lập Việt Nam Việt Nam năm 1945 Cách mạng thế kỷ 20"
    normal = "Cách mạng Tháng Tám năm 1945 đã giành chính quyền và mở ra nền độc lập."
    candidate = SimpleNamespace(title="Cách mạng Tháng Tám", text=f"{noisy}. {normal}")

    assert _claim_noise_reason(noisy) == "entity_or_tag_enumeration"
    claims = _best_candidate_claims(QUESTION, candidate, compare_target="Cách mạng Tháng Tám")
    assert noisy not in claims
    assert normal in claims


def test_original_cited_source_text_supports_year_omitted_from_compressed_claim():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [dbp]\n\nTrả lời:\nChiến dịch Điện Biên Phủ năm 1954 giành thắng lợi."
    )
    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="Chiến dịch Điện Biên Phủ diễn ra như thế nào?",
        contexts=[{
            "chunk_id": "dbp",
            "title": "Chiến dịch Điện Biên Phủ",
            "text": "Chiến dịch giành thắng lợi.",
            "validated_source_text": "Ngày 11 tháng 3 năm 1954, chiến dịch bước vào giai đoạn chuẩn bị.",
            "claims": ["Chiến dịch giành thắng lợi."],
        }],
        analysis={"facets": ["process"]},
        tool_trace=[],
    )
    assert result["unsupported_years"] == []


def test_year_not_in_any_cited_selected_source_remains_unsupported():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [dbp]\n\nTrả lời:\nChiến dịch này diễn ra vào năm 1968."
    )
    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="Chiến dịch Điện Biên Phủ diễn ra như thế nào?",
        contexts=[{
            "chunk_id": "dbp",
            "title": "Chiến dịch Điện Biên Phủ",
            "text": "Chiến dịch giành thắng lợi.",
            "validated_source_text": "Chiến dịch giành thắng lợi.",
            "claims": ["Chiến dịch giành thắng lợi."],
        }],
        analysis={"facets": ["process"]},
        tool_trace=[],
    )
    assert result["unsupported_years"] == ["1968"]


def test_structural_compare_accepts_concise_comparison_and_rejects_two_summaries():
    contexts = _compare_contexts()
    bad = "Cách mạng Tháng Tám giành chính quyền. Điện Biên Phủ giành thắng lợi quân sự."
    good = (
        "Cả hai đều giành thắng lợi, nhưng Cách mạng Tháng Tám là khởi nghĩa giành chính quyền, "
        "trong khi Điện Biên Phủ là chiến dịch quân sự chống Pháp."
    )
    assert "shallow_comparison" in _deep_answer_quality_issues(
        QUESTION, contexts, bad, answer_depth="deep"
    )
    assert "shallow_comparison" not in _deep_answer_quality_issues(
        QUESTION, contexts, good, answer_depth="deep"
    )


def test_good_concise_compare_uses_one_history_call_bad_summary_may_retry_once():
    contexts = _compare_contexts()
    good = (
        "Cả hai đều giành thắng lợi, nhưng Cách mạng Tháng Tám là khởi nghĩa giành chính quyền, "
        "trong khi Điện Biên Phủ là chiến dịch quân sự chống Pháp."
    )
    good_runtime = FakeHistoryRuntime(
        f"Nguồn được dùng: [cmt8] [dbp]\n\nTrả lời:\n{good}"
    )
    good_result = HistoryAnswererAgent(model_runtime=good_runtime).answer(
        question=QUESTION,
        contexts=contexts,
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
        inference_mode="agentic_rag",
    )
    assert good_result["answer_provenance"]["history_generation_calls"] == 1

    bad_runtime = FakeHistoryRuntime([
        "Nguồn được dùng: [cmt8] [dbp]\n\nTrả lời:\nCách mạng Tháng Tám giành chính quyền. Điện Biên Phủ thắng lợi.",
        f"Nguồn được dùng: [cmt8] [dbp]\n\nTrả lời:\n{good}",
    ])
    bad_result = HistoryAnswererAgent(model_runtime=bad_runtime).answer(
        question=QUESTION,
        contexts=contexts,
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
        inference_mode="agentic_rag",
    )
    assert bad_result["answer_provenance"]["history_generation_calls"] == 2
    assert bad_result["answer_provenance"]["history_retry_used"] is True


def test_natural_style_cleanup_removes_generic_provenance_and_preserves_named_source():
    contexts = _compare_contexts()[:1]
    generic_runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [cmt8]\n\nTrả lời:\nTài liệu cho thấy Mỹ thất bại. Các nguồn cho thấy chiến thắng này có ý nghĩa."
    )
    generic = HistoryAnswererAgent(model_runtime=generic_runtime).answer(
        question="Mỹ thất bại trong chiến tranh Việt Nam vì sao?",
        contexts=contexts,
        analysis={"facets": ["cause"]},
        tool_trace=[],
        avoid_generic_source_prefix=True,
    )
    assert generic["answer"] == "Mỹ thất bại. chiến thắng này có ý nghĩa."

    named_runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [cmt8]\n\nTrả lời:\nTheo Đại Việt sử ký Toàn thư, Ngô Quyền chuẩn bị trận địa."
    )
    named = HistoryAnswererAgent(model_runtime=named_runtime).answer(
        question="Ngô Quyền chuẩn bị trận Bạch Đằng như thế nào?",
        contexts=contexts,
        analysis={"facets": ["process"]},
        tool_trace=[],
        avoid_generic_source_prefix=True,
    )
    assert named["answer"].startswith("Theo Đại Việt sử ký Toàn thư")


def test_normalized_trace_has_sections_scores_and_safe_failure_error():
    trace = _build_debug({
        "question": QUESTION,
        "inference_mode": "hybrid_rag",
        "analysis": {"facet": "compare", "comparison_targets": ["A", "B"]},
        "retrieval": {
            "retrieval_question": QUESTION,
            "target_specific_queries": {"target_a_query": "A", "target_b_query": "B"},
            "candidates20": [{"chunk_id": "a", "title": "A", "final_retrieval_score": 0.9}],
            "final_context": [{"chunk_id": "a", "title": "A", "text": "secret full chunk"}],
        },
        "history_debug": {
            "generation_calls": 1,
            "question_type": "compare",
            "prompt": "private model prompt",
            "chain_of_thought": "private reasoning",
        },
        "answer_provenance": {"history_generation_calls": 1, "total_llm_calls": 1},
    })
    assert trace["request"]["question_type"] == "compare"
    assert trace["retrieval"]["merged_candidates"][0]["final_retrieval_score"] == 0.9
    assert trace["research"] == {}
    assert trace["evidence"] == {}
    assert "prompt" not in trace["history"]
    assert "chain_of_thought" not in trace["history"]

    payload = ChatRequest(
        conversation_id="00000000-0000-0000-0000-000000000001",
        question=QUESTION,
        debug=True,
    )
    failed = _failure_debug_trace(
        payload=payload,
        mode="agentic_rag",
        stage="evidence",
        code="grounding_contract_failed",
        diagnostics={"authorization": "secret", "validation_errors": [{"code": "cross_id_claim"}]},
    )
    assert failed["errors"][0]["code"] == "grounding_contract_failed"
    assert "authorization" not in failed["errors"][0]


def test_debug_trace_persists_with_its_assistant_message(tmp_path):
    store = ConversationStore(tmp_path / "trace.db")
    conversation = store.create_conversation("trace-client", "Trace")
    stored = store.add_message(
        "trace-client",
        conversation["id"],
        "assistant",
        "Câu trả lời.",
        debug_trace={"request": {"mode": "hybrid_rag"}, "retrieval": {"query_variants": ["q"]}},
    )
    reloaded = store.get_conversation_detail("trace-client", conversation["id"])

    assert stored["debug_trace"]["request"]["mode"] == "hybrid_rag"
    assert reloaded["messages"][0]["debug_trace"]["retrieval"]["query_variants"] == ["q"]
