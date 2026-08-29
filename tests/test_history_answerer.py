from __future__ import annotations

import pytest

from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.history_contract import HistoryAnswerContractError


class FakeHistoryRuntime:
    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def _contexts():
    return [
        {
            "chunk_id": "ev_relevant",
            "title": "Nguồn liên quan",
            "text": "Bằng chứng trực tiếp trả lời câu hỏi.",
            "source_kind": "history",
        },
        {
            "chunk_id": "ev_distractor",
            "title": "Nguồn nhiễu",
            "text": "Một đoạn có cùng chủ đề nhưng không trả lời trọng tâm.",
            "source_kind": "history",
        },
    ]


def _answer(agent: HistoryAnswererAgent, contexts=None, history=None):
    return agent.answer(
        question="Câu hỏi lịch sử cần bằng chứng?",
        contexts=_contexts() if contexts is None else contexts,
        analysis={"facets": ["significance"]},
        tool_trace=["agent:research", "agent:evidence_critic"],
        history=history,
    )


def test_grounded_history_role_calls_adapter_once_with_only_supplied_evidence():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant]\n\n"
        "Trả lời:\nCâu trả lời ngắn nhưng có căn cứ."
    )
    result = _answer(
        HistoryAnswererAgent(model_runtime=runtime),
        history=[{"role": "user", "content": "Bí mật hội thoại không thuộc SFT."}],
    )

    assert len(runtime.calls) == 1
    assert runtime.calls[0]["adapter"] == "history"
    assert [message["role"] for message in runtime.calls[0]["messages"]] == ["user"]
    prompt = runtime.calls[0]["messages"][0]["content"]
    assert "[ev_relevant] Nguồn liên quan" in prompt
    assert "[ev_distractor] Nguồn nhiễu" in prompt
    assert "Bí mật hội thoại" not in prompt
    assert result["source_ids"] == ["ev_relevant"]
    assert set(result["source_ids"]) <= {"ev_relevant", "ev_distractor"}


def test_noisy_context_does_not_trigger_a_legacy_guard_override():
    expected = "Câu trả lời trực tiếp từ History adapter."
    runtime = FakeHistoryRuntime(
        f"Nguồn được dùng: [ev_relevant]\n\nTrả lời:\n{expected}"
    )

    result = _answer(HistoryAnswererAgent(model_runtime=runtime))

    assert result["answer"] == expected
    assert result["status"] == "ok"
    assert result["rewrite_used"] is False
    assert result["repair_attempted"] is False
    assert result["structured_expansion_used"] is False
    assert result["answer_provenance"]["guard_override"] is False


def test_insufficient_output_preserves_training_contract_without_hallucinated_sources():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: []\n\n"
        "Trả lời:\nTài liệu chưa cung cấp thông tin cần thiết để kết luận."
    )

    result = _answer(HistoryAnswererAgent(model_runtime=runtime))

    assert result["status"] == "insufficient"
    assert result["source_ids"] == []
    assert result["answer_provenance"]["source"] == "history_adapter"


def test_false_premise_output_can_correct_with_valid_supplied_citations():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant], [ev_distractor]\n\n"
        "Trả lời:\nTiền đề chưa đúng theo hai nguồn được cung cấp."
    )

    result = _answer(HistoryAnswererAgent(model_runtime=runtime))

    assert result["status"] == "ok"
    assert result["source_ids"] == ["ev_relevant", "ev_distractor"]


def test_provenance_reports_direct_history_adapter_generation():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant]\n\nTrả lời:\nCó căn cứ."
    )

    result = _answer(HistoryAnswererAgent(model_runtime=runtime))

    assert result["answer_provenance"] == {
        "source": "history_adapter",
        "history_adapter_called": True,
        "history_generation_calls": 1,
        "guard_short_circuit": False,
        "guard_name": None,
        "guard_override": False,
        "answer_depth": "standard",
        "structured_expansion_used": False,
    }
    assert result["history_debug"]["generation_calls"] == 1
    assert result["history_debug"]["input_evidence_ids"] == [
        "ev_relevant",
        "ev_distractor",
    ]


def test_invented_history_citation_is_rejected():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_unknown]\n\nTrả lời:\nKhông hợp lệ."
    )

    with pytest.raises(HistoryAnswerContractError, match="invented citation IDs"):
        _answer(HistoryAnswererAgent(model_runtime=runtime))


def test_deep_factual_prompt_requests_direct_answer_with_context():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant]\n\nTrả lời:\nVõ Nguyên Giáp là câu trả lời."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="Ai được mệnh danh là anh cả Quân đội Nhân dân Việt Nam?",
        contexts=_contexts(),
        analysis={"facets": ["general"]},
        tool_trace=[],
        answer_depth="deep",
    )

    prompt = runtime.calls[0]["messages"][0]["content"]
    assert "answer_depth=deep" in prompt
    assert "Trả lời trực tiếp ngay ở câu đầu" in prompt
    assert "bối cảnh liên quan" in prompt
    assert result["history_debug"]["question_type"] == "factual"


def test_history_debug_reports_claim_breadth_and_unsupported_years():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_01] [ev_02]\n\n"
        "Trả lời:\nBạch Đằng năm 938 có ý nghĩa lớn. "
        "Tuy nhiên, câu trả lời này còn nhắc thêm biến cố năm 1968."
    )
    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        contexts=[
            {
                "chunk_id": "ev_01",
                "title": "Trận Bạch Đằng (938)",
                "text": "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc.",
                "source_kind": "history",
                "claims": ["Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc."],
            },
            {
                "chunk_id": "ev_02",
                "title": "Nhà Ngô",
                "text": "Sau chiến thắng, Ngô Quyền xưng vương và mở ra nền độc lập tự chủ.",
                "source_kind": "history",
                "claims": ["Sau chiến thắng, Ngô Quyền xưng vương và mở ra nền độc lập tự chủ."],
            },
        ],
        analysis={"facets": ["significance"]},
        tool_trace=[],
        answer_depth="deep",
    )

    assert result["history_debug"]["input_claim_count"] == 2
    assert result["history_debug"]["input_source_kind_counts"] == {"history": 2}
    assert result["unsupported_years"] == ["1968"]
    assert "unsupported_year:1968" in result["quality_warnings"]


def test_deep_significance_two_sentence_collapse_expands_from_grounded_claims():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_01] [ev_02]\n\n"
        "Trả lời:\nChiến thắng Bạch Đằng năm 938 có ý nghĩa rất lớn. "
        "Nó mở ra thời kỳ độc lập tự chủ."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        contexts=[
            {
                "chunk_id": "ev_01",
                "title": "Trận Bạch Đằng (938)",
                "text": (
                    "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc. "
                    "Chiến thắng Bạch Đằng năm 938 mở ra thời kỳ độc lập tự chủ."
                ),
                "source_kind": "history",
                "claims": [
                    "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc.",
                    "Chiến thắng Bạch Đằng năm 938 mở ra thời kỳ độc lập tự chủ.",
                ],
            },
            {
                "chunk_id": "ev_02",
                "title": "Nhà Ngô",
                "text": (
                    "Sau chiến thắng, Ngô Quyền xưng vương. "
                    "Sự kiện này đánh dấu bước chuyển sang nền độc lập."
                ),
                "source_kind": "history",
                "claims": [
                    "Sau chiến thắng, Ngô Quyền xưng vương.",
                    "Sự kiện này đánh dấu bước chuyển sang nền độc lập.",
                ],
            },
        ],
        analysis={"facets": ["significance"]},
        tool_trace=[],
        answer_depth="deep",
    )

    assert result["structured_expansion_used"] is True
    assert "deep_answer_collapse" not in result["quality_warnings"]
    assert "Các khía cạnh được tài liệu hỗ trợ" in result["answer"]
    assert set(result["source_ids"]) == {"ev_01", "ev_02"}


def test_deep_compare_short_unstructured_answer_expands_from_both_targets():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        "Trả lời:\nCách mạng Tháng Tám và chiến thắng Điện Biên Phủ đều là mốc lớn, "
        "nhưng khác nhau về tính chất."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {
                "chunk_id": "ev_august",
                "title": "Cách mạng Tháng Tám",
                "text": (
                    "Cách mạng Tháng Tám năm 1945 giành chính quyền. "
                    "Cách mạng Tháng Tám lập nên nước Việt Nam Dân chủ Cộng hòa."
                ),
                "source_kind": "history",
                "claims": [
                    "Cách mạng Tháng Tám năm 1945 giành chính quyền.",
                    "Cách mạng Tháng Tám lập nên nước Việt Nam Dân chủ Cộng hòa.",
                ],
            },
            {
                "chunk_id": "ev_dien_bien",
                "title": "Chiến thắng Điện Biên Phủ",
                "text": (
                    "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève. "
                    "Chiến thắng Điện Biên Phủ chấm dứt chiến tranh ở Đông Dương."
                ),
                "source_kind": "history",
                "claims": [
                    "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève.",
                    "Chiến thắng Điện Biên Phủ chấm dứt chiến tranh ở Đông Dương.",
                ],
            },
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    assert result["structured_expansion_used"] is True
    assert "Khái quát" in result["answer"]
    assert "- Cách mạng Tháng Tám:" in result["answer"]
    assert "- chiến thắng Điện Biên Phủ:" in result["answer"]
    assert set(result["source_ids"]) == {"ev_august", "ev_dien_bien"}


def test_compare_target_b_claim_is_not_rendered_under_target_a_section():
    b_claim = "Để bảo đảm nguyên tắc 'trận đầu phải thắng', tham mưu đã bố trí một lực lượng mạnh hơn quân Pháp."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        "Trả lời:\nHai sự kiện có thể so sánh, nhưng câu trả lời này còn quá ngắn."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {
                "chunk_id": "ev_august",
                "title": "Cách mạng Tháng Tám",
                "text": "Cách mạng Tháng Tám năm 1945 là cuộc khởi nghĩa giành chính quyền.",
                "claims": ["Cách mạng Tháng Tám năm 1945 là cuộc khởi nghĩa giành chính quyền."],
                "comparison_target": "target_a",
            },
            {
                "chunk_id": "ev_dien_bien",
                "title": "Chiến dịch Điện Biên Phủ",
                "text": b_claim,
                "claims": [b_claim],
                "comparison_target": "target_b",
            },
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    target_a_section = result["answer"].split("- chiến thắng Điện Biên Phủ:", 1)[0]
    assert result["structured_expansion_used"] is True
    assert b_claim not in target_a_section
    assert b_claim in result["answer"].split("- chiến thắng Điện Biên Phủ:", 1)[1]
    assert result["history_debug"]["comparison_evidence_groups"]["target_b"]["evidence"][0]["chunk_id"] == "ev_dien_bien"


def test_compare_target_a_claim_is_not_rendered_under_target_b_section():
    a_claim = "Cách mạng Tháng Tám năm 1945 giành chính quyền và lập nên nước Việt Nam Dân chủ Cộng hòa."
    b_claim = "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        "Trả lời:\nHai sự kiện khác nhau về tính chất."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {"chunk_id": "ev_august", "title": "Cách mạng Tháng Tám", "text": a_claim, "claims": [a_claim], "comparison_target": "target_a"},
            {"chunk_id": "ev_dien_bien", "title": "Chiến thắng Điện Biên Phủ", "text": b_claim, "claims": [b_claim], "comparison_target": "target_b"},
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    target_b_section = result["answer"].split("- chiến thắng Điện Biên Phủ:", 1)[1]
    assert a_claim not in target_b_section
    assert b_claim in target_b_section


def test_compare_shared_evidence_can_support_similarity_section():
    shared_claim = "Hai sự kiện đều được nêu là mốc có ý nghĩa chiến lược trong lịch sử Việt Nam."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_shared]\n\nTrả lời:\nSo sánh còn ngắn."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {
                "chunk_id": "ev_shared",
                "title": "Tổng kết lịch sử Việt Nam",
                "text": shared_claim,
                "claims": [shared_claim],
                "comparison_target": "shared",
            },
            {
                "chunk_id": "ev_august",
                "title": "Cách mạng Tháng Tám",
                "text": "Cách mạng Tháng Tám năm 1945 giành chính quyền.",
                "claims": ["Cách mạng Tháng Tám năm 1945 giành chính quyền."],
                "comparison_target": "target_a",
            },
            {
                "chunk_id": "ev_dien_bien",
                "title": "Điện Biên Phủ",
                "text": "Điện Biên Phủ năm 1954 là thắng lợi quân sự.",
                "claims": ["Điện Biên Phủ năm 1954 là thắng lợi quân sự."],
                "comparison_target": "target_b",
            },
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    similarity_section = result["answer"].split("Điểm khác nhau", 1)[0]
    assert shared_claim in similarity_section
    assert "ev_shared" in result["source_ids"]


def test_compare_unknown_evidence_is_not_force_assigned_to_either_target():
    unknown_claim = "Một tài liệu phụ nói về bối cảnh khu vực nhưng không nêu rõ sự kiện nào."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien] [ev_unknown]\n\n"
        "Trả lời:\nSo sánh quá ngắn."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {"chunk_id": "ev_august", "title": "Cách mạng Tháng Tám", "text": "Cách mạng Tháng Tám năm 1945 giành chính quyền.", "comparison_target": "target_a"},
            {"chunk_id": "ev_dien_bien", "title": "Điện Biên Phủ", "text": "Điện Biên Phủ năm 1954 là thắng lợi quân sự.", "comparison_target": "target_b"},
            {"chunk_id": "ev_unknown", "title": "Bối cảnh", "text": unknown_claim, "claims": [unknown_claim], "comparison_target": "unknown"},
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    assert unknown_claim not in result["answer"]
    assert "ev_unknown" not in result["source_ids"]
    assert result["history_debug"]["comparison_evidence_groups"]["unknown_evidence"][0]["chunk_id"] == "ev_unknown"


def test_compare_similarity_section_requires_shared_or_two_sided_support():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\nTrả lời:\nSo sánh ngắn."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {"chunk_id": "ev_august", "title": "Cách mạng Tháng Tám", "text": "Cách mạng Tháng Tám năm 1945 giành chính quyền.", "comparison_target": "target_a"},
            {"chunk_id": "ev_dien_bien", "title": "Điện Biên Phủ", "text": "Điện Biên Phủ năm 1954 là thắng lợi quân sự.", "comparison_target": "target_b"},
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    similarity_section = result["answer"].split("Điểm khác nhau", 1)[0]
    assert "chưa nêu một điểm giống nhau đủ rõ cho cả hai đối tượng" in similarity_section


def test_compare_one_sided_claim_is_not_duplicated_across_target_sections():
    claim = "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\nTrả lời:\nSo sánh ngắn."
    )

    result = HistoryAnswererAgent(model_runtime=runtime).answer(
        question="So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        contexts=[
            {"chunk_id": "ev_august", "title": "Cách mạng Tháng Tám", "text": "Cách mạng Tháng Tám năm 1945 giành chính quyền.", "comparison_target": "target_a"},
            {"chunk_id": "ev_dien_bien", "title": "Điện Biên Phủ", "text": claim, "claims": [claim], "comparison_target": "target_b"},
        ],
        analysis={"facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    assert result["answer"].count(claim) == 1
    assert set(result["source_ids"]) == {"ev_august", "ev_dien_bien"}


def test_hybrid_generic_source_prefix_is_removed_when_enabled():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant]\n\n"
        "Trả lời:\nTheo tài liệu, chiến thắng Bạch Đằng năm 938 có ý nghĩa lớn."
    )

    result = _answer(
        HistoryAnswererAgent(model_runtime=runtime),
        contexts=_contexts(),
    )
    direct = HistoryAnswererAgent(model_runtime=FakeHistoryRuntime(runtime.output)).answer(
        question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        contexts=_contexts(),
        analysis={"facets": ["significance"]},
        tool_trace=[],
        answer_depth="standard",
        avoid_generic_source_prefix=True,
    )

    assert result["answer"].startswith("Theo tài liệu")
    assert direct["answer"] == "chiến thắng Bạch Đằng năm 938 có ý nghĩa lớn."
    assert direct["source_ids"] == ["ev_relevant"]


def test_hybrid_direct_opening_and_named_source_attribution_are_preserved():
    direct_runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant]\n\n"
        "Trả lời:\nChiến thắng Bạch Đằng năm 938 có ý nghĩa lớn."
    )
    named_runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_relevant]\n\n"
        "Trả lời:\nTheo Đại Việt sử ký Toàn thư, Ngô Quyền cho đóng cọc trên sông."
    )

    direct = HistoryAnswererAgent(model_runtime=direct_runtime).answer(
        question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        contexts=_contexts(),
        analysis={"facets": ["significance"]},
        tool_trace=[],
        avoid_generic_source_prefix=True,
    )
    named = HistoryAnswererAgent(model_runtime=named_runtime).answer(
        question="Ngô Quyền chuẩn bị trận Bạch Đằng như thế nào?",
        contexts=_contexts(),
        analysis={"facets": ["process"]},
        tool_trace=[],
        avoid_generic_source_prefix=True,
    )

    assert direct["answer"] == "Chiến thắng Bạch Đằng năm 938 có ý nghĩa lớn."
    assert named["answer"].startswith("Theo Đại Việt sử ký Toàn thư")
    assert direct["source_ids"] == ["ev_relevant"]
    assert named["source_ids"] == ["ev_relevant"]


def test_no_selected_evidence_is_an_explicit_guard_not_a_model_output():
    runtime = FakeHistoryRuntime("unused")

    result = _answer(HistoryAnswererAgent(model_runtime=runtime), contexts=[])

    assert runtime.calls == []
    assert result["status"] == "blocked_no_context"
    assert result["answer_provenance"]["source"] == "deterministic_guard"
    assert result["answer_provenance"]["guard_name"] == "no_selected_evidence"
