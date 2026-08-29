from __future__ import annotations

import pytest

from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.history_contract import HistoryAnswerContractError


FORBIDDEN_META = (
    "Kết luận trực tiếp",
    "Các khía cạnh được tài liệu hỗ trợ",
    "Tổng hợp",
    "câu trả lời nên được hiểu",
    "cần được đặt cạnh nhau theo đúng nhóm bằng chứng",
    "Các nguồn cho phép so sánh",
)


class FakeHistoryRuntime:
    def __init__(self, output: str | list[str]):
        self.outputs = [output] if isinstance(output, str) else list(output)
        self.output = self.outputs[0]
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


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


def _answer(agent: HistoryAnswererAgent, contexts=None, history=None, **kwargs):
    return agent.answer(
        question="Câu hỏi lịch sử cần bằng chứng?",
        contexts=_contexts() if contexts is None else contexts,
        analysis={"facets": ["significance"]},
        tool_trace=["agent:research", "agent:evidence_critic"],
        history=history,
        **kwargs,
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

    assert result["answer_provenance"]["source"] == "history_adapter"
    assert result["answer_provenance"]["history_adapter_called"] is True
    assert result["answer_provenance"]["history_generation_calls"] == 1
    assert result["answer_provenance"]["history_retry_used"] is False
    assert result["answer_provenance"]["guard_short_circuit"] is False
    assert result["answer_provenance"]["guard_override"] is False
    assert result["answer_provenance"]["answer_depth"] == "standard"
    assert result["answer_provenance"]["structured_expansion_used"] is False
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
    assert "Yêu cầu trả lời:" in prompt
    assert "Trả lời trực tiếp ngay ở câu đầu" in prompt
    assert "bối cảnh ngắn" in prompt
    assert result["history_debug"]["question_type"] == "factual"
    assert result["history_debug"]["generation_calls"] == 1


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


def test_deep_significance_two_sentence_collapse_retries_history_once_with_same_evidence():
    rich_answer = (
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa trước hết ở chỗ chấm dứt thời kỳ Bắc thuộc "
        "và mở ra thời kỳ độc lập tự chủ. Từ thắng lợi đó, Ngô Quyền có cơ sở xưng vương, "
        "đánh dấu bước chuyển chính trị sang một nhà nước tự chủ. Vì vậy, ý nghĩa của chiến thắng "
        "không chỉ là đánh bại quân Nam Hán trong một trận thủy chiến, mà còn là khẳng định khả năng "
        "tự quyết của người Việt sau nhiều thế kỷ bị phương Bắc đô hộ."
    )
    runtime = FakeHistoryRuntime(
        [
            "Nguồn được dùng: [ev_01] [ev_02]\n\n"
            "Trả lời:\nChiến thắng Bạch Đằng năm 938 có ý nghĩa rất lớn. "
            "Nó mở ra thời kỳ độc lập tự chủ.",
            f"Nguồn được dùng: [ev_01] [ev_02]\n\nTrả lời:\n{rich_answer}",
        ]
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
        inference_mode="agentic_rag",
    )

    assert len(runtime.calls) == 2
    assert result["answer"] == rich_answer
    assert result["structured_expansion_used"] is False
    assert result["answer_provenance"]["history_generation_calls"] == 2
    assert result["answer_provenance"]["history_retry_used"] is True
    assert result["answer_provenance"]["history_retry_reason"] == "deep_answer_collapse"
    assert result["history_debug"]["first_answer_words"] < result["history_debug"]["final_answer_words"]
    assert "[ev_01]" in runtime.calls[0]["messages"][0]["content"]
    assert "[ev_02]" in runtime.calls[0]["messages"][0]["content"]
    assert "[ev_01]" in runtime.calls[1]["messages"][0]["content"]
    assert "[ev_02]" in runtime.calls[1]["messages"][0]["content"]
    assert "deep_answer_collapse" not in result["quality_warnings"]
    assert set(result["source_ids"]) == {"ev_01", "ev_02"}
    assert not any(phrase in result["answer"] for phrase in FORBIDDEN_META)


def test_invalid_history_retry_falls_back_to_first_valid_model_answer():
    first_answer = (
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa rất lớn. "
        "Nó mở ra thời kỳ độc lập tự chủ."
    )
    runtime = FakeHistoryRuntime(
        [
            f"Nguồn được dùng: [ev_01] [ev_02]\n\nTrả lời:\n{first_answer}",
            "Nguồn được dùng: [ev_unknown]\n\nTrả lời:\nCâu trả lời retry không hợp lệ.",
        ]
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
        inference_mode="agentic_rag",
    )

    assert len(runtime.calls) == 2
    assert result["answer"] == first_answer
    assert result["source_ids"] == ["ev_01", "ev_02"]
    assert result["answer_provenance"]["history_generation_calls"] == 2
    assert result["answer_provenance"]["history_retry_used"] is True
    assert result["answer_provenance"]["history_retry_selected"] is False
    assert result["answer_provenance"]["history_retry_error"].startswith("HistoryAnswerContractError:")
    assert "history:retry_fallback_first" in result["tool_trace"]


def test_good_deep_first_answer_does_not_retry():
    answer = (
        "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc, mở ra thời kỳ độc lập tự chủ "
        "và tạo cơ sở để Ngô Quyền xưng vương. Ý nghĩa của thắng lợi nằm ở cả kết quả quân sự trước "
        "Nam Hán lẫn bước chuyển chính trị sang quyền tự quyết của người Việt."
    )
    runtime = FakeHistoryRuntime(
        f"Nguồn được dùng: [ev_01] [ev_02]\n\nTrả lời:\n{answer}"
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
                "claims": [
                    "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc.",
                    "Chiến thắng Bạch Đằng năm 938 mở ra thời kỳ độc lập tự chủ.",
                ],
            },
            {
                "chunk_id": "ev_02",
                "title": "Nhà Ngô",
                "text": "Sau chiến thắng, Ngô Quyền xưng vương và đánh dấu bước chuyển sang nền độc lập.",
                "claims": ["Sau chiến thắng, Ngô Quyền xưng vương và đánh dấu bước chuyển sang nền độc lập."],
            },
        ],
        analysis={"facets": ["significance"]},
        tool_trace=[],
        answer_depth="deep",
        inference_mode="agentic_rag",
    )

    assert len(runtime.calls) == 1
    assert result["answer"] == answer
    assert result["answer_provenance"]["history_generation_calls"] == 1
    assert result["answer_provenance"]["history_retry_used"] is False
    assert result["structured_expansion_used"] is False


def test_deep_compare_short_answer_retries_history_and_uses_model_synthesis():
    retry_answer = (
        "Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ đều là những mốc có thể đặt cạnh nhau, "
        "nhưng phần khác biệt giữa chúng nổi bật rõ hơn. Cách mạng Tháng Tám năm 1945 "
        "là quá trình giành chính quyền và lập nên nước Việt Nam Dân chủ Cộng hòa. Chiến thắng "
        "Điện Biên Phủ năm 1954 là thắng lợi quân sự buộc Pháp ký Hiệp định Genève và chấm dứt "
        "chiến tranh ở Đông Dương. Vì vậy, một bên nổi bật ở bước ngoặt chính trị lập chính quyền, "
        "còn bên kia nổi bật ở thắng lợi quân sự - ngoại giao kết thúc chiến tranh."
    )
    runtime = FakeHistoryRuntime(
        [
            "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
            "Trả lời:\nCách mạng Tháng Tám và chiến thắng Điện Biên Phủ đều là mốc lớn.",
            f"Nguồn được dùng: [ev_august] [ev_dien_bien]\n\nTrả lời:\n{retry_answer}",
        ]
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
        inference_mode="agentic_rag",
    )

    assert len(runtime.calls) == 2
    assert result["answer"] == retry_answer
    assert result["structured_expansion_used"] is False
    assert result["answer_provenance"]["history_generation_calls"] == 2
    assert result["answer_provenance"]["history_retry_used"] is True
    assert set(result["source_ids"]) == {"ev_august", "ev_dien_bien"}
    assert not any(phrase in result["answer"] for phrase in FORBIDDEN_META)


def test_compare_target_leakage_triggers_one_history_retry():
    b_claim = "Để bảo đảm nguyên tắc 'trận đầu phải thắng', tham mưu đã bố trí một lực lượng mạnh hơn quân Pháp."
    corrected = (
        "Cách mạng Tháng Tám năm 1945 là cuộc khởi nghĩa giành chính quyền. "
        "Chiến dịch Điện Biên Phủ được nguồn riêng mô tả qua việc bố trí lực lượng mạnh hơn quân Pháp "
        "để bảo đảm nguyên tắc 'trận đầu phải thắng'. Hai ý này phải được giữ đúng phía: "
        "Cách mạng Tháng Tám thuộc nhóm giành chính quyền, còn Điện Biên Phủ thuộc nhóm chiến dịch quân sự."
    )
    runtime = FakeHistoryRuntime(
        [
            "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
            f"Trả lời:\nCách mạng Tháng Tám:\n{b_claim}\n"
            f"Chiến thắng Điện Biên Phủ:\n{b_claim}",
            f"Nguồn được dùng: [ev_august] [ev_dien_bien]\n\nTrả lời:\n{corrected}",
        ]
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
        inference_mode="agentic_rag",
    )

    assert len(runtime.calls) == 2
    assert result["answer"] == corrected
    assert result["answer_provenance"]["history_retry_reason"] == "comparison_target_leakage"
    assert result["structured_expansion_used"] is False
    target_a_section = result["answer"].split("Điện Biên Phủ", 1)[0]
    assert b_claim not in target_a_section
    assert result["history_debug"]["comparison_evidence_groups"]["target_b"]["evidence"][0]["chunk_id"] == "ev_dien_bien"


def test_compare_target_a_claim_is_not_rendered_under_target_b_section():
    a_claim = "Cách mạng Tháng Tám năm 1945 giành chính quyền và lập nên nước Việt Nam Dân chủ Cộng hòa."
    b_claim = "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        f"Trả lời:\nCách mạng Tháng Tám: {a_claim}\n"
        f"Chiến thắng Điện Biên Phủ: {b_claim}"
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

    target_b_section = result["answer"].split("Chiến thắng Điện Biên Phủ:", 1)[1]
    assert a_claim not in target_b_section
    assert b_claim in target_b_section
    assert result["structured_expansion_used"] is False


def test_compare_shared_evidence_can_support_similarity_section():
    shared_claim = "Hai sự kiện đều được nêu là mốc có ý nghĩa chiến lược trong lịch sử Việt Nam."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_shared]\n\n"
        f"Trả lời:\nĐiểm tương đồng được nguồn chung hỗ trợ là: {shared_claim}"
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

    assert shared_claim in result["answer"]
    assert "ev_shared" in result["source_ids"]
    assert result["structured_expansion_used"] is False


def test_compare_unknown_evidence_is_not_force_assigned_to_either_target():
    unknown_claim = "Một tài liệu phụ nói về bối cảnh khu vực nhưng không nêu rõ sự kiện nào."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        "Trả lời:\nCách mạng Tháng Tám được mô tả là sự kiện giành chính quyền, "
        "còn Điện Biên Phủ được mô tả là thắng lợi quân sự; phần bối cảnh không rõ đối tượng không được gán ép."
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
    assert result["structured_expansion_used"] is False
    assert result["history_debug"]["comparison_evidence_groups"]["unknown_evidence"][0]["chunk_id"] == "ev_unknown"


def test_compare_similarity_section_requires_shared_or_two_sided_support():
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        "Trả lời:\nKhông nêu điểm giống nhau cụ thể ngoài việc cả hai được đặt trong câu hỏi so sánh; "
        "phần chắc hơn là khác biệt giữa một sự kiện giành chính quyền và một thắng lợi quân sự."
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

    assert "không nêu điểm giống nhau cụ thể" in result["answer"].lower()
    assert "thắng lợi quân sự" in result["answer"]
    assert result["structured_expansion_used"] is False


def test_compare_one_sided_claim_is_not_duplicated_across_target_sections():
    claim = "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève."
    runtime = FakeHistoryRuntime(
        "Nguồn được dùng: [ev_august] [ev_dien_bien]\n\n"
        f"Trả lời:\nCách mạng Tháng Tám năm 1945 giành chính quyền. "
        f"Điện Biên Phủ được nguồn riêng mô tả như sau: {claim}"
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
    assert result["structured_expansion_used"] is False


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
