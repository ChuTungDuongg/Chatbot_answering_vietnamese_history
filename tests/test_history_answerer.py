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


def test_no_selected_evidence_is_an_explicit_guard_not_a_model_output():
    runtime = FakeHistoryRuntime("unused")

    result = _answer(HistoryAnswererAgent(model_runtime=runtime), contexts=[])

    assert runtime.calls == []
    assert result["status"] == "blocked_no_context"
    assert result["answer_provenance"]["source"] == "deterministic_guard"
    assert result["answer_provenance"]["guard_name"] == "no_selected_evidence"
