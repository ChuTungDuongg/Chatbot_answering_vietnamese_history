from __future__ import annotations

from app.agents.history_answerer import HistoryAnswererAgent


class FakeGenerator:
    def answer_from_retrieval(self, *, question, retrieval, history=None):
        return {
            "question": question,
            "answer": "Nguồn được dùng: [c1]\nTrả lời: OK",
            "status": "ok",
            "source_ids": ["c1"],
            "source_chunks": retrieval["final_context"],
            "retrieval": retrieval,
            "tool_trace": retrieval["tool_trace"],
        }


def test_history_answerer_passes_agent_contexts():
    agent = HistoryAnswererAgent(FakeGenerator())
    result = agent.answer(
        question="Q",
        contexts=[{"chunk_id": "c1", "text": "ctx"}],
        analysis={},
        tool_trace=["agent:research"],
    )
    assert result["source_ids"] == ["c1"]
    assert result["retrieval"]["final_context"][0]["chunk_id"] == "c1"

