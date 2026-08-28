from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agents.orchestrator import HybridRAGOrchestrator
from app.api import routes
from app.api.routes import _execute_chat, _resolve_inference_mode
from app.chat.store import ConversationStore
from app.schemas import ChatRequest


class FakeRetriever:
    final_context_k = 6

    def __init__(self):
        self.calls = []

    def retrieve(self, question: str, final_k: int):
        self.calls.append({"question": question, "final_k": final_k})
        return {
            "question": question,
            "final_context": [
                {"chunk_id": "ev_01", "text": "Ngô Quyền thắng trên sông Bạch Đằng năm 938.", "source_kind": "history"}
            ],
            "tool_trace": ["hybrid_retriever"],
            "is_ood": False,
            "ood_reason": "",
        }

    def analyze_question(self, question: str):
        return {"question": question, "is_multi_part": False}

    def context_title_diversity(self, _contexts):
        return 1.0


class FakeRetrievalRuntime:
    max_history_messages = 6
    retrieval_history_messages = 4

    @staticmethod
    def normalize_history(history, current_question=None):
        return history or []

    @staticmethod
    def build_retrieval_question(question, history):
        return question, bool(history)


class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        contexts = kwargs["contexts"]
        return {
            "question": kwargs["question"],
            "answer": "Bạch Đằng năm 938 là một thắng lợi quan trọng. [ev_01]",
            "status": "ok",
            "source_ids": ["ev_01"],
            "source_chunks": contexts,
            "retrieval": {},
            "analysis": kwargs["analysis"],
            "tool_trace": kwargs["tool_trace"],
            "answer_provenance": {
                "source": "history_adapter",
                "history_adapter_called": True,
                "history_generation_calls": 1,
            },
            "history_debug": {"generation_calls": 1},
        }


def test_hybrid_rag_calls_retriever_and_history_without_research_or_evidence():
    retriever = FakeRetriever()
    answerer = FakeAnswerer()
    hybrid = HybridRAGOrchestrator(
        retriever=retriever,
        retrieval_runtime=FakeRetrievalRuntime(),
        answerer=answerer,
    )

    result = hybrid.chat(question="Bạch Đằng năm 938 có ý nghĩa gì?", final_k=3)

    assert result["inference_mode"] == "hybrid_rag"
    assert result["answer_provenance"]["mode"] == "hybrid_rag"
    assert result["answer_provenance"]["research_generation_calls"] == 0
    assert result["answer_provenance"]["evidence_generation_calls"] == 0
    assert retriever.calls == [{"question": "Bạch Đằng năm 938 có ý nghĩa gì?", "final_k": 3}]
    assert answerer.calls[0]["contexts"][0]["chunk_id"] == "ev_01"
    assert "agent:research" not in result["tool_trace"]
    assert "agent:evidence_critic" not in result["tool_trace"]


def test_chat_request_accepts_two_modes_and_rejects_legacy_names():
    conversation_id = uuid4()

    assert ChatRequest(conversation_id=conversation_id, question="Hỏi?", mode="hybrid_rag").mode == "hybrid_rag"
    assert ChatRequest(conversation_id=conversation_id, question="Hỏi?", mode="agentic_rag").mode == "agentic_rag"
    with pytest.raises(ValueError):
        ChatRequest(conversation_id=conversation_id, question="Hỏi?", mode="deterministic")


def test_omitted_mode_uses_configured_default(monkeypatch):
    monkeypatch.setattr(routes.settings, "default_inference_mode", "hybrid_rag")

    assert _resolve_inference_mode(SimpleNamespace(mode=None)) == "hybrid_rag"


def test_request_summary_records_selected_mode(caplog, tmp_path):
    owner_id = "test-client"
    store = ConversationStore(tmp_path / "chat.db")
    conversation = store.create_conversation(owner_id, "Modes")
    payload = SimpleNamespace(
        conversation_id=conversation["id"],
        question="Bạch Đằng?",
        final_k=6,
        mode="hybrid_rag",
    )
    generator = SimpleNamespace(
        max_history_messages=6,
        retrieval_history_messages=4,
        chat=lambda **_: {
            "answer": "Trả lời.",
            "status": "ok",
            "source_ids": [],
            "source_chunks": [],
            "retrieval": {},
            "analysis": {},
            "tool_trace": [],
            "answer_provenance": {"history_generation_calls": 0},
            "inference_mode": "hybrid_rag",
        },
    )
    service = SimpleNamespace(chunk_by_id={}, deployment_id="qwen3-test")

    with caplog.at_level("INFO"):
        result = _execute_chat(store, generator, service, owner_id, payload, "req-mode", "hybrid_rag")

    assert result["inference_mode"] == "hybrid_rag"
    summaries = [record for record in caplog.records if record.message == "REQUEST_SUMMARY"]
    assert getattr(summaries[0], "inference_mode") == "hybrid_rag"
