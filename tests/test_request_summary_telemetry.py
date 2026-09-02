from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from app.agents.evidence.agent import EvidenceModelContractError
from app.api.routes import _execute_chat
from app.chat.store import ConversationStore


class SuccessfulGenerator:
    max_history_messages = 6
    retrieval_history_messages = 4

    def chat(self, **_kwargs):
        return {
            "answer": "Trả lời ngắn.",
            "status": "ok",
            "source_ids": [],
            "source_chunks": [],
            "retrieval": {},
            "analysis": {},
            "tool_trace": [],
            "answer_provenance": {"history_generation_calls": 0},
        }


class FailingGenerator(SuccessfulGenerator):
    def chat(self, **_kwargs):
        raise EvidenceModelContractError("full private evidence text", code="claim_not_extractive")


def _store(tmp_path):
    owner_id = "test-client"
    store = ConversationStore(tmp_path / "chat.db")
    conversation = store.create_conversation(owner_id, "Telemetry")
    return store, owner_id, conversation["id"]


def _service():
    return SimpleNamespace(
        chunk_by_id={},
        deployment_id="qwen3-test",
    )


def test_request_summary_success_is_emitted_without_prompt(caplog, tmp_path):
    store, owner_id, conversation_id = _store(tmp_path)
    payload = SimpleNamespace(
        conversation_id=conversation_id,
        question="VERY_PRIVATE_PROMPT",
        final_k=6,
    )

    with caplog.at_level(logging.INFO):
        _execute_chat(store, SuccessfulGenerator(), _service(), owner_id, payload, "req-success", "agentic_rag")

    summaries = [record for record in caplog.records if record.message == "REQUEST_SUMMARY"]
    assert len(summaries) == 1
    assert getattr(summaries[0], "result") == "success"
    assert getattr(summaries[0], "inference_mode") == "three_llm"
    assert getattr(summaries[0], "deployment_id") == "qwen3-test"
    assert "VERY_PRIVATE_PROMPT" not in caplog.text


def test_request_summary_failure_is_emitted_without_evidence_text(caplog, tmp_path):
    store, owner_id, conversation_id = _store(tmp_path)
    payload = SimpleNamespace(
        conversation_id=conversation_id,
        question="VERY_PRIVATE_PROMPT",
        final_k=6,
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(EvidenceModelContractError):
            _execute_chat(store, FailingGenerator(), _service(), owner_id, payload, "req-fail", "agentic_rag")

    summaries = [record for record in caplog.records if record.message == "REQUEST_SUMMARY"]
    assert len(summaries) == 1
    assert getattr(summaries[0], "result") == "failed"
    assert getattr(summaries[0], "failed_stage") == "evidence"
    assert getattr(summaries[0], "failure_code") == "claim_not_extractive"
    assert getattr(summaries[0], "history_generation_calls") == 0
    assert "VERY_PRIVATE_PROMPT" not in caplog.text
    assert "full private evidence text" not in caplog.text
