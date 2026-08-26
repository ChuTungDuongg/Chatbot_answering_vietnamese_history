from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.research_agent import ResearchAgent
from app.chat.attachments import TemporaryCorpusRetriever
from app.chat.store import ConversationStore
from app.tools.attachment_search import SearchUploadedDocumentsTool
from app.tools.evidence_tools import SessionEvidenceStore
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolExecutionContext, ToolRegistry


class FakeTemporaryRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, owner_id, conversation_id, question, top_k):
        self.calls.append(
            {
                "owner_id": owner_id,
                "conversation_id": conversation_id,
                "question": question,
                "top_k": top_k,
            }
        )
        return [
            {
                "chunk_id": f"temp:{conversation_id}:1:0",
                "attachment_id": f"attachment-{conversation_id}",
                "title": f"{conversation_id}.pdf - trang 1",
                "text": f"Nội dung riêng của {owner_id}/{conversation_id}",
                "page_number": 1,
                "temporary_dense_score": 0.95,
                "reranker_score": 0.9,
            }
        ]


class FakeGlobalRetriever:
    def retrieve(self, question, final_k):
        return {
            "question": question,
            "final_context": [],
            "is_ood": True,
            "ood_reason": "Không có trong corpus toàn cục.",
        }

    def analyze_question(self, question):
        return {"normalized_question": question}


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeGenerator:
    def __init__(self, retriever):
        self.retriever = retriever
        self.prompt_builder = SimpleNamespace(max_history_messages=6)
        self.retrieval_history_messages = 6

    def normalize_history(self, history, current_question):
        return history or []

    def build_retrieval_question(self, question, history):
        return question, bool(history)

    def temporary_context_is_relevant(self, question, contexts):
        return bool(contexts)

    def answer_from_retrieval(self, *, question, retrieval, history=None):
        contexts = retrieval["final_context"]
        return {
            "question": question,
            "answer": contexts[0]["text"] if contexts else "Không đủ dữ kiện.",
            "status": "ok" if contexts else "insufficient",
            "source_ids": [row["chunk_id"] for row in contexts],
            "source_chunks": contexts,
            "retrieval": retrieval,
            "tool_trace": [],
        }


def build_orchestrator(temporary_retriever):
    global_retriever = FakeGlobalRetriever()
    generator = FakeGenerator(global_retriever)
    evidence_store = SessionEvidenceStore()
    registry = ToolRegistry()
    registry.register(SearchHistoryTool(global_retriever))
    registry.register(SearchUploadedDocumentsTool(temporary_retriever))
    research_agent = ResearchAgent(
        registry=registry,
        evidence_store=evidence_store,
        generator=generator,
    )
    return AgentOrchestrator(
        research_agent=research_agent,
        evidence_agent=EvidenceCriticAgent(),
        answerer=HistoryAnswererAgent(generator),
    )


def test_attachment_tool_keeps_scope_out_of_model_schema():
    retriever = FakeTemporaryRetriever()
    registry = ToolRegistry()
    registry.register(SearchUploadedDocumentsTool(retriever))

    schema = registry.describe()[0]["input_schema"]
    assert set(schema["properties"]) == {"query", "top_k"}

    result, record = asyncio.run(
        registry.call(
            "search_uploaded_documents",
            {"query": "Theo tài liệu này", "top_k": 3},
            context=ToolExecutionContext(
                owner_id="owner-a",
                conversation_id="conversation-a",
                session_id="session-a",
            ),
        )
    )

    assert record.error is None
    assert record.arguments == {"query": "Theo tài liệu này", "top_k": 3}
    assert result[0]["source_kind"] == "attachment"
    assert retriever.calls[0]["owner_id"] == "owner-a"
    assert retriever.calls[0]["conversation_id"] == "conversation-a"


def test_agent_flow_uses_only_current_conversation_attachments():
    temporary_retriever = FakeTemporaryRetriever()
    orchestrator = build_orchestrator(temporary_retriever)

    first = asyncio.run(
        orchestrator.run(
            question="Theo tài liệu này, nội dung chính là gì?",
            final_k=4,
            owner_id="owner-a",
            conversation_id="conversation-a",
        )
    )
    second = asyncio.run(
        orchestrator.run(
            question="Hãy đọc file PDF vừa tải lên.",
            final_k=4,
            owner_id="owner-b",
            conversation_id="conversation-b",
        )
    )

    assert first["answer"] == "Nội dung riêng của owner-a/conversation-a"
    assert second["answer"] == "Nội dung riêng của owner-b/conversation-b"
    assert first["retrieval"]["temporary_context_count"] == 1
    assert first["source_chunks"][0]["page_number"] == 1
    assert first["source_chunks"][0]["attachment_id"] == "attachment-conversation-a"
    assert [call["conversation_id"] for call in temporary_retriever.calls] == [
        "conversation-a",
        "conversation-b",
    ]
    assert orchestrator.research_agent.evidence_store._sessions == {}


def test_real_temporary_retriever_isolates_sqlite_conversations(tmp_path):
    store = ConversationStore(tmp_path / "chat.db")
    first_conversation = store.create_conversation("owner-a")
    second_conversation = store.create_conversation("owner-a")

    for conversation, text in (
        (first_conversation, "Tài liệu thứ nhất"),
        (second_conversation, "Tài liệu thứ hai"),
    ):
        attachment = store.create_attachment(
            owner_id="owner-a",
            conversation_id=conversation["id"],
            filename=f"{conversation['id']}.pdf",
            mime_type="application/pdf",
            size_bytes=100,
        )
        store.replace_temporary_chunks(
            owner_id="owner-a",
            conversation_id=conversation["id"],
            attachment_id=attachment["id"],
            chunks=[
                {
                    "chunk_id": f"temp:{attachment['id']}:1:0",
                    "title": attachment["filename"],
                    "text": text,
                    "page_number": 1,
                    "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                }
            ],
        )
        store.update_attachment_status(
            owner_id="owner-a",
            conversation_id=conversation["id"],
            attachment_id=attachment["id"],
            status="ready",
            chunk_count=1,
        )

    rag_service = SimpleNamespace(embedder=FakeEmbedder(), reranker=None)
    registry = ToolRegistry()
    registry.register(
        SearchUploadedDocumentsTool(TemporaryCorpusRetriever(store, rag_service))
    )

    result, record = asyncio.run(
        registry.call(
            "search_uploaded_documents",
            {"query": "Đọc tài liệu", "top_k": 4},
            context=ToolExecutionContext(
                owner_id="owner-a",
                conversation_id=first_conversation["id"],
            ),
        )
    )
    wrong_owner_result, wrong_owner_record = asyncio.run(
        registry.call(
            "search_uploaded_documents",
            {"query": "Đọc tài liệu", "top_k": 4},
            context=ToolExecutionContext(
                owner_id="owner-b",
                conversation_id=first_conversation["id"],
            ),
        )
    )

    assert record.error is None
    assert [row["text"] for row in result] == ["Tài liệu thứ nhất"]
    assert wrong_owner_record.error is None
    assert wrong_owner_result == []
