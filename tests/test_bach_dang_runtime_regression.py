from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.research_agent import ResearchAgent
from app.api.routes import _build_debug
from app.rag.research_runtime import ResearchRetrievalRuntime
from app.tools.evidence_tools import (
    InspectEvidenceTool,
    RetrieveEvidenceTool,
    SessionEvidenceStore,
)
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolRegistry


QUESTION = "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?"
LATE_RELEVANT_PASSAGE = (
    "Ý nghĩa của chiến thắng Bạch Đằng năm 938 được trình bày trực tiếp trong đoạn này."
)


class RegressionRetriever:
    def __init__(self):
        self.queries = []
        prefix = "Đoạn đầu mô tả mưu kế, bãi cọc và quy luật thủy triều. " * 90
        self.contexts = [
            {
                "chunk_id": "bach_dang_significance",
                "title": "Trận Bạch Đằng (938)",
                "text": prefix + LATE_RELEVANT_PASSAGE,
                "source_kind": "history",
                "best_dense_score": 0.85,
                "best_bm25_score": 11.6,
                "rrf_score": 0.06,
                "reranker_score": 0.99,
                "final_retrieval_score": 1.08,
            },
            {
                "chunk_id": "bach_dang_distractor",
                "title": "Bãi cọc Bạch Đằng",
                "text": "Đoạn nhiễu chỉ mô tả cấu tạo của bãi cọc.",
                "source_kind": "history",
                "final_retrieval_score": 0.7,
            },
        ]

    def retrieve(self, question, final_k):
        self.queries.append({"query": question, "top_k": final_k})
        return {"question": question, "final_context": self.contexts[:final_k]}

    def analyze_question(self, question):
        return {"normalized_question": question, "facets": ["significance"]}

    def classify_question(self, question):
        return {"is_ood": False, "ood_reason": "", "intent": "history"}


class FakeSharedRoleRuntime:
    def __init__(self):
        self.calls = []

    def generate_json(self, *, adapter, messages, **kwargs):
        self.calls.append({"adapter": adapter, "messages": messages})
        if adapter == "research":
            state = json.loads(messages[1]["content"])
            if state["step"] == 1:
                return {
                    "action": "tool",
                    "tool_name": "search_history",
                    "arguments": {"query": state["retrieval_question"], "top_k": 8},
                }
            if state["step"] == 2:
                return {
                    "action": "tool",
                    "tool_name": "inspect_evidence",
                    "arguments": {"ids": ["bach_dang_significance"]},
                }
            return {"action": "finish", "sufficient": True, "missing_information": []}

        assert adapter == "evidence"
        request = json.loads(messages[1]["content"])
        visible = {item["evidence_id"]: item["text"] for item in request["evidence"]}
        assert LATE_RELEVANT_PASSAGE in visible["bach_dang_significance"]
        return {
            "status": "sufficient",
            "selected_evidence": [
                {
                    "evidence_id": "bach_dang_significance",
                    "relevance": 1.0,
                    "claims": [LATE_RELEVANT_PASSAGE],
                    "compressed_text": LATE_RELEVANT_PASSAGE,
                }
            ],
            "conflicts": [],
            "missing_information": [],
            "summary": "Evidence đã chọn trả lời đúng facet được hỏi.",
        }

    def generate_text(self, *, adapter, messages, **kwargs):
        self.calls.append({"adapter": adapter, "messages": messages})
        assert adapter == "history"
        prompt = messages[0]["content"]
        assert "[bach_dang_significance]" in prompt
        assert LATE_RELEVANT_PASSAGE in prompt
        assert "bach_dang_distractor" not in prompt
        return (
            "Nguồn được dùng: [bach_dang_significance]\n\n"
            "Trả lời:\nCâu trả lời được tạo từ evidence đã chọn."
        )


def test_bach_dang_pipeline_preserves_significance_evidence_to_history_adapter():
    retriever = RegressionRetriever()
    role_runtime = FakeSharedRoleRuntime()
    evidence_store = SessionEvidenceStore()
    registry = ToolRegistry()
    registry.register(SearchHistoryTool(retriever))
    registry.register(RetrieveEvidenceTool(evidence_store))
    registry.register(InspectEvidenceTool(evidence_store))
    retrieval_runtime = ResearchRetrievalRuntime(
        SimpleNamespace(config={"prompt": {}, "generation": {}}),
        retriever,
    )
    orchestrator = AgentOrchestrator(
        research_agent=ResearchAgent(
            registry=registry,
            evidence_store=evidence_store,
            retrieval_runtime=retrieval_runtime,
            model_runtime=role_runtime,
            max_steps=4,
        ),
        evidence_agent=EvidenceCriticAgent(model_runtime=role_runtime),
        answerer=HistoryAnswererAgent(model_runtime=role_runtime),
    )

    result = asyncio.run(orchestrator.run(question=QUESTION, final_k=8))

    assert retriever.queries[0]["query"] == QUESTION
    assert all(term in retriever.queries[0]["query"] for term in ("Bạch Đằng", "938", "ý nghĩa"))
    assert result["evidence_critique"]["selected_ids"] == ["bach_dang_significance"]
    assert result["source_ids"] == ["bach_dang_significance"]
    assert result["answer_provenance"]["source"] == "history_adapter"
    assert result["answer_provenance"]["history_generation_calls"] == 1
    assert result["answer_provenance"]["guard_override"] is False
    assert [call["adapter"] for call in role_runtime.calls].count("history") == 1
    assert not any("structured" in item or "repair" in item for item in result["tool_trace"])

    debug = _build_debug(result)
    assert debug["research"]["tools"][0]["arguments"]["query"] == QUESTION
    assert debug["evidence"]["status"] == "sufficient"
    assert debug["history"]["input_evidence_ids"] == ["bach_dang_significance"]
    assert debug["answer_provenance"]["source"] == "history_adapter"
