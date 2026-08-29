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
BACH_DANG_ID = "hf_wikipedia_trận_bạch_đằng_938_0002_d5f8e1eedf68"
BACH_DANG_FOREIGN_ID = "hf_wikipedia_trận_bạch_đằng_938_0003_cf59d98b5f8d"
LATE_RELEVANT_PASSAGE = (
    "Ý nghĩa của chiến thắng Bạch Đằng năm 938 là chấm dứt hơn 1000 năm Bắc thuộc và mở ra thời kỳ độc lập tự chủ."
)
BACH_DANG_FOREIGN_PASSAGE = (
    "Đến khi tiến binh trên Bạch Đằng, quả thấy trên không có tiếng xe ngựa, trận ấy quả được đại tiệp."
)


class RegressionRetriever:
    def __init__(self):
        self.queries = []
        prefix = "Đoạn đầu mô tả mưu kế, bãi cọc và quy luật thủy triều. " * 90
        self.contexts = [
            {
                "chunk_id": BACH_DANG_ID,
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
                "chunk_id": BACH_DANG_FOREIGN_ID,
                "title": "Trận Bạch Đằng (938)",
                "text": BACH_DANG_FOREIGN_PASSAGE,
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
        self.history_outputs = [
            (
                f"Nguồn được dùng: [{BACH_DANG_ID}] [{BACH_DANG_FOREIGN_ID}]\n\n"
                "Trả lời:\nChiến thắng Bạch Đằng năm 938 chấm dứt hơn 1000 năm Bắc thuộc "
                "và mở ra thời kỳ độc lập tự chủ. Chi tiết trận ấy được đại tiệp cho thấy thắng lợi "
                "quân sự đã tạo nền cho bước chuyển chính trị lâu dài của lịch sử Việt Nam."
            ),
        ]

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
                    "arguments": {"ids": [BACH_DANG_ID]},
                }
            return {"action": "finish", "sufficient": True, "missing_information": []}

        assert adapter == "evidence"
        request = json.loads(messages[1]["content"])
        visible = {item["evidence_id"]: item["text"] for item in request["evidence"]}
        assert LATE_RELEVANT_PASSAGE in visible[BACH_DANG_ID]
        return {
            "status": "sufficient",
            "selected_evidence": [
                {
                    "evidence_id": BACH_DANG_ID,
                    "relevance": 1.0,
                    "claims": [LATE_RELEVANT_PASSAGE, BACH_DANG_FOREIGN_PASSAGE],
                    "compressed_text": f"{LATE_RELEVANT_PASSAGE} {BACH_DANG_FOREIGN_PASSAGE}",
                }
            ],
            "conflicts": [],
            "missing_information": [],
            "summary": "Evidence đã chọn trả lời đúng facet được hỏi.",
        }

    def generate_text(self, *, adapter, messages, **kwargs):
        self.calls.append({"adapter": adapter, "messages": messages})
        assert adapter == "history"
        index = len([call for call in self.calls if call["adapter"] == "history"]) - 1
        prompt = messages[0]["content"]
        assert f"[{BACH_DANG_ID}]" in prompt
        assert f"[{BACH_DANG_FOREIGN_ID}]" in prompt
        assert LATE_RELEVANT_PASSAGE in prompt
        assert BACH_DANG_FOREIGN_PASSAGE in prompt
        return self.history_outputs[min(index, len(self.history_outputs) - 1)]


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
    assert result["evidence_critique"]["selected_ids"] == [BACH_DANG_ID, BACH_DANG_FOREIGN_ID]
    assert result["source_ids"] == [BACH_DANG_ID, BACH_DANG_FOREIGN_ID]
    assert result["answer_provenance"]["source"] == "history_adapter"
    assert result["answer_provenance"]["history_generation_calls"] == 1
    assert result["answer_provenance"]["history_retry_used"] is False
    assert result["answer_provenance"]["guard_override"] is False
    assert [call["adapter"] for call in role_runtime.calls].count("history") == 1
    assert [call["adapter"] for call in role_runtime.calls].count("evidence") == 1
    assert "history:deep_structured_expansion" not in result["tool_trace"]
    assert result["answer_provenance"]["structured_expansion_used"] is False
    assert "Câu trả lời được tạo từ evidence" not in result["answer"]

    debug = _build_debug(result)
    assert debug["research"]["tools"][0]["arguments"]["query"] == QUESTION
    assert debug["evidence"]["status"] == "sufficient"
    assert debug["evidence"]["repair_path"] == "deterministic_rebucket"
    assert debug["history"]["input_evidence_ids"] == [BACH_DANG_ID, BACH_DANG_FOREIGN_ID]
    assert LATE_RELEVANT_PASSAGE in debug["history"]["input_evidence_preview"][0]["text_preview"]
    assert debug["answer_provenance"]["source"] == "history_adapter"
