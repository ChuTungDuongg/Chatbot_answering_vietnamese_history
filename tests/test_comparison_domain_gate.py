from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.agents.evidence_agent import EvidenceCriticAgent, EvidenceModelContractError
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.orchestrator import AgentOrchestrator, HybridRAGOrchestrator
from app.agents.research_agent import ResearchAgent
from app.rag.research_runtime import ResearchRetrievalRuntime
from app.rag.retrieval import extract_comparison_targets
from app.tools.evidence_tools import InspectEvidenceTool, RetrieveEvidenceTool, SessionEvidenceStore
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolRegistry


QUESTION = "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ."
A_ID = "ev_august"
B_ID = "ev_dien_bien"
INCIDENTAL_ID = "ev_1975_mentions_both"
A_CLAIM = "Cách mạng Tháng Tám năm 1945 giành chính quyền và lập nên nước Việt Nam Dân chủ Cộng hòa."
B_CLAIM = "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp ký Hiệp định Genève và chấm dứt chiến tranh ở Đông Dương."
INCIDENTAL_CLAIM = (
    "Lê Duẩn nhắc đến Tổng khởi nghĩa tháng Tám, chiến thắng Điện Biên Phủ "
    "và chiến thắng mùa xuân 1975 như ba mốc chói lọi."
)


class FakeEvidenceModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_json(self, *, adapter, messages, **kwargs):
        assert adapter == "evidence"
        self.calls.append(json.loads(messages[1]["content"]))
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        output = self.outputs[index]
        if isinstance(output, Exception):
            raise output
        return output


class FakeHistoryRuntime:
    def __init__(self):
        self.calls = []

    def generate_text(self, *, adapter, messages, **kwargs):
        assert adapter == "history"
        self.calls.append(messages)
        return (
            f"Nguồn được dùng: [{A_ID}] [{B_ID}]\n\n"
            "Trả lời:\nCách mạng Tháng Tám và Điện Biên Phủ đều là mốc lớn, "
            "nhưng một bên là cách mạng giành chính quyền, một bên là thắng lợi quân sự."
        )


def _compare_candidates():
    return [
        {
            "chunk_id": f"ev_august_{index}",
            "title": "Cách mạng Tháng Tám",
            "text": A_CLAIM,
            "source_kind": "history",
            "final_retrieval_score": 0.95 - index * 0.01,
        }
        for index in range(4)
    ] + [
        {
            "chunk_id": B_ID,
            "title": "Chiến thắng Điện Biên Phủ",
            "text": B_CLAIM,
            "source_kind": "history",
            "final_retrieval_score": 0.7,
        },
        {
            "chunk_id": A_ID,
            "title": "Cách mạng Tháng Tám",
            "text": A_CLAIM,
            "source_kind": "history",
            "final_retrieval_score": 0.99,
        },
    ]


def test_extracts_generic_comparison_targets():
    assert extract_comparison_targets(QUESTION) == [
        "Cách mạng Tháng Tám",
        "chiến thắng Điện Biên Phủ",
    ]


def test_compare_budget_reserves_model_visible_evidence_for_both_targets():
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {"evidence_id": A_ID, "relevance": 1.0, "claims": [A_CLAIM], "compressed_text": A_CLAIM},
            {"evidence_id": B_ID, "relevance": 1.0, "claims": [B_CLAIM], "compressed_text": B_CLAIM},
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Hai phía đều có bằng chứng.",
    }])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime, max_contexts=2).compress(
        QUESTION,
        _compare_candidates(),
        final_k=2,
    )

    visible_ids = [item["evidence_id"] for item in runtime.calls[0]["evidence"]]
    assert A_ID in visible_ids
    assert B_ID in visible_ids
    assert critique.comparison_targets == ["Cách mạng Tháng Tám", "chiến thắng Điện Biên Phủ"]
    assert critique.target_a_model_visible_count >= 1
    assert critique.target_b_model_visible_count >= 1


def test_compare_coverage_deterministically_adds_missing_target_without_retry():
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {"evidence_id": A_ID, "relevance": 1.0, "claims": [A_CLAIM], "compressed_text": A_CLAIM},
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Chỉ có phía A.",
    }])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        QUESTION,
        [
            {"chunk_id": A_ID, "title": "Cách mạng Tháng Tám", "text": A_CLAIM, "source_kind": "history"},
            {"chunk_id": B_ID, "title": "Chiến thắng Điện Biên Phủ", "text": B_CLAIM, "source_kind": "history"},
        ],
        final_k=4,
    )

    assert len(runtime.calls) == 1
    assert critique.repair_path == "deterministic_semantic_guard"
    assert critique.selected_ids == [A_ID, B_ID]
    assert critique.comparison_target_coverage == {
        "Cách mạng Tháng Tám": True,
        "chiến thắng Điện Biên Phủ": True,
    }
    assert critique.comparison_target_map == {A_ID: "target_a", B_ID: "target_b"}
    assert critique.target_a_selected_evidence == [A_ID]
    assert critique.target_b_selected_evidence == [B_ID]
    assert [context["chunk_id"] for context in contexts] == [A_ID, B_ID]
    assert [context["comparison_target"] for context in contexts] == ["target_a", "target_b"]


def test_compare_parse_failure_uses_source_local_deterministic_evidence():
    runtime = FakeEvidenceModel([ValueError("Model output does not contain a JSON object.")])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        QUESTION,
        [
            {"chunk_id": A_ID, "title": "Cách mạng Tháng Tám", "text": A_CLAIM, "source_kind": "history"},
            {"chunk_id": B_ID, "title": "Chiến thắng Điện Biên Phủ", "text": B_CLAIM, "source_kind": "history"},
        ],
        final_k=4,
    )

    assert len(runtime.calls) == 1
    assert critique.repair_path == "deterministic_parse_failure"
    assert critique.selected_ids == [A_ID, B_ID]
    assert critique.selected_evidence[0].claims == [A_CLAIM]
    assert critique.selected_evidence[1].claims == [B_CLAIM]
    assert [context["chunk_id"] for context in contexts] == [A_ID, B_ID]


def test_compare_parse_failure_prefers_title_match_over_incidental_mentions():
    runtime = FakeEvidenceModel([ValueError("Model output does not contain a JSON object.")])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        QUESTION,
        [
            {
                "chunk_id": INCIDENTAL_ID,
                "title": "Sự kiện 30 tháng 4 năm 1975",
                "text": INCIDENTAL_CLAIM,
                "source_kind": "history",
                "final_retrieval_score": 0.99,
            },
            {
                "chunk_id": A_ID,
                "title": "Cách mạng Tháng Tám",
                "text": A_CLAIM,
                "source_kind": "history",
                "final_retrieval_score": 0.4,
            },
            {
                "chunk_id": B_ID,
                "title": "Chiến dịch Điện Biên Phủ",
                "text": B_CLAIM,
                "source_kind": "history",
                "final_retrieval_score": 0.4,
            },
        ],
        final_k=4,
    )

    assert critique.repair_path == "deterministic_parse_failure"
    assert critique.selected_ids == [A_ID, B_ID]


def test_compare_guard_replaces_incidental_both_sides_with_title_matched_targets():
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": INCIDENTAL_ID,
                "relevance": 1.0,
                "claims": [INCIDENTAL_CLAIM],
                "compressed_text": INCIDENTAL_CLAIM,
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Một nguồn nhắc cả hai mốc.",
    }])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        QUESTION,
        [
            {
                "chunk_id": INCIDENTAL_ID,
                "title": "Sự kiện 30 tháng 4 năm 1975",
                "text": INCIDENTAL_CLAIM,
                "source_kind": "history",
                "final_retrieval_score": 0.99,
            },
            {
                "chunk_id": A_ID,
                "title": "Cách mạng Tháng Tám",
                "text": A_CLAIM,
                "source_kind": "history",
                "final_retrieval_score": 0.4,
            },
            {
                "chunk_id": B_ID,
                "title": "Chiến dịch Điện Biên Phủ",
                "text": B_CLAIM,
                "source_kind": "history",
                "final_retrieval_score": 0.4,
            },
        ],
        final_k=4,
    )

    assert len(runtime.calls) == 1
    assert critique.repair_path == "deterministic_semantic_guard"
    assert critique.selected_ids == [A_ID, B_ID]
    assert critique.comparison_target_coverage == {
        "Cách mạng Tháng Tám": True,
        "chiến thắng Điện Biên Phủ": True,
    }


def test_compare_guard_replaces_navigation_heavy_target_chunk():
    noisy_claim = (
        "Xem thêm Tổng khởi nghĩa Hà Nội Chú thích Tham khảo Liên kết ngoài "
        "Những sự thật về Cách mạng Tháng Tám."
    )
    clean_a = "Cách mạng Tháng Tám năm 1945 là cuộc khởi nghĩa giành chính quyền."
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {"evidence_id": "ev_august_noisy", "relevance": 0.8, "claims": [noisy_claim], "compressed_text": noisy_claim},
            {"evidence_id": B_ID, "relevance": 1.0, "claims": [B_CLAIM], "compressed_text": B_CLAIM},
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Có hai phía.",
    }])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        QUESTION,
        [
            {"chunk_id": "ev_august_noisy", "title": "Cách mạng Tháng Tám", "text": noisy_claim, "source_kind": "history", "final_retrieval_score": 0.8},
            {"chunk_id": A_ID, "title": "Cách mạng Tháng Tám", "text": clean_a, "source_kind": "history", "final_retrieval_score": 0.7},
            {"chunk_id": B_ID, "title": "Chiến thắng Điện Biên Phủ", "text": B_CLAIM, "source_kind": "history", "final_retrieval_score": 0.7},
        ],
        final_k=4,
    )

    assert critique.repair_path == "deterministic_semantic_guard"
    assert A_ID in critique.selected_ids
    assert "ev_august_noisy" not in critique.selected_ids
    assert [context["comparison_target"] for context in contexts] == ["target_a", "target_b"]


def test_compare_claim_recovery_uses_exact_source_local_spans():
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": A_ID,
                "relevance": 1.0,
                "claims": ["Cách mạng Tháng Tám giành chính quyền năm 1945."],
                "compressed_text": "Cách mạng Tháng Tám giành chính quyền năm 1945.",
            },
            {
                "evidence_id": B_ID,
                "relevance": 1.0,
                "claims": ["Điện Biên Phủ khiến Pháp phải ký Genève."],
                "compressed_text": "Điện Biên Phủ khiến Pháp phải ký Genève.",
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Có đủ hai phía.",
    }])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        QUESTION,
        [
            {"chunk_id": A_ID, "title": "Cách mạng Tháng Tám", "text": A_CLAIM, "source_kind": "history"},
            {"chunk_id": B_ID, "title": "Chiến thắng Điện Biên Phủ", "text": B_CLAIM, "source_kind": "history"},
        ],
        final_k=4,
    )

    assert critique.repair_path == "deterministic"
    assert critique.selected_evidence[0].claims == [A_CLAIM]
    assert critique.selected_evidence[1].claims == [B_CLAIM]


def test_compare_source_local_contract_rejects_cross_source_compression():
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": A_ID,
                "relevance": 1.0,
                "claims": [A_CLAIM],
                "compressed_text": f"{A_CLAIM} {B_CLAIM}",
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Bị trộn nguồn.",
    }])

    with pytest.raises(EvidenceModelContractError) as exc:
        EvidenceCriticAgent(model_runtime=runtime).compress(
            QUESTION,
            [
                {"chunk_id": A_ID, "title": "Cách mạng Tháng Tám", "text": A_CLAIM, "source_kind": "history"},
                {"chunk_id": B_ID, "title": "Chiến thắng Điện Biên Phủ", "text": B_CLAIM, "source_kind": "history"},
            ],
            final_k=4,
        )

    assert exc.value.code == "cross_id_compressed_text"
    assert len(runtime.calls) == 1


def test_compare_history_prompt_owns_synthesis_shape():
    runtime = FakeHistoryRuntime()
    answerer = HistoryAnswererAgent(model_runtime=runtime)

    result = answerer.answer(
        question=QUESTION,
        contexts=[
            {"chunk_id": A_ID, "title": "Cách mạng Tháng Tám", "text": A_CLAIM},
            {"chunk_id": B_ID, "title": "Chiến thắng Điện Biên Phủ", "text": B_CLAIM},
        ],
        analysis={"facet": "compare", "facets": ["compare"]},
        tool_trace=[],
        answer_depth="deep",
    )

    prompt = runtime.calls[0][0]["content"]
    assert "Trả lời theo hướng so sánh sâu" in prompt
    assert "điểm giống nhau" in prompt
    assert set(result["source_ids"]) == {A_ID, B_ID}


class GateRetriever:
    final_context_k = 6

    def __init__(self, gate_result):
        self.gate_result = gate_result
        self.retrieve_calls = []

    def classify_question(self, question):
        return {
            "is_ood": self.gate_result == "out_of_domain",
            "ood_reason": "explicit_ood" if self.gate_result == "out_of_domain" else "",
            "domain_gate_result": self.gate_result,
            "domain_gate_reason": self.gate_result,
            "history_anchor": 0.1,
            "ood_anchor": 0.9,
            "domain_margin": -0.8,
            "intent": {"history_anchor": 0.1, "ood_anchor": 0.9, "margin": -0.8},
        }

    def retrieve(self, question, final_k):
        self.retrieve_calls.append((question, final_k))
        return {"final_context": [], "is_ood": False, "ood_reason": "", "tool_trace": []}

    def analyze_question(self, question):
        return {"question": question, "facet": "general", "facets": ["general"]}

    def context_title_diversity(self, contexts):
        return 0.0


class CountingAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": "called", "status": "ok", "source_ids": [], "source_chunks": [], "answer_provenance": {}}


def _agentic_for_gate(retriever):
    store = SessionEvidenceStore()
    registry = ToolRegistry()
    registry.register(SearchHistoryTool(retriever))
    registry.register(RetrieveEvidenceTool(store))
    registry.register(InspectEvidenceTool(store))
    research = ResearchAgent(
        registry=registry,
        evidence_store=store,
        retrieval_runtime=ResearchRetrievalRuntime(SimpleNamespace(config={}), retriever),
        model_runtime=None,
    )
    return AgentOrchestrator(
        research_agent=research,
        evidence_agent=EvidenceCriticAgent(model_runtime=None),
        answerer=CountingAnswerer(),
    )


@pytest.mark.parametrize("question", [
    "Tôi bị đau họng quá phải làm sao?",
    "Viết tôi đoạn code hello world đi",
])
def test_ood_gate_short_circuits_hybrid_and_agentic(question):
    hybrid_retriever = GateRetriever("out_of_domain")
    hybrid_answerer = CountingAnswerer()
    hybrid = HybridRAGOrchestrator(
        retriever=hybrid_retriever,
        retrieval_runtime=SimpleNamespace(
            max_history_messages=6,
            retrieval_history_messages=4,
            normalize_history=lambda history, current_question=None: history or [],
            build_retrieval_question=lambda question, history: (question, bool(history)),
        ),
        answerer=hybrid_answerer,
    )
    hybrid_result = hybrid.chat(question)

    agentic_retriever = GateRetriever("out_of_domain")
    agentic_result = asyncio.run(_agentic_for_gate(agentic_retriever).run(question=question, final_k=6))

    assert hybrid_retriever.retrieve_calls == []
    assert hybrid_answerer.calls == []
    assert hybrid_result["answer_provenance"]["total_llm_calls"] == 0
    assert agentic_retriever.retrieve_calls == []
    assert agentic_result["research_debug"]["generation_calls"] == 0
    assert agentic_result["evidence_debug"]["generation_calls"] == 0
    assert agentic_result["history_debug"]["generation_calls"] == 0


def test_meta_gate_short_circuits_without_retrieval():
    retriever = GateRetriever("meta")
    answerer = CountingAnswerer()
    hybrid = HybridRAGOrchestrator(
        retriever=retriever,
        retrieval_runtime=SimpleNamespace(
            max_history_messages=6,
            retrieval_history_messages=4,
            normalize_history=lambda history, current_question=None: history or [],
            build_retrieval_question=lambda question, history: (question, bool(history)),
        ),
        answerer=answerer,
    )

    result = hybrid.chat("Bạn có thể làm gì?")

    assert retriever.retrieve_calls == []
    assert answerer.calls == []
    assert result["retrieval"]["domain_gate_result"] == "meta"
