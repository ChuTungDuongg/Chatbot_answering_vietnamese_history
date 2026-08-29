from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.agents.evidence_agent import EvidenceCriticAgent
from app.agents.history_answerer import HistoryAnswererAgent
from app.agents.orchestrator import AgentOrchestrator, HybridRAGOrchestrator
from app.agents.research_agent import ResearchAgent
from app.agents.research_agent import _external_research_reason, _select_wikipedia_candidate, needs_external_research
from app.tools.evidence_tools import InspectEvidenceTool, RetrieveEvidenceTool, SessionEvidenceStore
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolRegistry
from app.tools.wikipedia import FetchWikipediaPageInput, SearchWikipediaInput


TRAN_WEAK_ID = "ev_buddhism"
TRAN_POLITICAL_ID = "ev_tran_political"
TRAN_SOCIAL_ID = "ev_tran_social"
TRAN_MILITARY_ID = "ev_tran_military"


class FakeResearchRuntime:
    max_history_messages = 6
    retrieval_history_messages = 4

    def __init__(self, retriever):
        self.retriever = retriever

    @staticmethod
    def normalize_history(history, current_question=None):
        return history or []

    @staticmethod
    def build_retrieval_question(question, history):
        return question, bool(history)


class LocalEvidenceRetriever:
    def __init__(self, contexts):
        self.contexts = contexts
        self.calls = []

    def retrieve(self, question, final_k):
        self.calls.append({"question": question, "final_k": final_k})
        return {"final_context": self.contexts[:final_k]}

    def analyze_question(self, question):
        return {"normalized_question": question, "facets": ["cause"]}

    def classify_question(self, question):
        return {"is_ood": False, "ood_reason": "", "intent": "history"}


class FakeResearchModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def generate_json(self, *, adapter, messages, **kwargs):
        assert adapter == "research"
        state = json.loads(messages[1]["content"])
        self.calls.append(state)
        index = min(len(self.calls) - 1, len(self.decisions) - 1)
        return self.decisions[index]


class FakeEvidenceModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_json(self, *, adapter, messages, **kwargs):
        assert adapter == "evidence"
        self.calls.append(json.loads(messages[1]["content"]))
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


class FakeWikipediaSearch:
    name = "search_wikipedia"
    description = "fake wiki search"
    input_schema = SearchWikipediaInput

    def __init__(self):
        self.calls = []

    def run(self, arguments):
        self.calls.append(arguments)
        return [{
            "chunk_id": "wiki_vi_1",
            "source_kind": "wikipedia",
            "title": "Nhà Trần",
            "text": "Nhà Trần suy yếu vào cuối thế kỷ XIV.",
            "metadata": {"page_id": 1, "language": "vi"},
        }]


class InstrumentedInspectEvidenceTool(InspectEvidenceTool):
    def __init__(self, store):
        super().__init__(store)
        self.calls = []

    def run(self, arguments):
        self.calls.append(arguments)
        return super().run(arguments)


class FakeWikipediaFetch:
    name = "fetch_wikipedia_page"
    description = "fake wiki fetch"
    input_schema = FetchWikipediaPageInput

    def __init__(self):
        self.calls = []

    def run(self, arguments):
        self.calls.append(arguments)
        return {
            "chunk_id": "wiki_vi_1",
            "source_kind": "wikipedia",
            "title": "Nhà Trần",
            "text": "Cuối thời Trần, quyền lực triều đình suy yếu và nhiều cuộc khởi nghĩa nông dân bùng nổ.",
            "metadata": {"page_id": 1, "language": "vi"},
        }


def _research_agent(contexts, model, *, wiki_search=None, wiki_fetch=None):
    store = SessionEvidenceStore()
    retriever = LocalEvidenceRetriever(contexts)
    registry = ToolRegistry()
    registry.register(SearchHistoryTool(retriever))
    registry.register(RetrieveEvidenceTool(store))
    registry.register(InstrumentedInspectEvidenceTool(store))
    if wiki_search is not None:
        registry.register(wiki_search)
    if wiki_fetch is not None:
        registry.register(wiki_fetch)
    return ResearchAgent(
        registry=registry,
        evidence_store=store,
        retrieval_runtime=FakeResearchRuntime(retriever),
        model_runtime=model,
        max_steps=2,
    )


def test_external_evidence_visibility_balances_budget_and_reports_drops():
    local = [
        {
            "chunk_id": f"local_{index}",
            "title": "Nguồn địa phương",
            "text": f"Nguồn local {index} nói về một chi tiết phụ.",
            "source_kind": "history",
            "reranker_score": 0.4,
        }
        for index in range(8)
    ]
    wiki = [
        {
            "chunk_id": f"wiki_{index}",
            "title": "Trận Bạch Đằng (938)" if index == 1 else "Nguồn wiki phụ",
            "text": (
                "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc và mở ra nền độc lập tự chủ."
                if index == 1
                else "Một đoạn wiki ít liên quan."
            ),
            "source_kind": "wikipedia",
            "reranker_score": 0.9 if index == 1 else 0.2,
        }
        for index in range(3)
    ]
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "wiki_1",
            "relevance": 1.0,
            "claims": [wiki[1]["text"]],
            "compressed_text": wiki[1]["text"],
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "wiki_1 đủ.",
    }])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        [*local, *wiki],
        final_k=8,
    )

    visible_ids = [item["evidence_id"] for item in runtime.calls[0]["evidence"]]
    assert "wiki_1" in visible_ids
    assert critique.raw_candidate_count == 11
    assert critique.model_visible_candidate_count == 8
    assert critique.dropped_for_budget_count == 3
    assert critique.source_kind_counts_raw["wikipedia"] == 3
    assert critique.source_kind_counts_visible["wikipedia"] >= 1


def _tran_candidates():
    return [
        {
            "chunk_id": TRAN_WEAK_ID,
            "title": "Lịch sử Phật giáo Việt Nam",
            "text": "Thời Trần, Phật giáo suy giảm dần và một số nhà sư từng được triều đình ưu đãi.",
            "source_kind": "history",
            "reranker_score": 0.2,
        },
        {
            "chunk_id": TRAN_POLITICAL_ID,
            "title": "Nhà Trần",
            "text": "Cuối thời Trần, quyền lực triều đình suy yếu, vua quan sa sút và Hồ Quý Ly thao túng chính sự.",
            "source_kind": "history",
            "reranker_score": 0.96,
        },
        {
            "chunk_id": TRAN_SOCIAL_ID,
            "title": "Nhà Trần",
            "text": "Nhiều cuộc khởi nghĩa nông dân và biến động xã hội làm suy yếu quyền lực nhà Trần.",
            "source_kind": "history",
            "reranker_score": 0.94,
        },
        {
            "chunk_id": TRAN_MILITARY_ID,
            "title": "Chiến tranh Việt-Chiêm",
            "text": "Các cuộc chiến tranh với Chiêm Thành gây bất ổn quân sự và góp phần làm suy yếu nhà Trần.",
            "source_kind": "history",
            "reranker_score": 0.9,
        },
    ]


def _weak_evidence_output():
    claim = "Thời Trần, Phật giáo suy giảm dần và một số nhà sư từng được triều đình ưu đãi."
    return {
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": TRAN_WEAK_ID,
            "relevance": 0.7,
            "claims": [claim],
            "compressed_text": claim,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": f"{TRAN_WEAK_ID} đủ; {TRAN_SOCIAL_ID} cũng liên quan.",
    }


def test_local_prefetch_prevents_false_no_supporting_local_evidence_message():
    contexts = [{
        "chunk_id": "ev_local",
        "title": "Nhà Trần",
        "text": "Nhiều cuộc khởi nghĩa nông dân làm suy yếu nhà Trần.",
    }]
    model = FakeResearchModel([{
        "action": "finish",
        "sufficient": False,
        "missing_information": ["No supporting local evidence was retrieved."],
    }])
    agent = _research_agent(contexts, model)

    result = asyncio.run(agent.run("Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?", final_k=6))

    assert result.evidence[0]["chunk_id"] == "ev_local"
    finish = next(step for step in result.debug["tools"] if step.get("action") == "finish")
    assert "No supporting local evidence was retrieved." not in finish["missing_information"]
    assert result.debug["generation_calls"] == 1
    assert model.calls[0]["observations"][0]["tool"] == "search_history"


def test_research_insufficient_triggers_one_wikipedia_fallback_round():
    wiki_search = FakeWikipediaSearch()
    wiki_fetch = FakeWikipediaFetch()
    model = FakeResearchModel([{
        "action": "finish",
        "sufficient": False,
        "missing_information": ["Cần nguồn ngoài."],
    }])
    agent = _research_agent([], model, wiki_search=wiki_search, wiki_fetch=wiki_fetch)

    result = asyncio.run(agent.run("Một câu hỏi lịch sử cần kiểm tra ngoài?", final_k=6))

    assert len(wiki_search.calls) == 1
    assert len(wiki_fetch.calls) == 1
    assert any(step.get("external_fallback") for step in result.debug["tools"])
    assert "wiki_vi_1" in result.debug["evidence_ids"]


def test_research_sufficient_does_not_call_wikipedia():
    wiki_search = FakeWikipediaSearch()
    wiki_fetch = FakeWikipediaFetch()
    model = FakeResearchModel([{
        "action": "finish",
        "sufficient": True,
        "missing_information": [],
    }])
    agent = _research_agent([{
        "chunk_id": "ev_local",
        "title": "Nhà Trần",
        "text": "Nhiều cuộc khởi nghĩa nông dân làm suy yếu nhà Trần.",
    }], model, wiki_search=wiki_search, wiki_fetch=wiki_fetch)

    result = asyncio.run(agent.run("Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?", final_k=6))

    assert wiki_search.calls == []
    assert wiki_fetch.calls == []
    assert result.debug["generation_calls"] == 1


def test_verification_query_prefetches_external_research_even_with_local_hits():
    wiki_search = FakeWikipediaSearch()
    wiki_fetch = FakeWikipediaFetch()
    model = FakeResearchModel([{
        "action": "finish",
        "sufficient": True,
        "missing_information": [],
    }])
    agent = _research_agent([{
        "chunk_id": "ev_local",
        "title": "Một nguồn nội bộ",
        "text": "Nguồn nội bộ nêu một nhận định cần kiểm chứng.",
    }], model, wiki_search=wiki_search, wiki_fetch=wiki_fetch)

    result = asyncio.run(agent.run("Có thật sự tin đồn này đúng không?", final_k=6))

    assert len(wiki_search.calls) == 1
    assert len(wiki_fetch.calls) == 1
    assert result.debug["external_research_needed"] is True
    assert result.debug["external_research_reason"] == "verification_or_rumor"


def test_external_research_policy_classifies_query_and_keeps_simple_factual_local():
    assert needs_external_research("Có thật sự tin đồn này đúng không?", {"local_result_count": 4})
    assert _external_research_reason("Nhận định gây tranh cãi này có mâu thuẫn không?", {"local_result_count": 4}) == "disputed_claim"
    assert _external_research_reason("Ai là người giỏi nhất?", {"local_result_count": 4}) == "evaluative_superlative"
    assert not needs_external_research("Nhà Trần thành lập năm nào?", {"local_result_count": 4})


def test_wikipedia_year_filtering_prefers_matching_938_page_over_1288():
    rows = [
        {"chunk_id": "wiki_1288", "title": "Trận Bạch Đằng (1288)", "text": "Trận Bạch Đằng năm 1288."},
        {"chunk_id": "wiki_vn", "title": "Việt Nam", "text": "Năm 938 có một sự kiện lịch sử."},
        {"chunk_id": "wiki_938", "title": "Trận Bạch Đằng (938)", "text": "Chiến thắng Bạch Đằng năm 938."},
    ]

    selected, rejected = _select_wikipedia_candidate(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        rows,
    )

    assert selected["title"] == "Trận Bạch Đằng (938)"
    assert rejected == 1


def test_duplicate_inspect_after_prefetch_is_skipped_without_tool_execution():
    contexts = [{
        "chunk_id": "ev_local",
        "title": "Bạch Đằng",
        "text": "Chiến thắng Bạch Đằng năm 938 mở ra nền độc lập tự chủ.",
    }]
    model = FakeResearchModel([{
        "action": "tool",
        "tool_name": "inspect_evidence",
        "arguments": {"ids": ["ev_local"]},
    }])
    agent = _research_agent(contexts, model)

    result = asyncio.run(agent.run("Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?", final_k=6))

    assert result.debug["generation_calls"] == 1
    assert any(step.get("action") == "duplicate_inspect_skipped" for step in result.debug["tools"])
    inspect_tool = agent.registry.get("inspect_evidence")
    assert inspect_tool.calls == []


def test_evidence_relevance_guard_rejects_weak_buddhism_only_selection():
    runtime = FakeEvidenceModel([_weak_evidence_output(), _weak_evidence_output()])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?",
        _tran_candidates(),
        final_k=4,
    )

    assert len(runtime.calls) == 1
    assert critique.repair_path == "deterministic_semantic_guard"
    assert critique.sufficient is True
    assert critique.selected_ids != [TRAN_WEAK_ID]
    assert TRAN_POLITICAL_ID in critique.selected_ids or TRAN_SOCIAL_ID in critique.selected_ids
    assert any(context["chunk_id"] in {TRAN_POLITICAL_ID, TRAN_SOCIAL_ID} for context in contexts)


def test_evidence_summary_only_mentions_retained_selected_ids():
    claim = "Nhiều cuộc khởi nghĩa nông dân và biến động xã hội làm suy yếu quyền lực nhà Trần."
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": TRAN_SOCIAL_ID,
            "relevance": 1.0,
            "claims": [claim],
            "compressed_text": claim,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": f"{TRAN_SOCIAL_ID} và {TRAN_POLITICAL_ID} đều chứng minh nguyên nhân.",
    }])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?",
        _tran_candidates(),
        final_k=4,
    )

    assert TRAN_SOCIAL_ID in critique.summary
    if TRAN_POLITICAL_ID in critique.summary:
        assert TRAN_POLITICAL_ID in critique.selected_ids
    assert all(chunk_id in critique.selected_ids for chunk_id in (TRAN_SOCIAL_ID,) if chunk_id in critique.summary)


def test_factual_direct_evidence_accepts_valid_first_pass_without_reconsideration():
    question = "Ai được mệnh danh là anh cả Quân đội Nhân dân Việt Nam?"
    claim = "Đại tướng Võ Nguyên Giáp được mệnh danh là anh cả của Quân đội Nhân dân Việt Nam."
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "vo_nguyen_giap",
            "relevance": 1.0,
            "claims": [claim],
            "compressed_text": claim,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Bằng chứng trực tiếp trả lời câu hỏi.",
    }])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        question,
        [
            {
                "chunk_id": "vo_nguyen_giap",
                "title": "Võ Nguyên Giáp",
                "text": claim,
                "source_kind": "history",
                "final_retrieval_score": 0.98,
            },
            {
                "chunk_id": "dien_bien",
                "title": "Chiến thắng Điện Biên Phủ",
                "text": "Chiến thắng Điện Biên Phủ năm 1954 là thắng lợi quân sự lớn.",
                "source_kind": "history",
                "final_retrieval_score": 0.8,
            },
        ],
        final_k=2,
    )

    assert len(runtime.calls) == 1
    assert critique.question_type == "factual"
    assert critique.repair_path is None
    assert critique.semantic_guard_findings["guard_policy"] == "accept_valid_factual_first_pass"
    assert critique.semantic_guard_findings["coverage_triggered"] is False
    assert critique.semantic_guard_findings["relevance_triggered"] is False
    assert critique.first_validation_issues == []
    assert critique.final_validation_issues == []
    assert contexts[0]["claims"] == [claim]


def test_cause_coverage_guard_prevents_unjustified_single_factor_collapse():
    claim = "Cuối thời Trần, quyền lực triều đình suy yếu, vua quan sa sút và Hồ Quý Ly thao túng chính sự."
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": TRAN_POLITICAL_ID,
            "relevance": 1.0,
            "claims": [claim],
            "compressed_text": claim,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": f"{TRAN_POLITICAL_ID} đủ.",
    }, {
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": TRAN_POLITICAL_ID,
            "relevance": 1.0,
            "claims": [claim],
            "compressed_text": claim,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": f"{TRAN_POLITICAL_ID} đủ.",
    }])

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Phân tích nguyên nhân dẫn đến sự suy yếu của nhà Trần.",
        _tran_candidates(),
        final_k=4,
    )

    assert critique.repair_path == "deterministic_semantic_guard"
    assert len(critique.selected_ids) >= 2
    selected_text = " ".join(item.compressed_text for item in critique.selected_evidence)
    assert "khởi nghĩa" in selected_text or "chiến tranh" in selected_text


def test_claim_level_relevance_drops_ritual_claim_for_bach_dang_significance():
    direct = "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc và mở ra nền độc lập tự chủ."
    ritual = "Đời sau lập miếu điện, tế lễ thái lao và dựng cờ hoàng đạo để tưởng niệm."
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": "bach_dang_direct",
                "relevance": 1.0,
                "claims": [direct],
                "compressed_text": direct,
            },
            {
                "evidence_id": "bach_dang_ritual",
                "relevance": 0.8,
                "claims": [ritual],
                "compressed_text": ritual,
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Hai nguồn được chọn.",
    }, {
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": "bach_dang_direct",
                "relevance": 1.0,
                "claims": [direct],
                "compressed_text": direct,
            },
            {
                "evidence_id": "bach_dang_ritual",
                "relevance": 0.8,
                "claims": [ritual],
                "compressed_text": ritual,
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Hai nguồn được chọn.",
    }])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        [
            {"chunk_id": "bach_dang_direct", "title": "Trận Bạch Đằng (938)", "text": direct, "source_kind": "history"},
            {"chunk_id": "bach_dang_ritual", "title": "Trận Bạch Đằng (938)", "text": ritual, "source_kind": "history"},
        ],
        final_k=2,
    )

    assert critique.repair_path == "deterministic_semantic_guard"
    assert critique.selected_ids == ["bach_dang_direct"]
    assert [context["chunk_id"] for context in contexts] == ["bach_dang_direct"]


def test_evidence_guard_drops_title_year_conflict_for_bach_dang_938():
    direct = "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc và mở ra nền độc lập tự chủ."
    wrong_year = "Trận Bạch Đằng năm 1288 là chiến thắng trước quân Nguyên."
    runtime = FakeEvidenceModel([{
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": "bach_dang_938",
                "relevance": 1.0,
                "claims": [direct],
                "compressed_text": direct,
            },
            {
                "evidence_id": "bach_dang_1288",
                "relevance": 0.9,
                "claims": [wrong_year],
                "compressed_text": wrong_year,
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Hai nguồn được chọn.",
    }, {
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": "bach_dang_938",
                "relevance": 1.0,
                "claims": [direct],
                "compressed_text": direct,
            },
            {
                "evidence_id": "bach_dang_1288",
                "relevance": 0.9,
                "claims": [wrong_year],
                "compressed_text": wrong_year,
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Hai nguồn được chọn.",
    }])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        [
            {"chunk_id": "bach_dang_938", "title": "Trận Bạch Đằng (938)", "text": direct, "source_kind": "history"},
            {"chunk_id": "bach_dang_1288", "title": "Trận Bạch Đằng (1288)", "text": wrong_year, "source_kind": "history"},
        ],
        final_k=2,
    )

    assert critique.repair_path == "deterministic_semantic_guard"
    assert critique.selected_ids == ["bach_dang_938"]
    assert [context["title"] for context in contexts] == ["Trận Bạch Đằng (938)"]


class DepthRuntime:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or ["Nguồn được dùng: [ev_01]\n\nTrả lời:\nCâu trả lời."])
        self.calls = []

    def generate_text(self, *, adapter, messages, **kwargs):
        self.calls.append({"adapter": adapter, "messages": messages})
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


class DepthRetriever:
    final_context_k = 6

    def retrieve(self, question, final_k):
        return {"question": question, "final_context": [{"chunk_id": "ev_01", "title": "Bạch Đằng", "text": "Chiến thắng Bạch Đằng năm 938 mở ra nền độc lập tự chủ."}]}

    def analyze_question(self, question):
        return {"facets": ["significance"]}

    def context_title_diversity(self, contexts):
        return 1.0


class DepthRetrievalRuntime(FakeResearchRuntime):
    pass


def test_deep_history_retries_short_answer_from_validated_evidence():
    final_answer = (
        "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc, mở ra nền độc lập tự chủ "
        "và tạo cơ sở để Ngô Quyền xưng vương. Vì vậy, ý nghĩa của thắng lợi nằm ở cả hệ quả "
        "trước mắt của trận đánh lẫn bước chuyển lâu dài sang quyền tự chủ."
    )
    runtime = DepthRuntime([
        "Nguồn được dùng: [ev_01]\n\nTrả lời:\nCâu trả lời.",
        f"Nguồn được dùng: [ev_01] [ev_02]\n\nTrả lời:\n{final_answer}",
    ])
    answerer = HistoryAnswererAgent(model_runtime=runtime)
    contexts = [
        {
            "chunk_id": "ev_01",
            "title": "Trận Bạch Đằng (938)",
            "text": "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc.",
            "claims": ["Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc."],
        },
        {
            "chunk_id": "ev_02",
            "title": "Nhà Ngô",
            "text": (
                "Sau chiến thắng, Ngô Quyền xưng vương và mở ra nền độc lập tự chủ. "
                "Sự kiện này đánh dấu bước chuyển lâu dài sang quyền tự chủ."
            ),
            "claims": [
                "Sau chiến thắng, Ngô Quyền xưng vương và mở ra nền độc lập tự chủ.",
                "Sự kiện này đánh dấu bước chuyển lâu dài sang quyền tự chủ.",
            ],
        },
    ]

    result = answerer.answer(
        question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        contexts=contexts,
        analysis={},
        tool_trace=[],
        answer_depth="deep",
        inference_mode="agentic_rag",
    )

    assert len(runtime.calls) == 2
    assert result["structured_expansion_used"] is False
    assert result["source_ids"] == ["ev_01", "ev_02"]
    assert result["answer"] == final_answer
    assert result["answer_provenance"]["history_retry_used"] is True


def test_standard_history_keeps_adapter_answer_without_deep_expansion():
    runtime = DepthRuntime()
    answerer = HistoryAnswererAgent(model_runtime=runtime)

    result = answerer.answer(
        question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
        contexts=[
            {
                "chunk_id": "ev_01",
                "title": "Trận Bạch Đằng (938)",
                "text": "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc.",
            },
            {
                "chunk_id": "ev_02",
                "title": "Nhà Ngô",
                "text": "Sau chiến thắng, Ngô Quyền xưng vương và mở ra nền độc lập tự chủ.",
            },
        ],
        analysis={},
        tool_trace=[],
        answer_depth="standard",
    )

    assert result["structured_expansion_used"] is False
    assert result["answer"] == "Câu trả lời."
    assert result["source_ids"] == ["ev_01"]


class DepthResearchAgent:
    retrieval_runtime = DepthRetrievalRuntime(DepthRetriever())
    evidence_store = SessionEvidenceStore()
    model_runtime = None

    async def run(self, *args, **kwargs):
        return SimpleNamespace(
            evidence=[{"chunk_id": "ev_01", "title": "Bạch Đằng", "text": "Chiến thắng Bạch Đằng năm 938 mở ra nền độc lập tự chủ."}],
            debug={"steps": 0, "generation_calls": 0, "tools": [], "evidence_ids": ["ev_01"], "retrieval_question": args[0] if args else ""},
            tool_trace=[],
            analysis={"facets": ["significance"]},
            is_ood=False,
            ood_reason="",
        )


class DepthEvidenceAgent:
    def compress(self, question, evidence, *, final_k, request_id=None):
        from app.agents.schemas import EvidenceCritique, SelectedEvidence

        selected = SelectedEvidence(
            evidence_id="ev_01",
            relevance=1.0,
            claims=[evidence[0]["text"]],
            compressed_text=evidence[0]["text"],
        )
        return EvidenceCritique(
            status="sufficient",
            selected_evidence=[selected],
            selected_ids=["ev_01"],
            sufficient=True,
            model_input_evidence=[{"evidence_id": "ev_01"}],
        ), evidence


def test_history_answer_depth_is_standard_for_hybrid_and_deep_for_agentic():
    hybrid_runtime = DepthRuntime()
    hybrid = HybridRAGOrchestrator(
        retriever=DepthRetriever(),
        retrieval_runtime=DepthRetrievalRuntime(DepthRetriever()),
        answerer=HistoryAnswererAgent(model_runtime=hybrid_runtime),
    )
    hybrid_result = hybrid.chat("Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?", final_k=1)

    agentic_runtime = DepthRuntime()
    agentic = AgentOrchestrator(
        research_agent=DepthResearchAgent(),
        evidence_agent=DepthEvidenceAgent(),
        answerer=HistoryAnswererAgent(model_runtime=agentic_runtime),
    )
    agentic_result = asyncio.run(agentic.run(question="Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?", final_k=1))

    assert hybrid_result["history_debug"]["answer_depth"] == "standard"
    assert "Yêu cầu trả lời:" not in hybrid_runtime.calls[0]["messages"][0]["content"]
    assert agentic_result["history_debug"]["answer_depth"] == "deep"
    assert "Yêu cầu trả lời:" in agentic_runtime.calls[0]["messages"][0]["content"]
