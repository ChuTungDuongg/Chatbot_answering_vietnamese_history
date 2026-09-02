import asyncio

from app.agents.research.agent import ResearchAgent
from app.tools.evidence_tools import SessionEvidenceStore
from app.tools.local_search import SearchHistoryTool
from app.tools.registry import ToolRegistry


class CountingRetriever:
    def __init__(self):
        self.retrieve_calls = 0

    def retrieve(self, question, final_k):
        self.retrieve_calls += 1
        return {"final_context": []}

    def analyze_question(self, question):
        return {"question": question, "facets": ["general"]}


class Generator:
    def __init__(self, retriever):
        self.retriever = retriever

    def normalize_history(self, history, current_question):
        return []

    def build_retrieval_question(self, question, history):
        return question, False


class FinishRuntime:
    def generate_json(self, **kwargs):
        return {"action": "finish", "sufficient": True, "missing_information": []}


def test_model_finish_still_uses_one_deterministic_local_prefetch():
    retriever = CountingRetriever()
    registry = ToolRegistry()
    registry.register(SearchHistoryTool(retriever))
    agent = ResearchAgent(
        registry=registry,
        evidence_store=SessionEvidenceStore(),
        retrieval_runtime=Generator(retriever),
        model_runtime=FinishRuntime(),
    )
    result = asyncio.run(agent.run("Xin chào", final_k=4))
    assert retriever.retrieve_calls == 1
    assert result.tool_trace == ["search_history:0", "agent:finish:1"]
    assert result.debug["tools"][0]["deterministic_prefetch"] is True
    assert result.debug["generation_calls"] == 1
