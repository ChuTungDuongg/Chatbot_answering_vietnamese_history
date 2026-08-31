from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pydantic import BaseModel

from app.agents.central_agent import CentralAgent, INSUFFICIENT_EVIDENCE_ANSWER
from app.agents.config import AgentConfig
from app.agents.model_registry import ROLE_MODELS
from app.chat_modes import ChatMode
from app.services.chat_mode_router import ChatModeRouter
from app.services.fast_service import FastChatService
from app.tools.registry import ToolRegistry


class FakeRuntime:
    max_history_messages = 6
    retrieval_history_messages = 4

    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": self.name, "status": "ok"}


class FakeAsyncOrchestrator:
    max_history_messages = 6
    retrieval_history_messages = 4

    def __init__(self, result=None, *, delay: float = 0):
        self.result = result or {"answer": "grounded", "status": "ok"}
        self.delay = delay
        self.calls = []
        self.research_agent = SimpleNamespace(max_steps=4)

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        return dict(self.result)


def test_mode_router_dispatches_fast_hybrid_and_agent_paths():
    fast = FakeRuntime("fast")
    hybrid = FakeRuntime("hybrid")
    agent = FakeRuntime("agent")
    router = ChatModeRouter(fast=fast, hybrid=hybrid, agent=agent)

    assert router.chat(ChatMode.FAST, question="q")["answer"] == "fast"
    assert router.chat(ChatMode.HYBRID, question="q")["answer"] == "hybrid"
    assert router.chat(ChatMode.AGENT, question="q")["answer"] == "agent"
    assert [len(runtime.calls) for runtime in (fast, hybrid, agent)] == [1, 1, 1]


def test_fast_service_uses_existing_direct_path_with_bounded_contexts():
    hybrid = FakeRuntime("direct-hybrid")
    fast = FastChatService(hybrid, max_contexts=3)

    result = fast.chat(question="q", final_k=9)

    assert result["fast_path"] is True
    assert hybrid.calls[0]["final_k"] == 3


def test_central_agent_is_bounded_and_preserves_three_registered_roles():
    orchestrator = FakeAsyncOrchestrator()
    fast = FakeRuntime("fast")
    config = AgentConfig(max_steps=4, timeout_seconds=1)
    agent = CentralAgent(orchestrator, fast_service=fast, config=config)

    result = agent.chat("So sánh hai sự kiện", final_k=6)

    assert result["answer"] == "grounded"
    assert result["central_agent"]["max_steps"] == 4
    assert result["central_agent"]["selected_path"] == "agent_tools"
    assert len(orchestrator.calls) == 1
    assert fast.calls == []
    assert set(ROLE_MODELS) == {"research", "evidence", "history"}


def test_central_agent_uses_fast_path_for_simple_request_without_three_role_pipeline():
    orchestrator = FakeAsyncOrchestrator()
    fast = FakeRuntime("direct")
    agent = CentralAgent(
        orchestrator,
        fast_service=fast,
        config=AgentConfig(max_steps=4, timeout_seconds=1),
    )

    result = agent.chat("Chiến thắng Bạch Đằng diễn ra năm nào?")

    assert result["answer"] == "direct"
    assert result["central_agent"]["selected_path"] == "fast_direct"
    assert len(fast.calls) == 1
    assert orchestrator.calls == []


def test_central_agent_timeout_and_no_evidence_fail_gracefully():
    fast = FakeRuntime("fast")
    timeout_agent = CentralAgent(
        FakeAsyncOrchestrator(delay=0.02),
        fast_service=fast,
        config=AgentConfig(max_steps=4, timeout_seconds=0.001),
    )
    empty_agent = CentralAgent(
        FakeAsyncOrchestrator(result={"answer": "", "status": "ok"}),
        fast_service=fast,
        config=AgentConfig(max_steps=4, timeout_seconds=1),
    )

    for result in (
        timeout_agent.chat("Hãy kiểm chứng nguồn về sự kiện này"),
        empty_agent.chat("Hãy kiểm chứng nguồn về sự kiện này"),
    ):
        assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
        assert result["status"] == "insufficient_evidence"
        assert result["source_ids"] == []


def test_tool_error_is_normalized_without_retry_loop():
    class EmptyInput(BaseModel):
        pass

    class BrokenTool:
        name = "broken"
        description = "Always fails"
        input_schema = EmptyInput

        def run(self, _arguments):
            raise RuntimeError("fixture failure")

    result, record = asyncio.run(ToolRegistryWithBroken(BrokenTool()).call("broken", {}))
    assert result is None
    assert record.error == "fixture failure"


class ToolRegistryWithBroken(ToolRegistry):
    def __init__(self, tool):
        super().__init__()
        self.register(tool)
