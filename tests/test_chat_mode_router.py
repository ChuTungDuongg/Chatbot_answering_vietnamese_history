from __future__ import annotations

import pytest

from app.chat_modes import ChatMode
from app.services.chat_mode_router import ChatModeRouter
from app.services.fast_service import FastChatService


class FakeRuntime:
    max_history_messages = 6
    retrieval_history_messages = 4

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": self.name, "status": "ok"}


def test_mode_router_dispatches_three_canonical_isolated_paths():
    hybrid = FakeRuntime("hybrid")
    three_llm = FakeRuntime("three_llm")
    central = FakeRuntime("central")
    router = ChatModeRouter(hybrid=hybrid, three_llm=three_llm, central=central)

    assert router.chat(ChatMode.HYBRID, question="q")["answer"] == "hybrid"
    assert router.chat(ChatMode.THREE_LLM, question="q")["answer"] == "three_llm"
    assert router.chat(ChatMode.CENTRAL, question="q")["answer"] == "central"
    assert [len(runtime.calls) for runtime in (hybrid, three_llm, central)] == [1, 1, 1]


def test_unavailable_central_has_no_cross_mode_fallback():
    router = ChatModeRouter(
        hybrid=FakeRuntime("hybrid"),
        three_llm=FakeRuntime("three_llm"),
        central=None,
    )

    with pytest.raises(RuntimeError, match="central.*not enabled"):
        router.runtime_for(ChatMode.CENTRAL)


def test_fast_service_remains_a_bounded_facade_for_canonical_hybrid():
    direct = FakeRuntime("direct-hybrid")
    hybrid = FastChatService(direct, max_contexts=3)

    result = hybrid.chat(question="q", final_k=9)

    assert result["fast_path"] is True
    assert direct.calls[0]["final_k"] == 3
