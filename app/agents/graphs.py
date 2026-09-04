"""Canonical mode-to-graph factory; graph implementations remain isolated."""
from __future__ import annotations

from typing import Any

from app.agents.central.graph import CentralGraphDependencies, build_central_graph
from app.agents.hybrid_graph import HybridGraphDependencies, build_hybrid_graph
from app.agents.three_llm.graph import ThreeLLMGraphDependencies, build_three_llm_graph
from app.chat_modes import ChatMode, normalize_chat_mode


def build_graph(mode: ChatMode | str, dependencies: Any):
    canonical = normalize_chat_mode(mode)
    if canonical is ChatMode.HYBRID:
        if not isinstance(dependencies, HybridGraphDependencies):
            raise TypeError("Hybrid mode requires HybridGraphDependencies")
        return build_hybrid_graph(dependencies)
    if canonical is ChatMode.THREE_LLM:
        if not isinstance(dependencies, ThreeLLMGraphDependencies):
            raise TypeError("three_llm mode requires ThreeLLMGraphDependencies")
        return build_three_llm_graph(dependencies)
    if not isinstance(dependencies, CentralGraphDependencies):
        raise TypeError("Central mode requires CentralGraphDependencies")
    return build_central_graph(dependencies)


__all__ = ["build_graph"]
