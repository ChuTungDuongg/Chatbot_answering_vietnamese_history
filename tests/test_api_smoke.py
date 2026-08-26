from __future__ import annotations

import asyncio

import pytest

for dependency in ("pymupdf", "torch", "bm25s", "faiss", "sentence_transformers", "transformers"):
    pytest.importorskip(dependency)

from app.main import app, health


def test_app_exposes_agentic_feature_flag():
    route_paths = {route.path for route in app.routes if hasattr(route, "path")}
    for route in app.routes:
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            route_paths.update(
                child.path for child in original_router.routes if hasattr(child, "path")
            )
    assert "/api/v1/chat" in route_paths
    assert "/ready" in route_paths


def test_api_only_lifespan_health_and_readiness():
    async def run_smoke():
        async with app.router.lifespan_context(app):
            assert app.state.rag_service.loaded is True
            assert app.state.orchestrator is None
            response = await health()
            assert response["status"] == "ok"

    asyncio.run(run_smoke())
