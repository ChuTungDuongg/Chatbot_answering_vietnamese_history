from __future__ import annotations

import pytest

for dependency in ("pymupdf", "torch", "bm25s", "faiss", "sentence_transformers", "transformers"):
    pytest.importorskip(dependency)

from app.main import app


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
