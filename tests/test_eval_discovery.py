"""Studio discovery follows its feature flag and existing owner authentication."""

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "enabled,authenticated,expected", [(True, True, True), (False, True, False), (True, False, False)]
)
def test_readiness_only_advertises_studio_when_it_can_be_accessed(
    monkeypatch, enabled, authenticated, expected
):
    import app.main as main

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            eval_studio_enabled=enabled,
            auth_enabled=authenticated,
            is_full=False,
            mcp_max_tools_per_request=0,
            mcp_schema_budget_bytes=4096,
        ),
    )
    monkeypatch.setattr(
        main,
        "app",
        SimpleNamespace(
            state=SimpleNamespace(
                rag_service=SimpleNamespace(readiness=lambda: {"ready": True}),
            )
        ),
    )
    assert asyncio.run(main.ready())["eval_studio"] == {"enabled": expected}
