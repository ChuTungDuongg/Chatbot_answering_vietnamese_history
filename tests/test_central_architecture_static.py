from __future__ import annotations

import inspect
from pathlib import Path

from app.agents.central_agent import CentralAgent


ROOT = Path(__file__).resolve().parents[1]


def test_central_constructor_has_no_three_llm_or_fast_runtime_dependency():
    parameters = inspect.signature(CentralAgent).parameters

    assert "model_runtime" in parameters
    assert "tool_registry" in parameters
    assert "orchestrator" not in parameters
    assert "fast_service" not in parameters


def test_central_module_does_not_import_role_agents_or_orchestrator():
    source = (ROOT / "app" / "agents" / "central_agent.py").read_text(encoding="utf-8")

    for forbidden in (
        "ResearchAgent", "EvidenceCriticAgent", "HistoryAnswererAgent", "AgentOrchestrator",
    ):
        assert forbidden not in source


def test_production_modal_uses_a100_and_lazy_independent_central_runtime():
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")

    assert 'gpu="A100"' in source
    assert 'gpu="L4"' not in source
    assert '"CENTRAL_AGENT_MODEL_ID": "Qwen/Qwen3-8B"' in source
    assert '"CENTRAL_AGENT_ADAPTER_PATH": "/artifacts/adapters/central"' in source
    assert '"RUNTIME_LOADING_STRATEGY": "lazy"' in source
    assert 'os.getenv("WEB_SEARCH_PROVIDER", "local-only")' in source
    assert "MODAL_WEB_SEARCH_SECRET_NAME" in source
    assert "WEB_SEARCH_API_KEY" not in source


def test_obsolete_root_modal_smoke_files_are_absent():
    for name in (
        "modal_agentic_smoke.py", "modal_l4_diagnostics.py", "full_modal_runtime_sanity.py",
        "modal_runtime_sanity.py", "modal_artifact_sanity.py", "modal_fix.py",
    ):
        assert not (ROOT / name).exists()
    assert (ROOT / "scripts" / "central_smoke.py").is_file()
    assert (ROOT / "scripts" / "modal_artifact_sanity.py").is_file()
