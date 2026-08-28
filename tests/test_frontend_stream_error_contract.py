from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_stream_error_stops_reader_and_exports_evidence_message():
    api_source = (ROOT / "frontend" / "src" / "services" / "api.js").read_text(encoding="utf-8")

    assert "EVIDENCE_CONTRACT_FAILURE_MESSAGE" in api_source
    assert "if (parsed.event === \"error\")" in api_source
    assert "sawStreamError = true" in api_source
    assert "await reader.cancel()" in api_source


def test_app_uses_specific_placeholder_for_evidence_contract_error():
    app_source = (ROOT / "frontend" / "src" / "App.jsx").read_text(encoding="utf-8")

    assert "data?.type === \"evidence_contract_error\"" in app_source
    assert "EVIDENCE_CONTRACT_FAILURE_MESSAGE" in app_source
    assert "setStatus(\"error\")" in app_source
