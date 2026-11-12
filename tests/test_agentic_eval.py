"""The offline fixtures also run in shallow CI checkouts without loading Git code."""

import asyncio
import json
from pathlib import Path

import pytest

from evaluation.agentic.run import evaluate_case, load_cases, source_digest


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_agentic_case_contracts(case):
    result = asyncio.run(evaluate_case(case))
    assert result["dependency_order_valid"]
    assert result["native_output_correlation_valid"]
    assert result["duplicate_calls"] == result["excess_calls"] == 0
    if case["id"] == "repeated_search":
        assert not result["task_complete"]  # A stopped loop is not a substantiated answer.
    else:
        assert result["task_complete"]


def test_committed_evaluation_report_matches_runtime_and_fixture_sources():
    path = Path(__file__).resolve().parents[1] / "evaluation/agentic/results.json"
    if not path.exists():
        pytest.fail("Run python -m evaluation.agentic.run to regenerate the report")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["source_sha256"] == source_digest()
    assert report["usd_cost"] is None
    assert "simulated" in report["measurement"].lower()
