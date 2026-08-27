import pytest
from pydantic import ValidationError

from app.agents.policy_schema import ResearchPolicyState, validate_runtime_decision


def test_policy_schema_rejects_unobserved_evidence_and_bad_decisions():
    with pytest.raises(ValidationError, match="prior observations"):
        ResearchPolicyState(question="q", tools=[], observations=[], evidence_ids=["future"])
    with pytest.raises(Exception):
        validate_runtime_decision({"action": "tool", "arguments": {}}, tool_names={"search_history"})
    with pytest.raises(Exception):
        validate_runtime_decision({"action": "finish", "sufficient": True})
    with pytest.raises(ValueError, match="unknown tool"):
        validate_runtime_decision({"action": "tool", "tool_name": "bad", "arguments": {}}, tool_names={"ok"})
