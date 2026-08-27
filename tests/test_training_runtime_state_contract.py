import json

from app.agents.policy_schema import ResearchPolicyState, default_research_tool_definitions, serialize_policy_state


def test_training_and_runtime_share_state_and_full_tool_definitions():
    state = ResearchPolicyState(question="Bạch Đằng?", tools=default_research_tool_definitions())
    restored = ResearchPolicyState.model_validate(json.loads(serialize_policy_state(state)))
    assert restored == state
    assert all(tool.description and tool.input_schema for tool in restored.tools)
    assert {tool.name for tool in restored.tools} >= {"search_history", "search_web", "inspect_evidence"}
