from tests.test_no_evidence_leakage import phase6_row
from training.research_agent.build_history_trajectories import build_trajectory_samples


def test_grounded_trajectory_is_unrolled_without_future_observations():
    rows = build_trajectory_samples(phase6_row())
    assert [row["training_target"]["action"] for row in rows] == ["tool", "tool", "finish"]
    assert [row["training_target"].get("tool_name") for row in rows] == ["search_history", "inspect_evidence", None]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[0]["training_prompt"]["observations"] == []
    assert rows[1]["training_prompt"]["observations"][0]["evidence_ids"] == ["gold-1"]
    assert rows[2]["training_prompt"]["observations"][1]["tool"] == "inspect_evidence"
    assert all(row["grounded"] and not row["synthetic"] for row in rows)
