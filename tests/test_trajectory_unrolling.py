from tests.test_no_evidence_leakage import phase6_row
from training.research_agent.build_history_trajectories import build_history_dataset, build_trajectory_samples


def test_grounded_trajectory_is_unrolled_without_future_observations():
    rows = build_trajectory_samples(phase6_row())
    assert [row["training_target"]["action"] for row in rows] == ["tool", "tool", "finish"]
    assert [row["training_target"].get("tool_name") for row in rows] == ["search_history", "inspect_evidence", None]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[0]["training_prompt"]["observations"] == []
    assert rows[1]["training_prompt"]["observations"][0]["evidence_ids"] == ["gold-1"]
    assert rows[2]["training_prompt"]["observations"][1]["tool"] == "inspect_evidence"
    assert all(row["grounded"] and not row["synthetic"] for row in rows)


def _variant(sample_type: str, suffix: str):
    return {
        "id": "sample_0001",
        "type": sample_type,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Câu hỏi:\nCâu hỏi variant {suffix}?\n\n"
                    f"Tài liệu tham khảo:\n[chunk-{suffix}] Evidence {suffix}"
                ),
            },
            {
                "role": "assistant",
                "content": f"Nguồn được dùng: [chunk-{suffix}]\n\nTrả lời: Target {suffix}",
            },
        ],
    }


def test_repeated_legacy_id_gets_shared_group_and_unique_deterministic_trajectories():
    source_rows = [
        _variant("noisy_context", "a"),
        _variant("grounded_qa", "b"),
        _variant("insufficient_context", "c"),
    ]
    built, stats = build_history_dataset(source_rows, include_no_tool=False)
    ids = [row["id"] for row in built]
    trajectory_ids = [build_trajectory_samples(row)[0]["trajectory_id"] for row in source_rows]
    group_ids = {row["group_id"] for row in built}
    assert len(ids) == len(set(ids))
    assert len(trajectory_ids) == len(set(trajectory_ids))
    assert group_ids == {"history-sample_0001"}
    assert stats["exact_duplicate_rows"] == 0
    rebuilt, _ = build_history_dataset(source_rows, include_no_tool=False)
    assert [row["id"] for row in rebuilt] == ids
