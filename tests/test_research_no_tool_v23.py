from __future__ import annotations

from training.common.datasets import split_rows
from training.research_agent.build_history_trajectories import (
    BOUNDARY_FAMILIES,
    NO_TOOL_FAMILIES,
    build_boundary_samples,
    build_no_tool_samples,
)
from training.research_agent.validate_dataset import validate_rows


def test_no_tool_families_are_diverse_and_finish_without_lookup():
    rows = build_no_tool_samples()
    assert len(rows) == sum(map(len, NO_TOOL_FAMILIES.values())) == 80
    assert {row["metadata"]["no_tool_category"] for row in rows} == set(NO_TOOL_FAMILIES)
    assert all(row["training_target"]["action"] == "finish" for row in rows)
    assert len({row["training_prompt"]["question"] for row in rows}) == len(rows)


def test_conversational_history_boundaries_still_search():
    rows = build_boundary_samples()
    assert len(rows) == sum(map(len, BOUNDARY_FAMILIES.values())) == 12
    assert all(row["training_target"]["tool_name"] == "search_history" for row in rows)
    assert all(row["training_target"]["arguments"]["query"] == row["training_prompt"]["question"] for row in rows)


def test_semantic_families_do_not_cross_grouped_splits():
    rows = build_no_tool_samples() + build_boundary_samples()
    splits = split_rows(
        rows, seed=42, train_ratio=0.70, eval_ratio=0.15,
        group_key="group_id", stratify_key="trajectory_class"
    )
    groups = [
        {row["group_id"] for row in getattr(splits, name)} for name in ("train", "eval", "test")
    ]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    report = validate_rows(rows)
    assert report["near_duplicate_no_tool_cross_split"] == 0
    assert report["no_tool_group_overlap"] == 0
