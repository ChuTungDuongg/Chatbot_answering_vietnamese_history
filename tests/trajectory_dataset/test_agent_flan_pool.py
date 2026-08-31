from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from training.trajectory_dataset import cli
from training.trajectory_dataset.adapters.agent_flan import normalize_agent_flan
from training.trajectory_dataset.adapters.common import AdapterError
from training.trajectory_dataset.io_utils import atomic_write_jsonl, read_jsonl
from training.trajectory_dataset.dedup import deduplicate
from training.trajectory_dataset.mix import (
    DEFAULT_MIX_RATIOS,
    agent_flan_pool_gate,
    mix_sources,
    required_source_rows,
)
from training.trajectory_dataset.schema import make_trajectory
from training.trajectory_dataset.validate import validate_rows


def _plain_agent_row(row_id: str, question: str) -> dict:
    return {
        "id": row_id,
        "conversation": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Answer for {row_id}."},
        ],
    }


def _toolbench_row(row_id: str, question: str, *, final: bool = True) -> dict:
    conversation = [
        {"role": "system", "content": "Tool protocol with reasoning fields."},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": json.dumps({
                "InnerThought": "This reasoning must not be supervised.",
                "Action": "example_api.lookup.item",
                "Parameters": {"item_id": 7},
            }),
        },
        {"role": "user", "content": json.dumps({"response": {"name": "fixture"}})},
    ]
    if final:
        conversation.append({
            "role": "assistant",
            "content": json.dumps({
                "cot": "More hidden reasoning.",
                "action": "FinalAction",
                "Arguments": {"return_type": "give_answer", "final_answer": "Fixture final answer."},
            }),
        })
    return {"id": row_id, "conversation": conversation}


def _normalization_args(tmp_path: Path, *extra: str):
    return cli.build_parser().parse_args([
        "normalize-public", "--source", "agent_flan",
        "--output", str(tmp_path / "agent.jsonl"),
        "--rejected-output", str(tmp_path / "agent.rejected.jsonl"),
        "--max-samples", "3", "--max-attempts", "20",
        *extra,
    ])


def test_toolbench_json_actions_convert_without_reasoning_targets():
    row = normalize_agent_flan(
        _toolbench_row("tb-1", "Look up item 7."),
        split="toolbench_react_10p",
    )
    assert validate_rows([row]).ok
    assert row["messages"][-1] == {"role": "assistant", "content": "Fixture final answer."}
    calls = [call for message in row["messages"] for call in message.get("tool_calls", [])]
    assert calls[0]["function"] == {
        "name": "example_api_lookup_item", "arguments": {"item_id": 7},
    }
    assert row["provenance"]["tool_name_map"] == {
        "example_api.lookup.item": "example_api_lookup_item",
    }
    assert all(
        "thought" not in str(message.get("content") or "").casefold()
        for message in row["messages"] if message["role"] == "assistant"
    )


def test_malformed_sql_ambiguous_observation_and_missing_final_remain_rejected():
    malformed_sql = {
        "id": "sql",
        "conversation": [
            {"role": "user", "content": "Query the database."},
            {"role": "assistant", "content": "Action: Operation <SQL SELECT * FROM users"},
        ],
    }
    with pytest.raises(AdapterError, match="unsafe Agent-FLAN action syntax"):
        normalize_agent_flan(malformed_sql, split="agent_instruct_react")

    ambiguous = {
        "id": "ambiguous",
        "conversation": [
            {"role": "user", "content": "Initial request."},
            {"role": "user", "content": '{"response": "orphan"}'},
        ],
    }
    with pytest.raises(AdapterError, match="observation has no safely paired action"):
        normalize_agent_flan(ambiguous, split="toolbench_react_10p")

    with pytest.raises(AdapterError, match="lacks a paired observation or Final Answer"):
        normalize_agent_flan(
            _toolbench_row("missing-final", "Use the tool.", final=False),
            split="toolbench_react_10p",
        )


def test_auto_pool_is_deterministic_and_deduplicates_globally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    sources = {
        "agent_instruct_react": [
            _plain_agent_row("a-shared", "Shared request."),
            _plain_agent_row("a-only", "Agent-only request."),
        ],
        "toolbench_react_10p": [
            _toolbench_row("tb-shared", "Shared request."),
            # Deliberately reuse a source-local ID from the first split. Pool
            # identity must be split-namespaced rather than silently dropping it.
            _toolbench_row("a-only", "ToolBench-only request."),
            _toolbench_row("tb-extra", "Unused after target request."),
        ],
    }

    def fake_rows(args, *, split):
        del args
        yield from sources[split]

    monkeypatch.setattr(cli, "_public_rows", fake_rows)
    first_args = _normalization_args(tmp_path / "first", "--split", "auto")
    second_args = _normalization_args(tmp_path / "second", "--split", "auto")
    first = cli._normalize_public(first_args)
    second = cli._normalize_public(second_args)

    assert first["target_reached"] and not first["source_exhausted"]
    assert first["source_splits"] == list(cli.AGENT_FLAN_COMPATIBLE_SPLITS)
    assert first["source_breakdown"]["agent_instruct_react"]["source_exhausted"]
    assert first["rejected_by_reason"] == {"duplicate normalized user question": 1}
    assert read_jsonl(first_args.output) == read_jsonl(second_args.output)
    assert len({row["id"] for row in read_jsonl(first_args.output)}) == 3
    assert first["source_breakdown"] == second["source_breakdown"]


def test_source_exhaustion_and_max_attempt_exhaustion_are_distinct(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    atomic_write_jsonl(source, [_plain_agent_row("only", "Only request.")])
    exhausted_args = _normalization_args(
        tmp_path / "exhausted", "--split", "agent_instruct_react",
        "--input-jsonl", str(source),
    )
    exhausted = cli._normalize_public(exhausted_args)
    assert exhausted["written"] == 1
    assert exhausted["source_exhausted"] and not exhausted["target_reached"]
    assert not exhausted["hit_max_attempts"] and exhausted["stop_reason"] == "source_exhausted"

    atomic_write_jsonl(source, [
        _plain_agent_row("one", "Request one."),
        _plain_agent_row("two", "Request two."),
    ])
    bounded_args = _normalization_args(
        tmp_path / "bounded", "--split", "agent_instruct_react",
        "--input-jsonl", str(source), "--max-attempts", "1",
    )
    bounded = cli._normalize_public(bounded_args)
    assert bounded["written"] == 1 and bounded["hit_max_attempts"]
    assert not bounded["source_exhausted"] and not bounded["target_reached"]
    assert bounded["stop_reason"] == "max_attempts"


def test_auto_pool_resume_uses_state_and_does_not_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    sources = {
        "agent_instruct_react": [
            _plain_agent_row("a-1", "Request one."),
            _plain_agent_row("a-2", "Request two."),
            _plain_agent_row("a-3", "Request three."),
        ],
        "toolbench_react_10p": [],
    }

    def fake_rows(args, *, split):
        del args
        yield from sources[split]

    monkeypatch.setattr(cli, "_public_rows", fake_rows)
    first_args = _normalization_args(tmp_path, "--split", "auto", "--max-samples", "2")
    first = cli._normalize_public(first_args)
    assert first["written"] == 2 and Path(first["state_output"]).exists()

    resumed_args = _normalization_args(
        tmp_path, "--split", "auto", "--max-samples", "3", "--resume",
    )
    resumed = cli._normalize_public(resumed_args)
    rows = read_jsonl(resumed_args.output)
    assert resumed["written"] == 3 and resumed["written_this_run"] == 1
    assert resumed["resume_skipped"] == 2
    assert len(rows) == len({row["id"] for row in rows}) == 3


def test_legacy_single_split_state_cannot_be_resumed_as_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    sources = {
        "agent_instruct_react": [_plain_agent_row("legacy", "Legacy request.")],
        "toolbench_react_10p": [],
    }

    def fake_rows(args, *, split):
        del args
        yield from sources[split]

    monkeypatch.setattr(cli, "_public_rows", fake_rows)
    explicit = _normalization_args(tmp_path, "--split", "agent_instruct_react")
    cli._normalize_public(explicit)
    pooled_resume = _normalization_args(tmp_path, "--split", "auto", "--resume")
    with pytest.raises(RuntimeError, match="Regenerate only"):
        cli._normalize_public(pooled_resume)


def test_pool_capacity_gate_requires_480_and_preserves_mix_ratios():
    assert DEFAULT_MIX_RATIOS == {
        "custom_history": 0.55,
        "multi_hop_function_calling": 0.17,
        "agent_flan": 0.12,
        "vietnam_history_200k": 0.16,
    }
    assert required_source_rows("agent_flan", final_max_samples=4000) == 480
    assert not agent_flan_pool_gate(479, final_max_samples=4000)["valid"]
    quota_only = agent_flan_pool_gate(480, final_max_samples=4000)
    assert quota_only["final_mix_quota_satisfied"] and not quota_only["valid"]
    assert quota_only["degraded_minimum_rows"] == 528
    degraded = agent_flan_pool_gate(528, final_max_samples=4000)
    assert degraded["valid"] and degraded["degraded_pool"]
    preferred = agent_flan_pool_gate(700, final_max_samples=4000)
    assert preferred["valid"] and preferred["preferred_target_reached"]


def test_normalize_public_cli_fails_when_final_agent_capacity_is_unsafe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    source = tmp_path / "insufficient.jsonl"
    atomic_write_jsonl(source, [_plain_agent_row("only", "Only request.")])
    exit_code = cli.main([
        "normalize-public", "--source", "agent_flan",
        "--split", "agent_instruct_react", "--input-jsonl", str(source),
        "--output", str(tmp_path / "agent.jsonl"),
        "--rejected-output", str(tmp_path / "agent.rejected.jsonl"),
        "--max-samples", "700", "--max-attempts", "10500",
        "--final-max-samples", "4000",
    ])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["source_exhausted"] and not report["target_reached"]
    assert report["final_mix_gate"]["final_required_rows"] == 480
    assert not report["final_mix_gate"]["valid"]


def test_unique_700_row_agent_pool_can_fill_exact_final_mix_after_dedup():
    counts = {
        "custom_history": 2200,
        "multi_hop_function_calling": 680,
        "agent_flan": 700,
        "vietnam_history_200k": 640,
    }
    sources = {}
    for source, count in counts.items():
        sources[source] = [
            make_trajectory(
                trajectory_id=f"{source}-{index}", source_dataset=source,
                task_type="fixture", messages=[
                    {"role": "user", "content": f"Unique {source} question {index}."},
                    {"role": "assistant", "content": "Answer."},
                ],
                provenance={"requires_final_answer": True, "source_group": f"{source}-{index}"},
            )
            for index in range(count)
        ]
    mixed = mix_sources(sources, DEFAULT_MIX_RATIOS, seed=42, max_total=4000)
    deduped = deduplicate(mixed)
    assert not deduped.rejected and len(deduped.rows) == 4000
    assert Counter(row["source_dataset"] for row in deduped.rows) == {
        "custom_history": 2200,
        "multi_hop_function_calling": 680,
        "agent_flan": 480,
        "vietnam_history_200k": 640,
    }


def test_single_split_cli_remains_backward_compatible(tmp_path: Path):
    source = tmp_path / "single.jsonl"
    atomic_write_jsonl(source, [_plain_agent_row("single", "Single split request.")])
    args = _normalization_args(
        tmp_path, "--split", "agent_instruct_react", "--input-jsonl", str(source),
    )
    result = cli._normalize_public(args)
    assert result["source_splits"] == ["agent_instruct_react"]
    assert result["written"] == 1 and validate_rows(read_jsonl(args.output)).ok
