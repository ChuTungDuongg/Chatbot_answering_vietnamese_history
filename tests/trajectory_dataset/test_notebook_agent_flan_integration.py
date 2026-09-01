from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.trajectory_dataset import cli
from training.trajectory_dataset.mix import DEFAULT_MIX_RATIOS, agent_flan_pool_gate
from training.trajectory_dataset.notebook_integration import (
    AGENT_FLAN_INTERMEDIATE_FILENAMES,
    evaluate_agent_flan_notebook_gate,
    prepare_agent_flan_pooled_resume,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "build_trajectory_dataset_colab_v4_4_fast_gpu.ipynb"


def _cell_source(marker: str) -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source") or [])
        for cell in notebook["cells"]
        if marker in "".join(cell.get("source") or [])
    ]
    assert len(matches) == 1
    return matches[0]


def _report(written: int, *, valid: bool) -> dict:
    gate = agent_flan_pool_gate(written, final_max_samples=4000, pool_target=700)
    gate["valid"] = valid
    return {
        "written": written,
        "target_reached": written >= 700,
        "source_exhausted": written < 700,
        "source_splits": list(cli.AGENT_FLAN_COMPATIBLE_SPLITS),
        "rejected": 3,
        "rejected_by_reason": {"fixture": 3},
        "final_mix_gate": gate,
    }


def test_full_notebook_agent_flan_command_uses_auto_and_final_mix_size():
    if not NOTEBOOK_PATH.is_file():
        pytest.skip("legacy V1 Colab notebook is not shipped in this checkout")
    source = _cell_source("FULL AGENT-FLAN (AUTO POOL)")
    compile(source, str(NOTEBOOK_PATH), "exec")
    command = source[source.index("agent_cmd = ["):source.index("agent_proc =")]

    assert '"--split", "auto"' in command
    assert '"--final-max-samples", str(FINAL_MAX_SAMPLES)' in command
    assert '"--split", "agent_instruct_react"' not in command
    assert '"--split", "train"' in source  # Other public jobs remain explicit.
    assert list(cli.AGENT_FLAN_COMPATIBLE_SPLITS) == [
        "agent_instruct_react",
        "toolbench_react_10p",
    ]


def test_explicit_single_split_cli_remains_supported():
    args = cli.build_parser().parse_args([
        "normalize-public",
        "--source", "agent_flan",
        "--split", "agent_instruct_react",
        "--input-jsonl", "fixture.jsonl",
        "--output", "agent.jsonl",
    ])
    assert cli._resolved_public_splits(args) == ("agent_instruct_react",)


def test_stale_single_split_state_regenerates_only_agent_flan_files(tmp_path: Path):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    for name in AGENT_FLAN_INTERMEDIATE_FILENAMES:
        (intermediate / name).write_text("{}" if name.endswith(".json") else "row\n", encoding="utf-8")
    (intermediate / "agent_flan.state.json").write_text(
        json.dumps({"splits": ["agent_instruct_react"]}), encoding="utf-8",
    )
    preserved = {
        name: intermediate / name
        for name in (
            "custom_history.jsonl",
            "multihop.jsonl",
            "vietnam_history.jsonl",
        )
    }
    for path in preserved.values():
        path.write_text("preserve\n", encoding="utf-8")

    result = prepare_agent_flan_pooled_resume(intermediate)

    assert result["regenerated"]
    assert {Path(path).name for path in result["removed"]} == set(AGENT_FLAN_INTERMEDIATE_FILENAMES)
    assert all(not (intermediate / name).exists() for name in AGENT_FLAN_INTERMEDIATE_FILENAMES)
    assert all(path.read_text(encoding="utf-8") == "preserve\n" for path in preserved.values())


def test_compatible_pooled_state_is_preserved(tmp_path: Path):
    intermediate = tmp_path / "intermediate"
    intermediate.mkdir()
    output = intermediate / "agent_flan.jsonl"
    output.write_text("row\n", encoding="utf-8")
    state = intermediate / "agent_flan.state.json"
    state.write_text(
        json.dumps({"splits": list(cli.AGENT_FLAN_COMPATIBLE_SPLITS)}), encoding="utf-8",
    )

    result = prepare_agent_flan_pooled_resume(intermediate)

    assert not result["regenerated"]
    assert output.exists() and state.exists()


def test_agent_flan_gate_accepts_preferred_and_cli_approved_degraded_pool():
    assert evaluate_agent_flan_notebook_gate(_report(700, valid=True), pool_target=700) == "PASS"
    assert evaluate_agent_flan_notebook_gate(_report(528, valid=True), pool_target=700) == "DEGRADED_POOL_PASS"


def test_agent_flan_gate_uses_final_mix_gate_through_480_to_699():
    assert evaluate_agent_flan_notebook_gate(_report(600, valid=True), pool_target=700) == "DEGRADED_POOL_PASS"
    with pytest.raises(RuntimeError, match=r"written=500.*required=480"):
        evaluate_agent_flan_notebook_gate(_report(500, valid=False), pool_target=700)
    with pytest.raises(RuntimeError, match=r"written=479.*source_splits"):
        evaluate_agent_flan_notebook_gate(_report(479, valid=False), pool_target=700)


def test_mix_ratios_remain_unchanged():
    assert DEFAULT_MIX_RATIOS == {
        "custom_history": 0.55,
        "multi_hop_function_calling": 0.17,
        "agent_flan": 0.12,
        "vietnam_history_200k": 0.16,
    }
