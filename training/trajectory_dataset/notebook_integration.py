from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .cli import AGENT_FLAN_COMPATIBLE_SPLITS


AGENT_FLAN_INTERMEDIATE_FILENAMES = (
    "agent_flan.jsonl",
    "agent_flan.rejected.jsonl",
    "agent_flan.state.json",
)


def prepare_agent_flan_pooled_resume(
    intermediate_dir: str | Path,
    *,
    expected_splits: Iterable[str] = AGENT_FLAN_COMPATIBLE_SPLITS,
) -> dict[str, Any]:
    """Remove only incompatible Agent-FLAN intermediates before pooled resume."""

    intermediate = Path(intermediate_dir)
    output, rejected, state = (
        intermediate / name for name in AGENT_FLAN_INTERMEDIATE_FILENAMES
    )
    expected = list(expected_splits)
    if not output.exists():
        return {
            "regenerated": False,
            "reason": None,
            "removed": [],
            "expected_splits": expected,
        }

    reason: str | None = None
    if not state.exists():
        reason = "state file missing"
    else:
        try:
            payload = json.loads(state.read_text(encoding="utf-8"))
            actual = list(payload.get("splits") or [])
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            reason = f"state unreadable: {type(exc).__name__}"
        else:
            if actual != expected:
                reason = f"source definition changed: {actual} -> {expected}"

    if reason is None:
        return {
            "regenerated": False,
            "reason": None,
            "removed": [],
            "expected_splits": expected,
        }

    removed: list[str] = []
    for path in (output, rejected, state):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return {
        "regenerated": True,
        "reason": reason,
        "removed": removed,
        "expected_splits": expected,
    }


def evaluate_agent_flan_notebook_gate(
    report: dict[str, Any],
    *,
    pool_target: int,
) -> str:
    """Apply the notebook's preferred/degraded Agent-FLAN capacity contract."""

    written = int(report.get("written") or 0)
    if written >= pool_target:
        return "PASS"

    final_mix_gate = report.get("final_mix_gate")
    if isinstance(final_mix_gate, dict) and bool(final_mix_gate.get("valid")):
        return "DEGRADED_POOL_PASS"

    gate = final_mix_gate if isinstance(final_mix_gate, dict) else {}
    required = gate.get("final_required_rows", "unknown")
    source_splits = report.get("source_splits") or []
    source_exhausted = bool(report.get("source_exhausted", False))
    rejected = int(report.get("rejected") or 0)
    rejected_by_reason = report.get("rejected_by_reason") or {}
    raise RuntimeError(
        "FULL AGENT-FLAN capacity FAIL: "
        f"written={written}, required={required}, pool_target={pool_target}, "
        f"source_splits={source_splits}, source_exhausted={source_exhausted}, "
        f"rejected={rejected}, rejected_by_reason={rejected_by_reason}"
    )
