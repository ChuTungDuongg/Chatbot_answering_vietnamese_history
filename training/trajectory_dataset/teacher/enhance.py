from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..validate import validate_rows
from .base import Teacher, TeacherRequest


DEFAULT_TEACHER_TASKS = {
    "cause", "significance", "compare", "summary", "multihop", "verification",
}


@dataclass(frozen=True)
class EnhancementResult:
    rows: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    enhanced: int
    fallback: int


def _observed_evidence(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    payloads: list[Any] = []
    ids: list[str] = []
    for message in row.get("messages") or []:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            payload = []
        payloads.append(payload)
        values = payload if isinstance(payload, list) else []
        ids.extend(
            str(value.get("chunk_id") or value.get("evidence_id"))
            for value in values
            if isinstance(value, dict) and (value.get("chunk_id") or value.get("evidence_id"))
        )
    return json.dumps(payloads, ensure_ascii=False, sort_keys=True), tuple(dict.fromkeys(ids))


def _question(row: dict[str, Any]) -> str:
    return next(
        str(message.get("content") or "")
        for message in row.get("messages") or []
        if message.get("role") == "user"
    )


def _citation_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[([^\[\]]+)\]", answer))


def enhance_rows(
    rows: Iterable[dict[str, Any]],
    teacher: Teacher,
    *,
    task_types: set[str] | None = None,
    failure_policy: str = "fallback",
    seed: int = 42,
) -> EnhancementResult:
    if failure_policy not in {"fallback", "reject"}:
        raise ValueError("teacher failure_policy must be fallback or reject")
    selected_tasks = task_types or DEFAULT_TEACHER_TASKS
    originals = list(rows)
    materialized = [copy.deepcopy(row) for row in originals]
    selected: list[tuple[int, TeacherRequest]] = []
    for index, row in enumerate(materialized):
        if str(row.get("task_type")) not in selected_tasks:
            continue
        evidence, allowed_ids = _observed_evidence(row)
        selected.append((index, TeacherRequest(
            task_type=str(row.get("task_type")),
            question=_question(row),
            evidence=evidence,
            allowed_evidence_ids=allowed_ids,
            seed=seed + index,
        )))
    teacher_error = ""
    try:
        responses = teacher.generate([request for _, request in selected]) if selected else []
        if len(responses) != len(selected):
            raise ValueError("teacher must return exactly one answer per selected row")
    except Exception as exc:  # A local model failure must not corrupt deterministic rows.
        teacher_error = f"teacher generation failed: {exc}"
        responses = []

    rejected: list[dict[str, Any]] = []
    enhanced = fallback = 0
    rejected_indices: set[int] = set()
    if teacher_error:
        for index, _request in selected:
            fallback += 1
            if failure_policy == "reject":
                rejected_indices.add(index)
                rejected.append({"id": materialized[index].get("id"), "reason": teacher_error, "record": materialized[index]})
            else:
                materialized[index].setdefault("provenance", {})["teacher_enhanced"] = False
                materialized[index]["provenance"]["teacher_fallback_reason"] = teacher_error
        output = [row for index, row in enumerate(materialized) if index not in rejected_indices]
        return EnhancementResult(rows=output, rejected=rejected, enhanced=0, fallback=fallback)
    for (index, request), response in zip(selected, responses):
        row = materialized[index]
        answer = str(response.answer or "").strip()
        citations = _citation_ids(answer)
        allowed = set(request.allowed_evidence_ids)
        insufficient = "chưa đủ bằng chứng" in answer.casefold()
        reason = ""
        if not answer:
            reason = "teacher returned an empty answer"
        elif not citations.issubset(allowed):
            reason = f"teacher cited unknown evidence IDs: {sorted(citations - allowed)}"
        elif allowed and not citations and not insufficient:
            reason = "teacher answer has no observed evidence citation"
        if not reason:
            original_messages = copy.deepcopy(row["messages"][:-1])
            row["messages"][-1]["content"] = answer
            row["provenance"]["evidence_ids"] = sorted(citations)
            row["provenance"]["teacher_enhanced"] = True
            validation = validate_rows([row])
            if validation.rejected:
                reason = validation.rejected[0]["reason"]
            elif row["messages"][:-1] != original_messages:
                reason = "teacher enhancement changed non-final conversation messages"
        if reason:
            fallback += 1
            if failure_policy == "reject":
                rejected_indices.add(index)
                rejected.append({"id": row.get("id"), "reason": reason, "record": row})
            else:
                materialized[index] = copy.deepcopy(originals[index])
                materialized[index].setdefault("provenance", {})["teacher_enhanced"] = False
                materialized[index]["provenance"]["teacher_fallback_reason"] = reason
        else:
            enhanced += 1
    output = [row for index, row in enumerate(materialized) if index not in rejected_indices]
    return EnhancementResult(rows=output, rejected=rejected, enhanced=enhanced, fallback=fallback)
