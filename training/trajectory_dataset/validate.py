from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .citations import extract_evidence_citations
from .schema import SCHEMA_VERSION, tool_names


VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class ValidationResult:
    valid: list[dict[str, Any]]
    rejected: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return not self.rejected


def _arguments(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    value = function.get("arguments")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _observation_evidence_ids(message: dict[str, Any]) -> set[str]:
    content = message.get("content")
    if not isinstance(content, str):
        return set()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return set()
    rows = value if isinstance(value, list) else value.get("results", []) if isinstance(value, dict) else []
    return {
        str(row.get("chunk_id") or row.get("evidence_id"))
        for row in rows
        if isinstance(row, dict) and (row.get("chunk_id") or row.get("evidence_id"))
    }


def validate_trajectory(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    row_id = str(row.get("id") or "")
    if not row_id:
        errors.append("missing id")
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not str(row.get("source_dataset") or "").strip():
        errors.append("missing source_dataset")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")

    tools = row.get("tools")
    if not isinstance(tools, list):
        errors.append("tools must be a list")
        tools = []
    defined_tools = tool_names(tools)
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return errors + ["messages must be a non-empty list"]
    if messages[0].get("role") not in {"system", "user"}:
        errors.append("first message must be system or user")

    pending: dict[str, str] = {}
    answered: set[str] = set()
    observed_evidence: set[str] = set()
    has_user = False
    final_assistant_content = ""
    final_assistant_index = -1
    last_tool_result_index = -1
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index} must be an object")
            continue
        role = message.get("role")
        if role not in VALID_ROLES:
            errors.append(f"message {index} has invalid role: {role}")
            continue
        if role == "system" and index != 0:
            errors.append(f"system message is only valid at position 0, found at {index}")
        if pending and role != "tool":
            errors.append(
                f"message {index} appears before pending tool results were supplied: {sorted(pending)}"
            )
        content = message.get("content")
        if role == "user":
            has_user = True
            if not isinstance(content, str) or not content.strip():
                errors.append(f"message {index} has an empty user question")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if calls and not isinstance(calls, list):
                errors.append(f"message {index} tool_calls must be a list")
                calls = []
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    errors.append(f"message {index} tool call {call_index} must be an object")
                    continue
                call_id = str(call.get("id") or "")
                function = call.get("function") or {}
                name = str(function.get("name") or "") if isinstance(function, dict) else ""
                arguments = _arguments(call)
                if not call_id:
                    errors.append(f"message {index} tool call {call_index} has no id")
                elif call_id in pending or call_id in answered:
                    errors.append(f"duplicate tool_call_id: {call_id}")
                if not name or name not in defined_tools:
                    errors.append(f"tool call {call_id or call_index} references undefined tool: {name}")
                if arguments is None:
                    errors.append(f"tool call {call_id or call_index} has invalid arguments")
                elif name in {"search_history", "retrieve", "search_wikipedia", "search_web"}:
                    if not str(arguments.get("query") or "").strip():
                        errors.append(f"tool call {call_id or call_index} has an empty query")
                if call_id:
                    pending[call_id] = name
            if isinstance(content, str) and content.strip() and not calls:
                final_assistant_content = content.strip()
                final_assistant_index = index
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or "")
            if call_id not in pending:
                errors.append(f"tool result references unknown tool_call_id: {call_id or '<empty>'}")
            else:
                expected = pending.pop(call_id)
                if name != expected:
                    errors.append(f"tool result {call_id} name {name!r} does not match {expected!r}")
                answered.add(call_id)
            if not isinstance(content, str):
                errors.append(f"tool result {call_id or index} content must be a string")
            observed_evidence.update(_observation_evidence_ids(message))
            last_tool_result_index = index

    if not has_user:
        errors.append("trajectory has no user message")
    if pending:
        errors.append(f"tool calls without results: {sorted(pending)}")
    require_final = (provenance or {}).get("requires_final_answer", True)
    if require_final:
        final_message = messages[-1] if isinstance(messages[-1], dict) else {}
        if (
            not final_assistant_content
            or final_assistant_index < last_tool_result_index
            or final_message.get("role") != "assistant"
            or not str(final_message.get("content") or "").strip()
            or final_message.get("tool_calls")
        ):
            errors.append("missing final assistant answer")
    declared_uses_tools = bool(row.get("uses_tools"))
    actual_uses_tools = any(message.get("tool_calls") for message in messages if isinstance(message, dict))
    if declared_uses_tools != actual_uses_tools:
        errors.append("uses_tools does not match messages")

    if str(row.get("source_dataset", "")).startswith("custom_history") or (provenance or {}).get("grounded"):
        declared_ids = {str(value) for value in (provenance or {}).get("evidence_ids", [])}
        if declared_ids and not declared_ids.issubset(observed_evidence):
            errors.append("provenance evidence_ids are absent from tool observations")
        parsed_citations = extract_evidence_citations(final_assistant_content, observed_evidence)
        if parsed_citations.unknown_ids:
            errors.append(f"grounded final answer cites unknown evidence IDs: {sorted(parsed_citations.unknown_ids)}")
        evidence_like_citations = set(parsed_citations.citations)
        if declared_ids and not evidence_like_citations:
            errors.append("grounded final answer has no internally consistent evidence citation")
    return errors


def validate_rows(rows: Iterable[dict[str, Any]]) -> ValidationResult:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        row_id = str(row.get("id") or "")
        errors = validate_trajectory(row)
        if row_id in seen_ids:
            errors.append(f"duplicate trajectory id: {row_id}")
        if row_id:
            seen_ids.add(row_id)
        if errors:
            rejected.append({"id": row_id or None, "reason": "; ".join(errors), "record": row})
        else:
            valid.append(row)
    return ValidationResult(valid=valid, rejected=rejected)
