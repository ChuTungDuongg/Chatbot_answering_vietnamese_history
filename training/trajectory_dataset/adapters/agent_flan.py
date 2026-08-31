from __future__ import annotations

import json
import re
from typing import Any

from ..schema import make_trajectory, tool_call
from .common import (
    AdapterError,
    get_messages,
    normalize_tool_definition,
    provenance,
    semantic_messages,
    trajectory_id,
)


DATASET_ID = "internlm/Agent-FLAN"

ACTION_HEADER = re.compile(r"(?im)^\s*action\s*:\s*\S+")
KNOWN_ACTION_CALL = re.compile(
    r"(?im)^\s*(?:get_relations|get_neighbors|intersection|argmax|argmin|search|click)\s*[\[(]"
)
THOUGHT_HEADER = re.compile(r"(?im)^\s*thought\s*:")
FINAL_ANSWER_HEADER = re.compile(r"(?im)^\s*final\s+answer\s*:\s*")
REACT_HEADER = re.compile(r"(?im)^\s*(action|observation|final\s+answer)\s*:\s*")


def contains_agent_flan_action_syntax(text: Any) -> bool:
    """Detect executable ReAct syntax, not ordinary explanatory use of 'action'."""
    value = str(text or "")
    return bool(ACTION_HEADER.search(value) or KNOWN_ACTION_CALL.search(value))


def contains_agent_flan_thought_target(text: Any) -> bool:
    return bool(THOUGHT_HEADER.search(str(text or "")))


def _split_arguments(value: str) -> list[str]:
    if not value.strip():
        return []
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    for index, character in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
            if depth < 0:
                raise AdapterError("Agent-FLAN action arguments have unbalanced delimiters")
        elif character == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    if quote or depth:
        raise AdapterError("Agent-FLAN action arguments have unbalanced quotes/delimiters")
    parts.append(value[start:].strip())
    if any(not part for part in parts):
        raise AdapterError("Agent-FLAN action contains an empty argument")
    return parts


def _argument_value(value: str) -> Any:
    stripped = value.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        if len(stripped) >= 2 and stripped[0] == stripped[-1] == "'":
            return stripped[1:-1]
        return stripped
    return parsed


def _parse_action(value: str) -> tuple[str, dict[str, Any]]:
    clean = value.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)", clean, flags=re.DOTALL)
    bracket = False
    if match is None:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)\]", clean, flags=re.DOTALL)
        bracket = match is not None
    if match is None:
        raise AdapterError(f"unsafe Agent-FLAN action syntax cannot be converted: {clean[:120]}")
    name, raw_arguments = match.group(1), match.group(2).strip()
    parts = _split_arguments(raw_arguments)
    if bracket:
        if len(parts) > 1:
            raise AdapterError(f"bracket action {name!r} has ambiguous multiple arguments")
        key = "query" if name.casefold() == "search" else "id" if name.casefold() == "click" else "input"
        return name, ({key: _argument_value(parts[0])} if parts else {})
    arguments: dict[str, Any] = {}
    positional = 0
    for part in parts:
        named = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", part, flags=re.DOTALL)
        if named:
            key, raw_value = named.group(1), named.group(2)
        else:
            positional += 1
            key, raw_value = f"arg{positional}", part
        if key in arguments:
            raise AdapterError(f"Agent-FLAN action repeats argument {key!r}")
        arguments[key] = _argument_value(raw_value)
    return name, arguments


def _tool_definition(name: str, argument_names: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Canonical conversion of the Agent-FLAN {name} action.",
            "parameters": {
                "type": "object",
                "properties": {
                    argument: {"description": "Deterministically preserved Agent-FLAN argument."}
                    for argument in argument_names
                },
                "required": argument_names,
                "additionalProperties": False,
            },
        },
    }


def _final_answer_only(content: Any) -> str:
    value = str(content or "").strip()
    matches = list(FINAL_ANSWER_HEADER.finditer(value))
    if matches:
        answer = value[matches[-1].end():].strip()
        if not answer:
            raise AdapterError("Agent-FLAN Final Answer is empty")
        return answer
    if contains_agent_flan_thought_target(value):
        raise AdapterError("Agent-FLAN reasoning target has no separable Final Answer")
    return value


def _react_transcript(raw_messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    leading: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    started = False
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise AdapterError(f"message {index} is not an object")
        role = str(raw.get("role") or raw.get("from") or "").casefold()
        role = {"human": "user", "gpt": "assistant", "function": "tool", "observation": "tool"}.get(role, role)
        content = str(raw.get("content", raw.get("value")) or "").strip()
        if role in {"system", "user"} and not started:
            if not content:
                raise AdapterError(f"message {index} has empty {role} content")
            leading.append({"role": role, "content": content})
            continue
        started = True
        if role == "assistant":
            transcript_parts.append(content)
        elif role == "tool":
            transcript_parts.append(content if re.match(r"(?i)^\s*observation\s*:", content) else f"Observation: {content}")
        elif role == "user":
            # The real agent_instruct_react split encodes environment/KB
            # observations as user turns after the initial query.
            transcript_parts.append(
                content if re.match(r"(?i)^\s*observation\s*:", content) else f"Observation: {content}"
            )
        elif role == "system":
            raise AdapterError("Agent-FLAN ReAct transcript contains a late system turn")
        else:
            raise AdapterError(f"message {index} has unsupported role {role!r}")
    if not any(message["role"] == "user" for message in leading):
        raise AdapterError("Agent-FLAN ReAct transcript has no leading user message")
    transcript = "\n".join(part for part in transcript_parts if part)
    headers = list(REACT_HEADER.finditer(transcript))
    if not headers or headers[0].group(1).casefold() != "action":
        raise AdapterError("Agent-FLAN ReAct transcript does not start with a convertible Action")
    messages = list(leading)
    tool_arguments: dict[str, list[str]] = {}
    pending: tuple[str, str] | None = None
    final_seen = False
    for header_index, header in enumerate(headers):
        kind = " ".join(header.group(1).casefold().split())
        end = headers[header_index + 1].start() if header_index + 1 < len(headers) else len(transcript)
        body = transcript[header.end():end].strip()
        if kind == "action":
            if pending is not None or final_seen:
                raise AdapterError("Agent-FLAN actions/observations are not safely paired")
            name, arguments = _parse_action(body)
            call_id = f"call_agent_flan_{header_index + 1:04d}"
            messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call(call_id, name, arguments)]})
            pending = (call_id, name)
            names = tool_arguments.setdefault(name, [])
            names.extend(key for key in arguments if key not in names)
        elif kind == "observation":
            if pending is None or final_seen:
                raise AdapterError("Agent-FLAN observation has no safely paired action")
            body = re.split(r"(?im)^\s*thought\s*:", body, maxsplit=1)[0].strip()
            if not body:
                raise AdapterError("Agent-FLAN observation is empty")
            call_id, name = pending
            messages.append({"role": "tool", "name": name, "tool_call_id": call_id, "content": body})
            pending = None
        else:
            if pending is not None or final_seen:
                raise AdapterError("Agent-FLAN Final Answer appears before all observations")
            answer = _final_answer_only(f"Final Answer: {body}")
            messages.append({"role": "assistant", "content": answer})
            final_seen = True
    if pending is not None or not final_seen:
        raise AdapterError("Agent-FLAN ReAct transcript lacks a paired observation or Final Answer")
    tools = [_tool_definition(name, arguments) for name, arguments in tool_arguments.items()]
    return messages, tools


def normalize_agent_flan(
    row: dict[str, Any],
    *,
    index: int = 0,
    split: str = "train",
    include_reasoning: bool = False,
) -> dict[str, Any]:
    del include_reasoning  # Agent-FLAN hidden reasoning is never a canonical target.
    raw_tools = row.get("tools") or row.get("functions") or []
    if raw_tools and not isinstance(raw_tools, list):
        raise AdapterError("Agent-FLAN tools/functions must be a list")
    raw_messages = get_messages(row)
    has_literal_actions = any(
        isinstance(message, dict)
        and str(message.get("role") or message.get("from") or "").casefold() in {"assistant", "gpt"}
        and contains_agent_flan_action_syntax(message.get("content", message.get("value")))
        for message in raw_messages
    )
    converted_react = has_literal_actions and not any(
        isinstance(message, dict) and (message.get("tool_calls") or message.get("function_call"))
        for message in raw_messages
    )
    if converted_react:
        if raw_tools:
            raise AdapterError("textual Agent-FLAN actions with separate tool definitions are ambiguous")
        messages, tools = _react_transcript(raw_messages)
    else:
        tools = [normalize_tool_definition(tool) for tool in raw_tools]
        messages = semantic_messages(raw_messages, include_reasoning=False)
        for message in messages:
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                content = _final_answer_only(message.get("content"))
                if not content:
                    raise AdapterError("Agent-FLAN assistant final answer is empty")
                message["content"] = content
    called_names = {
        call["function"]["name"]
        for message in messages
        for call in message.get("tool_calls", [])
    }
    defined_names = {tool["function"]["name"] for tool in tools}
    if called_names - defined_names:
        raise AdapterError(
            "Agent-FLAN row contains semantic tool calls but no matching tool definitions: "
            f"{sorted(called_names - defined_names)}"
        )
    return make_trajectory(
        trajectory_id=trajectory_id(DATASET_ID, row, index),
        source_dataset="agent_flan",
        task_type="generic_agent_tool_behavior" if called_names else "generic_no_tool_behavior",
        messages=messages,
        tools=tools,
        difficulty=str(row.get("difficulty") or "medium"),
        provenance=provenance(
            dataset_id=DATASET_ID,
            split=split,
            row=row,
            index=index,
            license_name="Apache-2.0",
            transformations=[
                "roles_to_canonical",
                "text_actions_to_canonical_tools" if converted_react else "structured_tools_preserved",
                "reasoning_removed",
            ],
        ),
    )
