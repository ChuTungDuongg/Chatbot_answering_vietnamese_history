from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any

from app.agents.common.hermes_function_call import HermesFunctionCallCodec

from training.trajectory_dataset.schema import (
    CENTRAL_V2_SYSTEM_PROMPT,
    QWEN3_TOOL_TEMPLATE_CONTRACT,
    make_trajectory,
    tool_call,
    tool_names,
)
from training.trajectory_dataset.adapters.common import AdapterError, normalize_tool_definition, provenance, trajectory_id


HERMES_DATASET_ID = "NousResearch/hermes-function-calling-v1"
HERMES_FUNCTION_FILES = (
    "func-calling-singleturn.json",
    "func-calling.json",
    "glaive-function-calling-5k.json",
)
HERMES_JSON_MODE_FILES = ("json-mode-agentic.json", "json-mode-singleturn.json")
_CODEC = HermesFunctionCallCodec()
_HIDDEN_REASONING = re.compile(r"(?:^|\n)\s*(?:Thought|Action|Reasoning|Scratchpad)\s*:", re.I)


def _safe_value(value: Any) -> Any:
    if not isinstance(value, str):
        return copy.deepcopy(value)
    text = value.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise AdapterError("Hermes tools are neither valid JSON nor a safe Python literal") from exc


def _tools_block(system_text: str) -> str | None:
    start = system_text.find("<tools>")
    if start < 0:
        return None
    end = system_text.find("</tools>", start + len("<tools>"))
    if end < 0:
        raise AdapterError("Hermes system message has an unterminated <tools> block")
    return system_text[start + len("<tools>") : end].strip()


def _normalize_tools(row: dict[str, Any], raw_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_tools = row.get("tools")
    if raw_tools in (None, "", []):
        system_text = "\n".join(
            str(message.get("value", message.get("content", "")))
            for message in raw_messages
            if str(message.get("from", message.get("role", ""))).casefold() == "system"
        )
        block = _tools_block(system_text)
        if block is None:
            raise AdapterError("Hermes row has neither a tools column nor a <tools> system block")
        raw_tools = block
    parsed = _safe_value(raw_tools)
    if isinstance(parsed, dict) and isinstance(parsed.get("tools"), list):
        parsed = parsed["tools"]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise AdapterError("Hermes tools must be a non-empty list")
    normalized = [normalize_tool_definition(item) for item in parsed if isinstance(item, dict)]
    if len(normalized) != len(parsed):
        raise AdapterError("Hermes tools contain a non-object definition")
    names = [item["function"]["name"] for item in normalized]
    if len(names) != len(set(names)):
        raise AdapterError("Hermes tools contain duplicate function names")
    return normalized


def _raw_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("conversations") or row.get("messages") or row.get("conversation")
    if not isinstance(value, list) or not value:
        raise AdapterError("Hermes row has no ShareGPT conversation")
    if not all(isinstance(message, dict) for message in value):
        raise AdapterError("Hermes conversation contains a non-object message")
    return value


def _tool_response_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    remaining = text
    while True:
        start = remaining.find("<tool_response>")
        if start < 0:
            break
        end = remaining.find("</tool_response>", start + len("<tool_response>"))
        if end < 0:
            raise AdapterError("Hermes tool response has an unterminated wrapper")
        outside = (remaining[:start] + remaining[end + len("</tool_response>") :]).strip()
        if outside and "<tool_response>" not in outside:
            raise AdapterError("Hermes tool response mixes protocol text with an observation")
        bodies.append(remaining[start + len("<tool_response>") : end].strip())
        remaining = remaining[end + len("</tool_response>") :]
    return bodies or [text.strip()]


def normalize_hermes_function_calling(
    row: dict[str, Any],
    *,
    index: int,
    split: str = "train",
    source_file: str | None = None,
) -> dict[str, Any]:
    source_variant = source_file or str(row.get("source_file") or split)
    folded_variant = source_variant.casefold().replace("_", "-")
    if "json-mode" in folded_variant:
        raise AdapterError("Hermes JSON-mode samples are excluded from the Central V2 function-call mix")

    raw_messages = _raw_messages(row)
    tools = _normalize_tools(row, raw_messages)
    defined = tool_names(tools)
    messages: list[dict[str, Any]] = [{"role": "system", "content": CENTRAL_V2_SYSTEM_PROMPT}]
    pending: list[tuple[str, str]] = []
    call_number = 0
    structured_calls = 0
    final_answers = 0
    observed_results = 0

    for message_index, raw in enumerate(raw_messages):
        raw_role = str(raw.get("from", raw.get("role", ""))).casefold()
        role = {
            "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
            "function": "tool", "tool": "tool", "observation": "tool", "system": "system",
        }.get(raw_role)
        if role is None:
            raise AdapterError(f"Hermes message {message_index} has unsupported role {raw_role!r}")
        content = str(raw.get("value", raw.get("content", "")) or "")
        if role == "system":
            continue
        if role == "user":
            if pending:
                raise AdapterError("Hermes user turn appears before pending tool results")
            if not content.strip():
                raise AdapterError("Hermes user turn is empty")
            messages.append({"role": "user", "content": content.strip()})
            continue
        if role == "tool":
            bodies = _tool_response_bodies(content)
            for body in bodies:
                if not pending:
                    raise AdapterError("Hermes tool observation cannot be paired to a tool call")
                call_id, expected_name = pending.pop(0)
                explicit_name = str(raw.get("name") or expected_name)
                if explicit_name != expected_name:
                    raise AdapterError("Hermes tool observation name does not match its pending call")
                messages.append({
                    "role": "tool",
                    "name": expected_name,
                    "tool_call_id": call_id,
                    "content": body,
                })
                observed_results += 1
            continue

        if _HIDDEN_REASONING.search(content) or "<think>" in content.casefold():
            raise AdapterError("Hermes Thought/Action/scratchpad text would become assistant supervision")
        explicit_calls = raw.get("tool_calls")
        if explicit_calls:
            if not isinstance(explicit_calls, list):
                raise AdapterError("Hermes structured tool_calls must be a list")
            decoded_calls = []
            for item in explicit_calls:
                if not isinstance(item, dict):
                    raise AdapterError("Hermes structured tool call is not an object")
                function = item.get("function") if isinstance(item.get("function"), dict) else item
                name = str(function.get("name") or "").strip()
                arguments = _safe_value(function.get("arguments") or {})
                if not name or not isinstance(arguments, dict):
                    raise AdapterError("Hermes structured tool call has malformed arguments")
                decoded_calls.append((name, arguments))
            outside_content = content.strip()
        else:
            decoded = _CODEC.decode(content, allow_python_literal=True)
            if decoded.failures:
                raise AdapterError("Hermes assistant tool call is malformed")
            decoded_calls = [(call.name, call.arguments) for call in decoded.tool_calls]
            outside_content = decoded.content.strip()

        if decoded_calls:
            if pending:
                raise AdapterError("Hermes assistant issues a new call before prior results")
            if outside_content:
                raise AdapterError("Hermes protocol text remains beside a structured tool call")
            canonical_calls: list[dict[str, Any]] = []
            for name, arguments in decoded_calls:
                if name not in defined:
                    raise AdapterError(f"Hermes tool call references undefined tool: {name}")
                call_number += 1
                call_id = f"call_{call_number:04d}"
                canonical_calls.append(tool_call(call_id, name, arguments))
                pending.append((call_id, name))
                structured_calls += 1
            messages.append({"role": "assistant", "content": None, "tool_calls": canonical_calls})
        else:
            if pending:
                raise AdapterError("Hermes final response appears before pending tool-call resolution")
            if not outside_content:
                raise AdapterError("Hermes assistant text is empty")
            messages.append({"role": "assistant", "content": outside_content})
            final_answers += 1

    terminal_tool_call_only = bool(pending and messages[-1].get("tool_calls"))
    terminal_call_allowed = source_variant.replace("\\", "/").rsplit("/", 1)[-1] == HERMES_FUNCTION_FILES[0]
    if pending and not (terminal_tool_call_only and terminal_call_allowed):
        raise AdapterError("Hermes pending tool call has no result where a result is required")
    if not structured_calls:
        raise AdapterError("Hermes row contains no structured function call; JSON-mode/plain text excluded")
    if not terminal_tool_call_only and not final_answers:
        raise AdapterError("Hermes resolved tool trajectory has no final assistant response")

    source_provenance = provenance(
        dataset_id=HERMES_DATASET_ID,
        split=split,
        row=row,
        index=index,
        license_name="apache-2.0",
        transformations=[
            "extract_safe_tools", "strip_hermes_protocol_text", "canonicalize_structured_tool_calls",
            "drop_hidden_reasoning", "stable_tool_call_ids", "qwen3_enable_thinking_false",
        ],
    )
    source_provenance.update({
        "source_file": source_variant,
        "requires_final_answer": not (terminal_tool_call_only and terminal_call_allowed),
        "terminal_tool_call_only": terminal_tool_call_only and terminal_call_allowed,
        "observed_tool_results": observed_results,
        "chat_template_contract": copy.deepcopy(QWEN3_TOOL_TEMPLATE_CONTRACT),
    })
    return make_trajectory(
        trajectory_id=trajectory_id(HERMES_DATASET_ID, row, index),
        source_dataset="hermes_function_calling",
        task_type="function_calling",
        messages=messages,
        tools=tools,
        provenance=source_provenance,
    )
