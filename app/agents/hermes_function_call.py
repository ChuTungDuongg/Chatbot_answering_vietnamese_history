from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodedFunctionCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class HermesDecodeResult:
    content: str
    tool_calls: tuple[DecodedFunctionCall, ...]
    failures: int = 0
    malformed: tuple[str, ...] = ()


class HermesFunctionCallCodec:
    """One bounded codec for the Qwen3/Hermes ``<tool_call>`` contract.

    Qwen's official chat template serializes semantic ``tool_calls`` messages
    into these frames.  The application keeps that wire representation at this
    boundary only; agent state and training rows remain structured dictionaries.
    """

    open_tag = "<tool_call>"
    close_tag = "</tool_call>"

    def decode(self, text: str, *, allow_python_literal: bool = False) -> HermesDecodeResult:
        remaining = str(text or "")
        content_parts: list[str] = []
        calls: list[DecodedFunctionCall] = []
        malformed: list[str] = []
        failures = 0
        ordinal = 0

        while True:
            start = remaining.find(self.open_tag)
            if start < 0:
                content_parts.append(remaining)
                break
            content_parts.append(remaining[:start])
            body_start = start + len(self.open_tag)
            end = remaining.find(self.close_tag, body_start)
            if end < 0:
                failures += 1
                malformed.append(remaining[start : start + 300])
                break
            body = remaining[body_start:end].strip()
            remaining = remaining[end + len(self.close_tag) :]
            try:
                value = self._parse_value(body, allow_python_literal=allow_python_literal)
                items = value if isinstance(value, list) else [value]
                if not items:
                    raise ValueError("empty tool-call list")
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError("tool call must be an object")
                    function = item.get("function") if isinstance(item.get("function"), dict) else item
                    name = str(function.get("name") or "").strip()
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = self._parse_value(arguments, allow_python_literal=allow_python_literal)
                    if not name or not isinstance(arguments, dict):
                        raise ValueError("tool call requires a name and object arguments")
                    ordinal += 1
                    calls.append(DecodedFunctionCall(
                        id=str(item.get("id") or f"call_{ordinal:04d}"),
                        name=name,
                        arguments=arguments,
                    ))
            except (ValueError, SyntaxError, json.JSONDecodeError):
                failures += 1
                malformed.append(body[:300])

        content = self._clean_content("".join(content_parts))
        return HermesDecodeResult(content, tuple(calls), failures, tuple(malformed))

    @staticmethod
    def _parse_value(value: str, *, allow_python_literal: bool) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if not allow_python_literal:
                raise
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("literal tool payload must be an object or list")
        return parsed

    @staticmethod
    def _clean_content(text: str) -> str:
        value = text
        while True:
            start = value.lower().find("<think>")
            if start < 0:
                break
            end = value.lower().find("</think>", start + len("<think>"))
            value = value[:start] if end < 0 else value[:start] + value[end + len("</think>") :]
        for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"):
            value = value.replace(token, "")
        return value.strip()

