from __future__ import annotations

from typing import Any


IGNORE_INDEX = -100


def _ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded.get("input_ids")
    if not isinstance(values, list):
        raise ValueError("tokenizer did not return an input_ids list")
    return values


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


def _common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def build_canonical_sft_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Use the official chat template and supervise every assistant action.

    System/user/tool spans remain -100. Assistant tool-call and final-answer
    spans are selected from the same full rendered conversation.
    """
    messages = row.get("messages") or []
    tools = row.get("tools") or []
    if not messages or max_length < 1:
        raise ValueError("canonical SFT requires messages and a positive max_length")
    full_ids = _ids(tokenizer, _render(tokenizer, messages, tools, generation=False))
    labels = [IGNORE_INDEX] * len(full_ids)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        before_ids = _ids(tokenizer, _render(tokenizer, messages[:index], tools, generation=True))
        through_ids = _ids(tokenizer, _render(tokenizer, messages[: index + 1], tools, generation=False))
        start = _common_prefix(before_ids, full_ids)
        end = _common_prefix(through_ids, full_ids)
        if end <= start:
            raise ValueError(f"chat template exposed no assistant target at message {index}")
        labels[start:end] = full_ids[start:end]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("canonical SFT example has zero assistant target tokens")
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("truncation removed every assistant target; increase --max-seq-length")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
