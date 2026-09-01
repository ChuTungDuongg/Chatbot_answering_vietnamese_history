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
    if not messages:
        return ""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        # Lightweight legacy test tokenizers may predate the Qwen kwarg. Real
        # Central V2 preprocessing always uses a tokenizer that accepts it.
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def _common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _message_span(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    full_ids: list[int],
    index: int,
) -> tuple[int, int]:
    """Locate one rendered message without assuming a generation-prompt shape.

    Qwen3's generation prompt is not necessarily a literal prefix of a fully
    rendered assistant tool-call turn. Both training and truncation auditing
    therefore derive boundaries from the same official, non-generation render.
    """
    before_ids = _ids(tokenizer, _render(tokenizer, messages[:index], tools, generation=False))
    through_ids = _ids(tokenizer, _render(tokenizer, messages[: index + 1], tools, generation=False))
    return _common_prefix(before_ids, full_ids), _common_prefix(through_ids, full_ids)


def analyze_truncation(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    """Report user and assistant spans that left truncation would damage."""
    messages = row.get("messages") or []
    tools = row.get("tools") or []
    full_ids = _ids(tokenizer, _render(tokenizer, messages, tools, generation=False))
    cut = max(0, len(full_ids) - max_length)
    initial_user_span: tuple[int, int] | None = None
    assistant_spans: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user" and initial_user_span is None:
            initial_user_span = _message_span(tokenizer, messages, tools, full_ids, index)
        if role != "assistant":
            continue
        start, end = _message_span(tokenizer, messages, tools, full_ids, index)
        assistant_spans.append({
            "message_index": index,
            "start": start,
            "end": end,
            "is_tool_call": bool(message.get("tool_calls")),
            "is_final_answer": index == len(messages) - 1 and bool(str(message.get("content") or "").strip()),
            "fully_lost": end <= cut,
            "partially_lost": start < cut < end,
        })
    user_lost = bool(initial_user_span and initial_user_span[0] < cut)
    damaged = [span for span in assistant_spans if span["fully_lost"] or span["partially_lost"]]
    return {
        "total_tokens": len(full_ids),
        "max_length": max_length,
        "truncated": cut > 0,
        "truncation_start": cut,
        "initial_user_span": initial_user_span,
        "initial_user_lost": user_lost,
        "assistant_spans": assistant_spans,
        "lost_assistant_targets": len(damaged),
        "lost_tool_call_targets": sum(span["is_tool_call"] for span in damaged),
        "final_assistant_lost": any(span["is_final_answer"] for span in damaged),
        "all_assistant_supervision_lost": bool(assistant_spans) and len(damaged) == len(assistant_spans),
    }


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
        start, end = _message_span(tokenizer, messages, tools, full_ids, index)
        if end <= start:
            raise ValueError(f"chat template exposed no assistant target at message {index}")
        labels[start:end] = full_ids[start:end]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("canonical SFT example has zero assistant target tokens")
    if len(full_ids) > max_length:
        truncation = analyze_truncation(tokenizer, row, max_length=max_length)
        if truncation["initial_user_lost"] or truncation["lost_assistant_targets"]:
            raise ValueError(
                "left truncation would remove the initial user question or assistant action supervision; "
                "compact observations or increase --max-seq-length"
            )
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("truncation removed every assistant target; increase --max-seq-length")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def assistant_labeled_token_counts(tokenizer: Any, row: dict[str, Any]) -> dict[str, int]:
    """Count exact Qwen-rendered assistant targets by semantic target type."""
    messages = row.get("messages") or []
    tools = row.get("tools") or []
    full_ids = _ids(tokenizer, _render(tokenizer, messages, tools, generation=False))
    tool_call_tokens = 0
    final_answer_tokens = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start, end = _message_span(tokenizer, messages, tools, full_ids, index)
        count = max(0, end - start)
        if message.get("tool_calls"):
            tool_call_tokens += count
        else:
            final_answer_tokens += count
    return {
        "assistant_tool_call_labeled_tokens": tool_call_tokens,
        "assistant_final_answer_labeled_tokens": final_answer_tokens,
        "total_assistant_labeled_tokens": tool_call_tokens + final_answer_tokens,
        "trajectory_tokens": len(full_ids),
    }
