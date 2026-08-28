from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IGNORE_INDEX = -100


@dataclass(frozen=True)
class AssistantOnlyTokenStats:
    prompt_tokens: int
    assistant_tokens: int
    prompt_tokens_kept: int
    assistant_tokens_kept: int
    sequence_tokens: int
    truncated: bool

    @property
    def prompt_truncated(self) -> bool:
        return self.prompt_tokens_kept < self.prompt_tokens

    @property
    def assistant_truncated(self) -> bool:
        return self.assistant_tokens_kept < self.assistant_tokens


def _tokenize_ids(tokenizer, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = tokenized.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("assistant-only SFT tokenization produced an empty token sequence")
    return input_ids


def build_assistant_only_example_with_stats(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int,
) -> tuple[dict[str, list[int]], AssistantOnlyTokenStats]:
    """Tokenize a chat while preserving the complete final assistant target."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("assistant-only SFT requires a final assistant message")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = _tokenize_ids(tokenizer, prefix)
    full_ids = _tokenize_ids(tokenizer, full)

    # Chat templates normally render ``full`` as ``prefix + assistant + EOS``.
    # Comparing the untruncated token streams also handles tokenizers whose
    # boundary tokenization is not quite character-prefix stable.
    prefix_length = 0
    for prefix_token, full_token in zip(prefix_ids, full_ids):
        if prefix_token != full_token:
            break
        prefix_length += 1
    assistant_ids = full_ids[prefix_length:]
    prompt_ids = full_ids[:prefix_length]
    if not assistant_ids:
        raise ValueError("assistant-only SFT produced an empty final assistant target")
    if len(assistant_ids) > max_length:
        raise ValueError(
            "final assistant target alone exceeds max_length; increase max_length or cap the structured target"
        )

    prompt_budget = max_length - len(assistant_ids)
    prompt_ids_kept = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = prompt_ids_kept + assistant_ids
    labels = [IGNORE_INDEX] * len(prompt_ids_kept) + assistant_ids.copy()
    supervised_tokens = sum(label != IGNORE_INDEX for label in labels)
    if supervised_tokens == 0:
        raise ValueError("assistant-only SFT invariant failed: sample has zero supervised tokens")

    feature = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
    stats = AssistantOnlyTokenStats(
        prompt_tokens=len(prompt_ids),
        assistant_tokens=len(assistant_ids),
        prompt_tokens_kept=len(prompt_ids_kept),
        assistant_tokens_kept=len(assistant_ids),
        sequence_tokens=len(input_ids),
        truncated=len(full_ids) > max_length,
    )
    return feature, stats


def build_assistant_only_example(tokenizer, messages: list[dict[str, str]], *, max_length: int) -> dict[str, list[int]]:
    """Build an assistant-only feature with target-preserving truncation."""
    feature, _ = build_assistant_only_example_with_stats(tokenizer, messages, max_length=max_length)
    return feature


def assistant_only_token_stats(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int,
) -> AssistantOnlyTokenStats:
    """Return diagnostics produced by the exact training tokenization path."""
    _, stats = build_assistant_only_example_with_stats(tokenizer, messages, max_length=max_length)
    return stats


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("assistant-only collator received an empty batch")
        for index, feature in enumerate(features):
            if not feature.get("input_ids"):
                raise ValueError(f"assistant-only collator feature {index} has empty input_ids")
            if not any(label != IGNORE_INDEX for label in feature.get("labels", [])):
                raise ValueError(f"assistant-only collator feature {index} has zero supervised tokens")
            if not (
                len(feature["input_ids"])
                == len(feature.get("attention_mask", []))
                == len(feature["labels"])
            ):
                raise ValueError(f"assistant-only collator feature {index} has inconsistent lengths")
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {name: torch.tensor(values) for name, values in batch.items()}
