from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IGNORE_INDEX = -100


def build_assistant_only_example(tokenizer, messages: list[dict[str, str]], *, max_length: int) -> dict[str, list[int]]:
    """Tokenize a chat while applying CE loss only to the final assistant turn."""
    if len(messages) < 3 or messages[-1].get("role") != "assistant":
        raise ValueError("assistant-only SFT requires a final assistant message")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    prefix_length = min(len(prefix_ids), len(full_ids))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [IGNORE_INDEX] * prefix_length + full_ids[prefix_length:],
    }


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {name: torch.tensor(values) for name, values in batch.items()}
