from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from training.common.sft import (
    AssistantOnlyTokenStats,
    build_assistant_only_example_with_stats,
)

SOURCE_LINE_PREFIX = "Nguồn được dùng:"
ANSWER_PREFIX = "Trả lời:"
IGNORE_INDEX = -100


def _assistant_parts(text: str) -> tuple[str, str]:
    source = ""
    answer = text.strip()
    match = re.search(r"Nguồn được dùng\s*:\s*(.*?)\n\s*Trả lời\s*:\s*(.*)", text, flags=re.S)
    if match:
        source = f"{SOURCE_LINE_PREFIX} {match.group(1).strip()}\n"
        answer = f"{ANSWER_PREFIX} {match.group(2).strip()}"
    return source, answer


def _apply_chat_template(tokenizer: Any, user_text: str, assistant_text: str) -> str:
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"<|user|>\n{user_text}\n<|assistant|>\n{assistant_text}"


def _assistant_generation_prefix(tokenizer: Any, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|user|>\n{user_text}\n<|assistant|>\n"


def build_rag_training_example(
    tokenizer: Any,
    user_text: str,
    assistant_text: str,
    *,
    max_length: int = 4096,
    source_weight: float = 1.6,
    answer_weight: float = 1.0,
) -> dict[str, list[int] | list[float]]:
    feature, _ = build_rag_training_example_with_stats(
        tokenizer,
        user_text,
        assistant_text,
        max_length=max_length,
        source_weight=source_weight,
        answer_weight=answer_weight,
    )
    return feature


def build_rag_training_example_with_stats(
    tokenizer: Any,
    user_text: str,
    assistant_text: str,
    *,
    max_length: int = 4096,
    source_weight: float = 1.6,
    answer_weight: float = 1.0,
) -> tuple[dict[str, list[int] | list[float]], AssistantOnlyTokenStats]:
    """Build target-preserving Phase-6 labels and weighted assistant CE.

    The complete assistant target is identified before truncation.  Only the
    left side of the user/context prefix may be removed; an oversized target
    fails instead of silently creating an all--100 sample.
    """
    source_text, answer_text = _assistant_parts(assistant_text)
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": source_text + answer_text},
    ]
    core, stats = build_assistant_only_example_with_stats(
        tokenizer, messages, max_length=max_length
    )
    input_ids = core["input_ids"]
    labels = core["labels"]
    assistant_start = next(
        (index for index, label in enumerate(labels) if label != IGNORE_INDEX),
        len(labels),
    )
    supervised_tokens = len(labels) - assistant_start
    if supervised_tokens <= 0:
        raise ValueError("History Answerer SFT invariant failed: zero supervised assistant tokens")
    loss_weights = [0.0] * len(input_ids)
    source_ids = tokenizer(source_text, add_special_tokens=False)["input_ids"] if source_text else []
    source_end = assistant_start + min(len(source_ids), supervised_tokens)

    for idx in range(assistant_start, len(input_ids)):
        loss_weights[idx] = source_weight if idx < source_end else answer_weight

    return {
        "input_ids": input_ids,
        "attention_mask": core["attention_mask"],
        "labels": labels,
        "loss_weights": loss_weights,
    }, stats


def build_instruction_training_example(
    tokenizer: Any,
    user_text: str,
    assistant_text: str,
    *,
    max_length: int = 1024,
    analysis_weight: float = 0.5,
    final_weight: float = 1.0,
) -> dict[str, list[int] | list[float]]:
    core, _ = build_assistant_only_example_with_stats(
        tokenizer,
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        max_length=max_length,
    )
    input_ids = core["input_ids"]
    labels = core["labels"]
    loss_weights = [0.0] * len(input_ids)
    assistant_start = next(
        (index for index, label in enumerate(labels) if label != IGNORE_INDEX),
        len(labels),
    )

    analysis_token = tokenizer("<analysis>", add_special_tokens=False)["input_ids"]
    final_token = tokenizer("<final>", add_special_tokens=False)["input_ids"]
    tail = input_ids[assistant_start:]

    def find_subseq(haystack: list[int], needle: list[int]) -> int | None:
        if not needle:
            return None
        for i in range(0, len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return None

    final_pos = find_subseq(tail, final_token)
    analysis_pos = find_subseq(tail, analysis_token)
    for idx in range(assistant_start, len(input_ids)):
        relative = idx - assistant_start
        if analysis_pos is not None and (final_pos is None or relative < final_pos):
            loss_weights[idx] = analysis_weight
        else:
            loss_weights[idx] = final_weight

    return {
        "input_ids": input_ids,
        "attention_mask": core["attention_mask"],
        "labels": labels,
        "loss_weights": loss_weights,
    }


@dataclass
class WeightedDataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int | float]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
        }
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)
            batch["loss_weights"].append(feature["loss_weights"] + [0.0] * pad_len)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
        }


def weighted_trainer_class():
    from transformers import Trainer

    class WeightedCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch.nn.functional as F

            loss_weights = inputs.pop("loss_weights")
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weights[..., 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view_as(shift_labels)
            denom = shift_weights.sum().clamp_min(1.0)
            loss = (token_loss * shift_weights).sum() / denom
            return (loss, outputs) if return_outputs else loss

    return WeightedCETrainer
