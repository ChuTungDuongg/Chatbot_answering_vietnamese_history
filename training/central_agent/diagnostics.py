from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from training.trajectory_dataset.preprocess import (
    IGNORE_INDEX,
    analyze_truncation,
    build_canonical_sft_example,
)


SPAN_OTHER = 0
SPAN_TOOL_CALL = 1
SPAN_FINAL_ANSWER = 2


def build_test_diagnostic_feature(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    """Add span kinds to the exact canonical SFT feature without changing its labels."""
    truncation = analyze_truncation(tokenizer, row, max_length=max_length)
    feature = build_canonical_sft_example(tokenizer, row, max_length=max_length)
    labels = feature["labels"]
    cut = int(truncation["total_tokens"]) - len(labels)
    if cut < 0:
        raise ValueError("canonical diagnostic feature is longer than its untruncated rendering")
    span_kinds = [SPAN_OTHER] * len(labels)
    for span in truncation["assistant_spans"]:
        if span["is_tool_call"]:
            kind = SPAN_TOOL_CALL
        elif span["is_final_answer"]:
            kind = SPAN_FINAL_ANSWER
        else:
            continue
        start = max(0, int(span["start"]) - cut)
        end = min(len(labels), int(span["end"]) - cut)
        for index in range(start, max(start, end)):
            if labels[index] != IGNORE_INDEX:
                span_kinds[index] = kind
    return {**feature, "span_kinds": span_kinds}


def score_causal_batch(logits: Any, labels: Any, span_kinds: Any) -> list[dict[str, Any]]:
    """Score teacher-forced next-token predictions without retaining batch logits.

    Position ``t`` in causal-LM logits predicts the label at position ``t + 1``.
    The returned counts therefore use shifted labels and exclude IGNORE_INDEX.
    """
    import torch
    import torch.nn.functional as functional

    if logits.ndim != 3 or labels.ndim != 2 or span_kinds.ndim != 2:
        raise ValueError("diagnostic logits/labels/span kinds have invalid ranks")
    if logits.shape[:2] != labels.shape or labels.shape != span_kinds.shape:
        raise ValueError("diagnostic logits/labels/span kinds have incompatible shapes")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_kinds = span_kinds[:, 1:].contiguous()
    token_losses = functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(IGNORE_INDEX)
    predictions = shift_logits.argmax(dim=-1)
    correct = predictions.eq(shift_labels) & valid

    results: list[dict[str, Any]] = []
    for row_index in range(labels.shape[0]):
        row_valid = valid[row_index]
        row_result: dict[str, Any] = {}
        for name, mask in (
            ("supervised", row_valid),
            ("tool_call", row_valid & shift_kinds[row_index].eq(SPAN_TOOL_CALL)),
            ("final_answer", row_valid & shift_kinds[row_index].eq(SPAN_FINAL_ANSWER)),
        ):
            tokens = int(mask.sum().item())
            matches = int(correct[row_index][mask].sum().item())
            row_result[name] = {
                "tokens": tokens,
                "nll_sum": float(token_losses[row_index][mask].double().sum().item()),
                "correct": matches,
                "sequence_exact": bool(tokens and matches == tokens),
            }
        results.append(row_result)
    return results


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _perplexity(nll: float | None) -> float | None:
    if nll is None:
        return None
    try:
        value = math.exp(nll)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


@dataclass
class _Aggregate:
    rows: int = 0
    supervised_tokens: int = 0
    supervised_nll_sum: float = 0.0
    supervised_correct: int = 0
    tool_call_rows: int = 0
    tool_call_tokens: int = 0
    tool_call_nll_sum: float = 0.0
    tool_call_correct: int = 0
    tool_call_exact: int = 0
    final_answer_rows: int = 0
    final_answer_tokens: int = 0
    final_answer_nll_sum: float = 0.0
    final_answer_correct: int = 0
    final_answer_exact: int = 0

    def add(self, metrics: dict[str, Any]) -> None:
        self.rows += 1
        supervised = metrics["supervised"]
        if not supervised["tokens"]:
            raise ValueError("canonical test row has no causally evaluable supervised tokens")
        self.supervised_tokens += supervised["tokens"]
        self.supervised_nll_sum += supervised["nll_sum"]
        self.supervised_correct += supervised["correct"]
        for name in ("tool_call", "final_answer"):
            values = metrics[name]
            if not values["tokens"]:
                continue
            setattr(self, f"{name}_rows", getattr(self, f"{name}_rows") + 1)
            setattr(self, f"{name}_tokens", getattr(self, f"{name}_tokens") + values["tokens"])
            setattr(self, f"{name}_nll_sum", getattr(self, f"{name}_nll_sum") + values["nll_sum"])
            setattr(self, f"{name}_correct", getattr(self, f"{name}_correct") + values["correct"])
            if values["sequence_exact"]:
                setattr(self, f"{name}_exact", getattr(self, f"{name}_exact") + 1)

    def overall(self) -> dict[str, Any]:
        nll = _ratio(self.supervised_nll_sum, self.supervised_tokens)
        return {
            "rows": self.rows,
            "supervised_tokens": self.supervised_tokens,
            "supervised_token_nll": nll,
            "supervised_token_perplexity": _perplexity(nll),
            "supervised_token_accuracy": _ratio(self.supervised_correct, self.supervised_tokens),
        }

    def span(self, name: str) -> dict[str, Any]:
        rows = getattr(self, f"{name}_rows")
        tokens = getattr(self, f"{name}_tokens")
        nll = _ratio(getattr(self, f"{name}_nll_sum"), tokens)
        return {
            "rows": rows,
            "tokens": tokens,
            "token_nll": nll,
            "token_perplexity": _perplexity(nll),
            "token_accuracy": _ratio(getattr(self, f"{name}_correct"), tokens),
            "sequence_exact_match_rate": _ratio(getattr(self, f"{name}_exact"), rows),
        }

    def group(self) -> dict[str, Any]:
        result = self.overall()
        if self.tool_call_tokens:
            result["tool_call_token_accuracy"] = _ratio(
                self.tool_call_correct,
                self.tool_call_tokens,
            )
        if self.final_answer_tokens:
            result["final_answer_token_accuracy"] = _ratio(
                self.final_answer_correct,
                self.final_answer_tokens,
            )
        return result


def _model_inputs(
    model: Any,
    batch: dict[str, Any],
    prepare_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    inputs = {key: batch[key] for key in ("input_ids", "attention_mask")}
    if prepare_inputs is not None:
        return prepare_inputs(inputs)
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def evaluate_teacher_forced_test_diagnostics(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    batch_size: int,
    max_samples: int | None = None,
    prepare_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    identifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream held-out teacher-forced metrics from one shared model forward pass."""
    import torch

    from training.common.sft import AssistantOnlyCollator

    if batch_size < 1:
        raise ValueError("test diagnostic batch size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("test diagnostic max samples must be positive")
    selected_rows = rows if max_samples is None else rows[:max_samples]
    overall = _Aggregate()
    by_task: dict[str, _Aggregate] = {}
    by_source: dict[str, _Aggregate] = {}
    collator = AssistantOnlyCollator(tokenizer.pad_token_id)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(selected_rows), batch_size):
                row_batch = selected_rows[start : start + batch_size]
                features = [
                    build_test_diagnostic_feature(tokenizer, row, max_length=max_length)
                    for row in row_batch
                ]
                tensor_batch = collator([
                    {key: feature[key] for key in ("input_ids", "attention_mask", "labels")}
                    for feature in features
                ])
                padded_length = int(tensor_batch["labels"].shape[1])
                span_kinds = torch.tensor([
                    feature["span_kinds"] + [SPAN_OTHER] * (padded_length - len(feature["span_kinds"]))
                    for feature in features
                ])
                outputs = model(**_model_inputs(model, tensor_batch, prepare_inputs))
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                row_metrics = score_causal_batch(
                    logits,
                    tensor_batch["labels"].to(logits.device),
                    span_kinds.to(logits.device),
                )
                for row, metrics in zip(row_batch, row_metrics):
                    overall.add(metrics)
                    task_type = str(row.get("task_type") or "unknown")
                    source_dataset = str(row.get("source_dataset") or "unknown")
                    by_task.setdefault(task_type, _Aggregate()).add(metrics)
                    by_source.setdefault(source_dataset, _Aggregate()).add(metrics)
                del logits, outputs, tensor_batch, span_kinds
    finally:
        if was_training:
            model.train()

    return {
        "mode": "teacher_forced",
        "sequence_exact_match_scope": "all causally evaluated tokens of that assistant span type per row",
        **overall.overall(),
        "tool_calls": overall.span("tool_call"),
        "final_answers": overall.span("final_answer"),
        "by_task_type": {key: by_task[key].group() for key in sorted(by_task)},
        "by_source_dataset": {key: by_source[key].group() for key in sorted(by_source)},
        "identifiers": dict(identifiers or {}),
    }
