from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training.central_agent import engine
from training.central_agent.config import parse_args, validate_args
from training.central_agent.data import DatasetSplit, ResolvedPaths
from training.central_agent.diagnostics import (
    SPAN_FINAL_ANSWER,
    SPAN_OTHER,
    SPAN_TOOL_CALL,
    build_test_diagnostic_feature,
    evaluate_teacher_forced_test_diagnostics,
    score_causal_batch,
)
from training.trajectory_dataset.preprocess import (
    IGNORE_INDEX,
    analyze_truncation,
    build_canonical_sft_example,
)


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        assert tokenize is False
        parts = []
        for message in messages:
            payload = json.dumps(
                {key: value for key, value in message.items() if key != "role"},
                ensure_ascii=False,
                sort_keys=True,
            )
            parts.append(f"<{message['role']}>{payload}</{message['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def assistant_call(call_id: str, query: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "search_history",
            "arguments": json.dumps({"query": query, "top_k": 2}),
        },
    }


def diagnostic_row(
    row_id: str,
    *,
    task_type: str,
    source_dataset: str,
    tool_marker: str = "",
    final_marker: str = "",
    two_calls: bool = False,
    system_prefix: str = "rules",
) -> dict:
    messages = [
        {"role": "system", "content": system_prefix},
        {"role": "user", "content": f"question {row_id}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [assistant_call(f"{row_id}-call-1", f"query {tool_marker}")],
        },
        {
            "role": "tool",
            "name": "search_history",
            "tool_call_id": f"{row_id}-call-1",
            "content": "evidence one",
        },
    ]
    if two_calls:
        messages.extend([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [assistant_call(f"{row_id}-call-2", "second query")],
            },
            {
                "role": "tool",
                "name": "search_history",
                "tool_call_id": f"{row_id}-call-2",
                "content": "evidence two",
            },
        ])
    messages.append({"role": "assistant", "content": f"final answer {final_marker}"})
    return {
        "id": row_id,
        "source_dataset": source_dataset,
        "task_type": task_type,
        "tools": [{"type": "function", "function": {"name": "search_history"}}],
        "messages": messages,
        "provenance": {"requires_final_answer": True, "source_group": row_id},
    }


class NextTokenModel(torch.nn.Module):
    def __init__(self, *, wrong_target_ids: set[int] | None = None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.wrong_target_ids = set(wrong_target_ids or set())
        self.forward_calls = 0
        self.inference_mode_seen: list[bool] = []

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        self.inference_mode_seen.append(not torch.is_grad_enabled())
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 256), -20.0, device=input_ids.device)
        for batch_index in range(batch):
            for position in range(max(0, length - 1)):
                target = int(input_ids[batch_index, position + 1].item())
                prediction = (target + 1) % 256 if target in self.wrong_target_ids else target
                logits[batch_index, position, prediction] = 20.0
            logits[batch_index, length - 1, 0] = 20.0
        return SimpleNamespace(logits=logits)


def resolved_paths(tmp_path: Path) -> ResolvedPaths:
    return ResolvedPaths(
        drive_root=None,
        dataset_root=None,
        train_file=tmp_path / "train.jsonl",
        validation_file=tmp_path / "validation.jsonl",
        test_file=tmp_path / "test.jsonl",
        output_dir=tmp_path,
    )


def test_causal_shift_and_ignore_index_are_exactly_aligned():
    logits = torch.full((1, 4, 8), -10.0)
    labels = torch.tensor([[IGNORE_INDEX, 2, 3, IGNORE_INDEX]])
    kinds = torch.tensor([[SPAN_OTHER, SPAN_TOOL_CALL, SPAN_FINAL_ANSWER, SPAN_TOOL_CALL]])
    logits[0, 0, 2] = 10.0  # logits[t=0] predicts labels[t=1]
    logits[0, 1, 3] = 10.0  # logits[t=1] predicts labels[t=2]
    logits[0, 2, 7] = 10.0  # ignored because labels[t=3] is IGNORE_INDEX

    metrics = score_causal_batch(logits, labels, kinds)[0]

    assert metrics["supervised"]["tokens"] == 2
    assert metrics["supervised"]["correct"] == 2
    assert metrics["tool_call"]["tokens"] == 1
    assert metrics["tool_call"]["sequence_exact"] is True
    assert metrics["final_answer"]["tokens"] == 1
    assert metrics["final_answer"]["sequence_exact"] is True


def test_streaming_span_metrics_multiple_actions_exact_match_and_breakdowns():
    tokenizer = CharacterTokenizer()
    rows = [
        diagnostic_row("one", task_type="factual", source_dataset="custom"),
        diagnostic_row(
            "two",
            task_type="compare",
            source_dataset="agent_flan",
            tool_marker="~",
            final_marker="^",
            two_calls=True,
        ),
    ]
    model = NextTokenModel(wrong_target_ids={ord("~"), ord("^")})

    report = evaluate_teacher_forced_test_diagnostics(
        model,
        tokenizer,
        rows,
        max_length=10_000,
        batch_size=2,
        identifiers={"model_id": "fixture"},
    )

    expected_tool_tokens = sum(
        sum(
            label != IGNORE_INDEX and kind == SPAN_TOOL_CALL
            for label, kind in zip(
                build_test_diagnostic_feature(tokenizer, row, max_length=10_000)["labels"][1:],
                build_test_diagnostic_feature(tokenizer, row, max_length=10_000)["span_kinds"][1:],
            )
        )
        for row in rows
    )
    assert report["mode"] == "teacher_forced"
    assert report["rows"] == 2
    assert report["tool_calls"]["rows"] == 2
    assert report["tool_calls"]["tokens"] == expected_tool_tokens
    assert report["tool_calls"]["sequence_exact_match_rate"] == 0.5
    assert report["final_answers"]["rows"] == 2
    assert report["final_answers"]["sequence_exact_match_rate"] == 0.5
    assert report["supervised_token_accuracy"] < 1.0
    assert report["by_task_type"]["factual"]["rows"] == 1
    assert report["by_task_type"]["compare"]["tool_call_token_accuracy"] < 1.0
    assert report["by_source_dataset"]["custom"]["rows"] == 1
    assert report["by_source_dataset"]["agent_flan"]["final_answer_token_accuracy"] < 1.0
    assert report["identifiers"]["model_id"] == "fixture"
    assert model.forward_calls == 1
    assert model.inference_mode_seen == [True]


def test_left_truncation_keeps_span_mapping_and_supervision():
    tokenizer = CharacterTokenizer()
    row = diagnostic_row(
        "truncated",
        task_type="summary",
        source_dataset="custom",
        system_prefix="S" * 600,
    )
    full = analyze_truncation(tokenizer, row, max_length=100_000)
    assert full["initial_user_span"] is not None
    cut = max(1, int(full["initial_user_span"][0]) // 2)
    max_length = int(full["total_tokens"]) - cut
    truncated = analyze_truncation(tokenizer, row, max_length=max_length)
    assert truncated["truncated"] is True
    assert truncated["initial_user_lost"] is False
    assert truncated["lost_assistant_targets"] == 0

    feature = build_test_diagnostic_feature(tokenizer, row, max_length=max_length)
    canonical = build_canonical_sft_example(tokenizer, row, max_length=max_length)
    assert {key: feature[key] for key in canonical} == canonical
    report = evaluate_teacher_forced_test_diagnostics(
        NextTokenModel(),
        tokenizer,
        [row],
        max_length=max_length,
        batch_size=1,
    )
    assert any(kind == SPAN_TOOL_CALL for kind in feature["span_kinds"])
    assert any(kind == SPAN_FINAL_ANSWER for kind in feature["span_kinds"])
    assert report["supervised_token_accuracy"] == 1.0
    assert report["tool_calls"]["sequence_exact_match_rate"] == 1.0
    assert report["final_answers"]["sequence_exact_match_rate"] == 1.0


def test_max_sample_limit_uses_selected_prefix_and_streams_batches():
    tokenizer = CharacterTokenizer()
    rows = [
        diagnostic_row("one", task_type="factual", source_dataset="a"),
        diagnostic_row("two", task_type="compare", source_dataset="b"),
        diagnostic_row("three", task_type="summary", source_dataset="c"),
    ]
    model = NextTokenModel()
    report = evaluate_teacher_forced_test_diagnostics(
        model,
        tokenizer,
        rows,
        max_length=10_000,
        batch_size=1,
        max_samples=2,
    )
    assert report["rows"] == 2
    assert set(report["by_task_type"]) == {"factual", "compare"}
    assert set(report["by_source_dataset"]) == {"a", "b"}
    assert model.forward_calls == 2


def test_test_diagnostics_cli_defaults_and_validation():
    defaults = parse_args([])
    assert defaults.test_diagnostics is False
    assert defaults.test_diagnostics_max_samples is None
    assert parse_args(["--no-test-diagnostics"]).test_diagnostics is False
    enabled = parse_args([
        "--evaluate-test-after-train",
        "--test-diagnostics",
        "--test-diagnostics-max-samples",
        "7",
    ])
    validate_args(enabled)
    assert enabled.test_diagnostics is True
    assert enabled.test_diagnostics_max_samples == 7
    with pytest.raises(ValueError, match="requires --evaluate-test-after-train"):
        validate_args(parse_args(["--test-diagnostics"]))
    with pytest.raises(ValueError, match="must be positive"):
        validate_args(parse_args([
            "--evaluate-test-after-train",
            "--test-diagnostics",
            "--test-diagnostics-max-samples",
            "0",
        ]))


def test_diagnostics_disabled_keeps_old_test_metrics_only(tmp_path: Path, monkeypatch):
    args = parse_args(["--evaluate-test-after-train"])
    validate_args(args)
    row = diagnostic_row("one", task_type="factual", source_dataset="custom")
    split = DatasetSplit("test", tmp_path / "test.jsonl", [row], [row], "sha", {}, {})

    class FakeTrainer:
        state = SimpleNamespace(best_model_checkpoint="checkpoint-10")
        model = object()

        def evaluate(self, dataset, metric_key_prefix):
            assert metric_key_prefix == "test"
            return {"test_loss": 1.0, "test_runtime": 2.0}

    monkeypatch.setattr(
        engine,
        "evaluate_teacher_forced_test_diagnostics",
        lambda *args, **kwargs: pytest.fail("disabled diagnostics were invoked"),
    )
    engine.evaluate_held_out_test(
        args,
        resolved_paths(tmp_path),
        split,
        ["tokenized"],
        FakeTrainer(),
        CharacterTokenizer(),
    )
    metrics = json.loads((tmp_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_loss"] == 1.0
    assert not (tmp_path / "test_diagnostics.json").exists()


def test_enabled_diagnostics_uses_best_model_without_changing_best_selection(tmp_path: Path):
    args = parse_args([
        "--evaluate-test-after-train",
        "--test-diagnostics",
        "--test-diagnostics-max-samples",
        "1",
        "--max-seq-length",
        "10000",
    ])
    validate_args(args)
    row = diagnostic_row("one", task_type="factual", source_dataset="custom")
    split = DatasetSplit("test", tmp_path / "test.jsonl", [row], [row], "held-out-sha", {}, {})
    best_checkpoint = str((tmp_path / "checkpoint-50").resolve())

    class FakeTrainer:
        def __init__(self):
            self.state = SimpleNamespace(best_model_checkpoint=best_checkpoint)
            self.model = NextTokenModel()

        def evaluate(self, dataset, metric_key_prefix):
            return {"test_loss": 0.5}

        @staticmethod
        def _prepare_inputs(inputs):
            return inputs

    trainer = FakeTrainer()
    engine.evaluate_held_out_test(
        args,
        resolved_paths(tmp_path),
        split,
        ["tokenized"],
        trainer,
        CharacterTokenizer(),
    )
    report = json.loads((tmp_path / "test_diagnostics.json").read_text(encoding="utf-8"))
    assert report["rows"] == 1
    assert report["identifiers"]["model_state_source"] == "best_validation_checkpoint"
    assert report["identifiers"]["best_model_checkpoint"] == best_checkpoint
    assert trainer.state.best_model_checkpoint == best_checkpoint


def test_training_lifecycle_cannot_invoke_test_diagnostics_before_train(
    tmp_path: Path,
    monkeypatch,
):
    events: list[str] = []
    args = parse_args([
        "--evaluate-test-after-train",
        "--test-diagnostics",
        "--max-seq-length",
        "10000",
    ])
    validate_args(args)
    row = diagnostic_row("one", task_type="factual", source_dataset="custom")
    splits = {
        name: DatasetSplit(name, tmp_path / f"{name}.jsonl", [row], [row], name, {}, {})
        for name in ("train", "validation", "test")
    }

    class FakeModel:
        config = SimpleNamespace(use_cache=False)

    class FakeTrainerCallback:
        pass

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.state = SimpleNamespace(global_step=1, best_model_checkpoint="checkpoint-1")

        def train(self, resume_from_checkpoint=None):
            events.append("train")
            return SimpleNamespace(metrics={"train_loss": 1.0})

        def evaluate(self, dataset, metric_key_prefix=None):
            events.append("validation")
            return {"eval_loss": 1.0}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            EarlyStoppingCallback=object,
            Trainer=FakeTrainer,
            TrainerCallback=FakeTrainerCallback,
        ),
    )
    monkeypatch.setattr(engine, "_tokenized_dataset", lambda tokenizer, rows, max_length: list(rows))
    monkeypatch.setattr(engine, "load_model", lambda *args: FakeModel())
    monkeypatch.setattr(engine, "trainable_parameter_summary", lambda model: {"trainable_parameters": 1})
    monkeypatch.setattr(engine, "create_training_arguments", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(engine, "final_state_preserving_trainer_class", lambda base: base)
    monkeypatch.setattr(engine, "finalize_adapter_artifacts", lambda *args, **kwargs: {
        "status": "complete",
        "final_global_step": 1,
        "final_adapter_source": "fixture",
        "best_global_step": 1,
        "best_adapter_source": "fixture",
    })
    monkeypatch.setattr(engine, "print_adapter_metadata", lambda metadata: None)

    def held_out(*args, **kwargs):
        assert events and events[0] == "train"
        events.append("test_diagnostics")

    monkeypatch.setattr(engine, "evaluate_held_out_test", held_out)

    class FakeTokenizer(CharacterTokenizer):
        @staticmethod
        def save_pretrained(path):
            return None

    engine.train(
        args,
        resolved_paths(tmp_path),
        splits,
        FakeTokenizer(),
        {"splits": {"train": {"tokens": {"mean": 10.0}}}},
        SimpleNamespace(),
        "sdpa",
        {},
        None,
    )
    assert events == ["train", "validation", "test_diagnostics"]
