from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.common.qlora import PrecisionSettings
from training.central.train.engine import _tokenized_dataset
from training.train_qwen3_8b_agent import (
    MANIFEST_SCHEMA_VERSION,
    _safe_cli_arguments,
    audit_tokenized_split,
    build_lora_settings,
    build_qlora_settings,
    build_run_manifest,
    checkpoint_is_valid,
    create_training_arguments,
    effective_train_batch_size,
    find_latest_checkpoint,
    load_datasets,
    main,
    parse_args,
    parse_lora_targets,
    resolve_paths,
    resolve_resume_checkpoint,
    sha256_file,
    validate_args,
    validate_resume_compatibility,
)
from training.trajectory_dataset.io_utils import atomic_write_json, atomic_write_jsonl
from training.trajectory_dataset.preprocess import IGNORE_INDEX, build_canonical_sft_example
from training.trajectory_dataset.schema import SEARCH_HISTORY_TOOL, make_trajectory, tool_call


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "<eos>"

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


def canonical_row(row_id: str, group: str) -> dict:
    from training.trajectory_dataset.schema import QWEN3_TOOL_TEMPLATE_CONTRACT
    call = tool_call(f"call-{row_id}", "search_history", {"query": "Nhà Mạc", "top_k": 2})
    return make_trajectory(
        trajectory_id=row_id,
        source_dataset="hermes_function_calling",
        task_type="factual",
        tools=[SEARCH_HISTORY_TOOL],
        messages=[
            {"role": "system", "content": "Trả lời bằng tiếng Việt."},
            {"role": "user", "content": "Nhà Mạc là gì?"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {
                "role": "tool", "name": "search_history", "tool_call_id": f"call-{row_id}",
                "content": json.dumps([{"chunk_id": f"e-{row_id}", "text": "Nhà Mạc là một triều đại."}]),
            },
            {"role": "assistant", "content": "Nhà Mạc là một triều đại."},
        ],
        provenance={"requires_final_answer": True, "source_group": group,
                    "chat_template_contract": QWEN3_TOOL_TEMPLATE_CONTRACT},
    )


def dataset_root(tmp_path: Path, *, include_test: bool = True) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    atomic_write_jsonl(root / "train.jsonl", [canonical_row("train-1", "train-group")])
    atomic_write_jsonl(root / "validation.jsonl", [canonical_row("validation-1", "validation-group")])
    if include_test:
        atomic_write_jsonl(root / "test.jsonl", [canonical_row("test-1", "test-group")])
    return root


def make_checkpoint(root: Path, step: int, *, valid: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    if valid:
        for name in ("trainer_state.json", "adapter_config.json"):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors"):
            (checkpoint / name).write_bytes(b"fixture")
    return checkpoint


def test_cli_parsing_defaults_and_optimization_overrides():
    args = parse_args([
        "--learning-rate", "2e-4", "--per-device-train-batch-size", "2",
        "--per-device-eval-batch-size", "3", "--gradient-accumulation-steps", "8",
        "--num-train-epochs", "4", "--max-steps", "120", "--max-seq-length", "2048",
        "--optim", "adamw_torch", "--lr-scheduler-type", "linear", "--warmup-steps", "10",
    ])
    validate_args(args)
    assert args.learning_rate == 2e-4
    assert (args.per_device_train_batch_size, args.per_device_eval_batch_size) == (2, 3)
    assert (args.gradient_accumulation_steps, args.num_train_epochs, args.max_steps) == (8, 4, 120)
    assert (args.optim, args.lr_scheduler_type, args.warmup_steps) == ("adamw_torch", "linear", 10)


@pytest.mark.parametrize(
    "flags",
    [
        ["--per-device-train-batch-size", "0"],
        ["--per-device-eval-batch-size", "0"],
        ["--gradient-accumulation-steps", "0"],
        ["--bf16", "--fp16"],
        ["--dataloader-persistent-workers", "--dataloader-num-workers", "0"],
    ],
)
def test_invalid_cli_combinations_fail(flags: list[str]):
    with pytest.raises(ValueError):
        validate_args(parse_args(flags))


def test_json_config_precedence_defaults_then_config_then_cli(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "learning_rate": 3e-4,
        "per_device_train_batch_size": 2,
        "gradient_checkpointing": False,
    }), encoding="utf-8")
    from_config = parse_args(["--config", str(config)])
    overridden = parse_args(["--config", str(config), "--learning-rate", "4e-4", "--gradient-checkpointing"])
    assert from_config.learning_rate == 3e-4
    assert from_config.per_device_train_batch_size == 2
    assert from_config.gradient_checkpointing is False
    assert overridden.learning_rate == 4e-4
    assert overridden.gradient_checkpointing is True
    assert overridden.lora_r == 32


def test_drive_and_dataset_root_paths_are_resolved_and_test_is_optional(tmp_path: Path):
    drive = tmp_path / "drive"
    drive.mkdir()
    data = dataset_root(drive, include_test=False)
    args = parse_args([
        "--drive-root", str(drive), "--dataset-root", "dataset", "--run-name", "run-a",
    ])
    paths = resolve_paths(args)
    assert paths.dataset_root == data.resolve()
    assert paths.train_file == (data / "train.jsonl").resolve()
    assert paths.validation_file == (data / "validation.jsonl").resolve()
    assert paths.test_file is None
    assert paths.output_dir == (drive / "training_runs" / "run-a").resolve()


def test_drive_root_must_exist_and_be_directory(tmp_path: Path):
    args = parse_args(["--drive-root", str(tmp_path / "missing")])
    with pytest.raises(ValueError, match="does not exist"):
        resolve_paths(args)
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory does not exist"):
        resolve_paths(parse_args(["--drive-root", str(file_root)]))


def test_effective_batch_size_uses_world_size_without_mutating_values():
    args = parse_args([
        "--per-device-train-batch-size", "2", "--gradient-accumulation-steps", "8",
    ])
    assert effective_train_batch_size(args, distributed_world_size=3) == 48
    assert (args.per_device_train_batch_size, args.gradient_accumulation_steps) == (2, 8)


def test_lora_and_qlora_config_values_are_fully_cli_controlled():
    args = parse_args([
        "--no-load-in-4bit", "--bnb-4bit-quant-type", "fp4", "--no-bnb-use-double-quant",
        "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.1",
        "--lora-bias", "lora_only", "--lora-target-modules", "q_proj,v_proj,q_proj",
    ])
    precision = PrecisionSettings(compute_dtype="float16", bf16=False, fp16=True)
    assert parse_lora_targets(args.lora_target_modules) == ("q_proj", "v_proj")
    assert build_lora_settings(args).bias == "lora_only"
    qlora = build_qlora_settings(args, precision)
    assert (qlora.load_in_4bit, qlora.bnb_4bit_quant_type, qlora.bnb_4bit_use_double_quant) == (
        False, "fp4", False,
    )


def test_dataset_validation_empty_failure_hash_stability_and_group_leakage(tmp_path: Path):
    root = dataset_root(tmp_path)
    args = parse_args(["--dataset-root", str(root), "--output-dir", str(tmp_path / "run")])
    splits = load_datasets(resolve_paths(args), args)
    assert sha256_file(root / "train.jsonl") == splits["train"].sha256
    assert set(splits) == {"train", "validation", "test"}

    atomic_write_jsonl(root / "train.jsonl", [])
    with pytest.raises(ValueError, match="empty"):
        load_datasets(resolve_paths(args), args)

    atomic_write_jsonl(root / "train.jsonl", [{"not": "canonical"}])
    with pytest.raises(ValueError, match="validation failed"):
        load_datasets(resolve_paths(args), args)

    atomic_write_jsonl(root / "train.jsonl", [canonical_row("train-2", "shared")])
    atomic_write_jsonl(root / "validation.jsonl", [canonical_row("validation-2", "shared")])
    with pytest.raises(ValueError, match="leakage"):
        load_datasets(resolve_paths(args), args)


def test_sample_limits_are_deterministic_and_do_not_mutate_source(tmp_path: Path):
    root = dataset_root(tmp_path, include_test=False)
    rows = [canonical_row(f"train-{index}", f"group-{index}") for index in range(10)]
    atomic_write_jsonl(root / "train.jsonl", rows)
    before = (root / "train.jsonl").read_bytes()
    args = parse_args([
        "--dataset-root", str(root), "--output-dir", str(tmp_path / "run"),
        "--max-train-samples", "4", "--data-seed", "99",
    ])
    first = load_datasets(resolve_paths(args), args)["train"].selected_rows
    second = load_datasets(resolve_paths(args), args)["train"].selected_rows
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4
    assert (root / "train.jsonl").read_bytes() == before


def test_auto_resume_selects_highest_valid_checkpoint_and_explicit_wins(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    checkpoint_10 = make_checkpoint(output, 10)
    checkpoint_100 = make_checkpoint(output, 100)
    make_checkpoint(output, 200, valid=False)
    (output / "checkpoint-bad").mkdir()
    assert checkpoint_is_valid(checkpoint_10)
    assert find_latest_checkpoint(output) == checkpoint_100.resolve()
    assert resolve_resume_checkpoint(parse_args(["--auto-resume"]), output) == checkpoint_100.resolve()
    explicit_args = parse_args([
        "--auto-resume", "--resume-from-checkpoint", str(checkpoint_10),
    ])
    assert resolve_resume_checkpoint(explicit_args, output) == checkpoint_10.resolve()


def test_explicit_resume_rejects_checkpoint_from_another_run(tmp_path: Path):
    output = tmp_path / "run-a"
    output.mkdir()
    foreign = make_checkpoint(tmp_path / "run-b", 10)
    with pytest.raises(ValueError, match="outside this run"):
        resolve_resume_checkpoint(parse_args(["--resume-from-checkpoint", str(foreign)]), output)


def test_resume_dataset_mismatch_is_rejected_unless_explicitly_allowed():
    previous = {
        "run_name": "run", "model_id": "Qwen/Qwen3-8B", "max_seq_length": 4096,
        "qlora": {"load_in_4bit": True}, "lora": {"r": 32},
        "datasets": {"train": {"sha256": "old"}, "validation": {"sha256": "same"}},
        "cli_arguments": {"learning_rate": 1e-4},
    }
    current = copy.deepcopy(previous)
    current["datasets"]["train"]["sha256"] = "new"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_resume_compatibility(previous, current, allow_data_mismatch=False)
    validate_resume_compatibility(previous, current, allow_data_mismatch=True)


def test_early_stopping_validation_and_training_arguments_mapping(tmp_path: Path, monkeypatch):
    invalid = parse_args(["--early-stopping-patience", "2", "--no-load-best-model-at-end"])
    with pytest.raises(ValueError, match="early stopping"):
        validate_args(invalid)
    args = parse_args([
        "--learning-rate", "2e-4", "--per-device-train-batch-size", "2",
        "--per-device-eval-batch-size", "3", "--gradient-accumulation-steps", "4",
        "--optim", "adamw_torch", "--lr-scheduler-type", "polynomial",
        "--warmup-steps", "7", "--weight-decay", "0.02", "--max-grad-norm", "0.8",
        "--save-steps", "20", "--eval-steps", "10", "--logging-steps", "2",
    ])
    validate_args(args)

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainingArguments=FakeTrainingArguments))
    training_args = create_training_arguments(
        args, output_dir=tmp_path,
        precision=PrecisionSettings(compute_dtype="float32", bf16=False, fp16=False),
    )
    assert training_args.learning_rate == 2e-4
    assert training_args.per_device_train_batch_size == 2
    assert training_args.per_device_eval_batch_size == 3
    assert training_args.gradient_accumulation_steps == 4
    assert training_args.optim == "adamw_torch"
    assert training_args.lr_scheduler_type == "polynomial"
    assert training_args.warmup_steps == 7 and training_args.warmup_ratio == 0
    assert training_args.weight_decay == 0.02 and training_args.max_grad_norm == 0.8
    assert (training_args.save_steps, training_args.eval_steps, training_args.logging_steps) == (20, 10, 2)


def test_manifest_serializes_hashes_hardware_batch_and_redacts_secrets(tmp_path: Path):
    root = dataset_root(tmp_path)
    args = parse_args([
        "--dataset-root", str(root), "--output-dir", str(tmp_path / "run"), "--run-name", "manifest-run",
    ])
    setattr(args, "api_token", "must-not-leak")
    paths = resolve_paths(args)
    splits = load_datasets(paths, args)
    manifest = build_run_manifest(
        args, paths, splits,
        precision=PrecisionSettings(compute_dtype="float16", bf16=False, fp16=True),
        attention_implementation="sdpa",
        hardware={"cuda_available": True, "gpu_name": "fixture", "gpu_total_vram_gb": 24},
        resume_source=None,
    )
    encoded = json.dumps(manifest)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["effective_train_batch_size"] == 16
    assert manifest["datasets"]["train"]["sha256"] == sha256_file(root / "train.jsonl")
    assert manifest["adapter_artifacts"] == {
        "status": "training_not_started",
        "final_global_step": None,
        "final_adapter_source": None,
        "best_global_step": None,
        "best_adapter_source": None,
    }
    assert "must-not-leak" not in encoded and manifest["cli_arguments"]["api_token"] == "<redacted>"


def test_canonical_loss_masks_non_assistant_and_supervises_tool_call_and_final():
    row = canonical_row("loss", "loss-group")
    tokenizer = CharacterTokenizer()
    feature = build_canonical_sft_example(tokenizer, row, max_length=100_000)
    rendered = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False, tools=row["tools"],
    )
    labels = feature["labels"]
    for role in ("system", "user", "tool"):
        start = rendered.index(f"<{role}>")
        end = rendered.index(f"</{role}>", start) + len(f"</{role}>")
        assert all(label == IGNORE_INDEX for label in labels[start:end])
    call_start = rendered.index("tool_calls")
    answer_start = rendered.rindex("Nhà Mạc là một triều đại")
    assert any(label != IGNORE_INDEX for label in labels[call_start:call_start + 100])
    assert all(label != IGNORE_INDEX for label in labels[answer_start:answer_start + 20])
    audit = audit_tokenized_split(tokenizer, [row], max_seq_length=100_000)
    assert audit["supervision_invariants_ok"]
    assert audit["rows_zero_supervised_tokens"] == 0


def test_preflight_rows_with_heterogeneous_metadata_reach_trainer_arrow_safely(monkeypatch):
    tokenizer = CharacterTokenizer()
    rows = [
        canonical_row("heterogeneous-a", "group-a"),
        canonical_row("heterogeneous-b", "group-b"),
    ]
    rows[0]["provenance"]["some_metadata"] = 168
    rows[1]["provenance"]["some_metadata"] = "168"
    rows[0]["provenance"]["mixed_flag"] = True
    rows[1]["provenance"]["mixed_flag"] = "true"
    rows[0]["provenance"]["mixed_shape"] = [1, 2]
    rows[1]["provenance"]["mixed_shape"] = "1,2"
    original_rows = copy.deepcopy(rows)

    class StrictFeatureDataset:
        expected_columns = ["input_ids", "attention_mask", "labels"]

        def __init__(self, features):
            self.features = copy.deepcopy(features)
            self.column_names = list(self.expected_columns)

        @classmethod
        def from_list(cls, values):
            assert all(list(value) == cls.expected_columns for value in values), (
                "raw canonical trajectory fields reached the Arrow boundary"
            )
            assert all(
                isinstance(value[column], list)
                for value in values
                for column in cls.expected_columns
            )
            return cls(values)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            return self.features[index]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=StrictFeatureDataset))

    preflight = audit_tokenized_split(tokenizer, rows, max_seq_length=100_000)
    expected = [
        build_canonical_sft_example(tokenizer, row, max_length=100_000)
        for row in rows
    ]
    dataset = _tokenized_dataset(tokenizer, rows, max_seq_length=100_000)

    assert preflight["preprocessing_errors"] == 0
    assert preflight["supervision_invariants_ok"] is True
    assert dataset.column_names == ["input_ids", "attention_mask", "labels"]
    assert len(dataset) == len(rows) == 2
    assert [dataset[index] for index in range(len(dataset))] == expected
    assert rows == original_rows


def test_dry_run_loads_neither_tokenizer_nor_model(tmp_path: Path, monkeypatch):
    root = dataset_root(tmp_path, include_test=False)
    monkeypatch.setattr(
        "training.central.train.cli.load_tokenizer",
        lambda *_: pytest.fail("dry-run loaded tokenizer"),
    )
    monkeypatch.setattr(
        "training.central.train.cli.train",
        lambda *_: pytest.fail("dry-run entered training engine"),
    )
    assert main([
        "--dataset-root", str(root), "--output-dir", str(tmp_path / "run"), "--dry-run",
    ]) == 0


def test_preflight_loads_tokenizer_but_not_model(tmp_path: Path, monkeypatch):
    root = dataset_root(tmp_path, include_test=False)
    loaded = {"tokenizer": 0}

    def fake_tokenizer(_):
        loaded["tokenizer"] += 1
        return CharacterTokenizer()

    monkeypatch.setattr("training.central.train.cli.load_tokenizer", fake_tokenizer)
    monkeypatch.setattr(
        "training.central.train.cli.train",
        lambda *_: pytest.fail("preflight entered training engine"),
    )
    assert main([
        "--dataset-root", str(root), "--output-dir", str(tmp_path / "run"),
        "--max-seq-length", "100000", "--preflight-only",
    ]) == 0
    assert loaded["tokenizer"] == 1
