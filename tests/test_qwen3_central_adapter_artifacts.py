from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.central_agent.config import parse_args, validate_args
from training.central_agent.engine import (
    adapter_artifacts_exist,
    create_training_arguments,
    final_state_preserving_trainer_class,
    finalize_adapter_artifacts,
    print_adapter_metadata,
)
from training.central_agent.runtime import find_latest_checkpoint, resolve_resume_checkpoint


def write_adapter(path: Path, payload: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(payload)


def write_checkpoint(path: Path, step: int, payload: bytes) -> Path:
    checkpoint = path / f"checkpoint-{step}"
    write_adapter(checkpoint, payload)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
    return checkpoint


def adapter_payload(path: Path) -> bytes:
    return (path / "adapter_model.safetensors").read_bytes()


class FakeTrainer:
    """Small stateful stand-in for the pinned Trainer lifecycle."""

    def __init__(
        self,
        *,
        global_step: int,
        best_model_checkpoint: Path | None,
        model_payload: bytes,
        save_strategy: str,
        save_total_limit: int,
    ) -> None:
        self.state = SimpleNamespace(
            global_step=global_step,
            best_model_checkpoint=(
                str(best_model_checkpoint.resolve()) if best_model_checkpoint is not None else None
            ),
        )
        self.args = SimpleNamespace(
            should_save=True,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
        )
        self.model_payload = model_payload
        self.best_model_reload_count = 0

    def save_model(self, output_dir: str, _internal_call: bool = False) -> None:
        assert _internal_call is True
        write_adapter(Path(output_dir), self.model_payload)

    def _load_best_model(self) -> None:
        self.best_model_reload_count += 1
        checkpoint = Path(self.state.best_model_checkpoint)
        self.model_payload = adapter_payload(checkpoint)


def make_preserving_trainer(
    output_dir: Path,
    *,
    global_step: int,
    best_model_checkpoint: Path | None,
    model_payload: bytes,
    save_strategy: str = "steps",
    save_total_limit: int = 1,
):
    trainer_type = final_state_preserving_trainer_class(FakeTrainer)
    return trainer_type(
        global_step=global_step,
        best_model_checkpoint=best_model_checkpoint,
        model_payload=model_payload,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        final_adapter_output=output_dir / "final_adapter",
    )


@pytest.mark.parametrize("save_strategy", ["steps", "epoch"])
def test_true_final_non_save_boundary_survives_best_reload_and_rotation(
    tmp_path: Path,
    save_strategy: str,
):
    output = tmp_path / save_strategy
    checkpoint_50 = write_checkpoint(output, 50, b"best-step-50")
    checkpoint_100 = write_checkpoint(output, 100, b"periodic-step-100")
    trainer = make_preserving_trainer(
        output,
        global_step=123,
        best_model_checkpoint=checkpoint_50,
        model_payload=b"true-final-step-123",
        save_strategy=save_strategy,
        save_total_limit=1,
    )

    # This is the exact lifecycle call made by Transformers after the last
    # optimizer step and before train() returns with the best model loaded.
    trainer._load_best_model()

    assert adapter_payload(output / "final_adapter") == b"true-final-step-123"
    assert adapter_payload(output / "final_adapter") != adapter_payload(checkpoint_100)
    assert trainer.model_payload == b"best-step-50"
    assert trainer.best_model_reload_count == 1

    # Simulate save_total_limit rotation after final capture. The independent
    # final artifact and the best checkpoint both remain available.
    shutil.rmtree(checkpoint_100)
    metadata = finalize_adapter_artifacts(
        trainer,
        output,
        load_best_model_at_end=True,
    )

    assert adapter_payload(output / "final_adapter") == b"true-final-step-123"
    assert adapter_payload(output / "best_adapter") == b"best-step-50"
    assert metadata["final_global_step"] == 123
    assert metadata["best_global_step"] == 50
    assert metadata["final_adapter_source"] == "in_memory_before_best_model_reload"
    assert Path(metadata["best_adapter_source"]) == checkpoint_50.resolve()


def test_exact_save_boundary_and_best_equal_final_are_both_preserved(tmp_path: Path):
    output = tmp_path / "run"
    checkpoint_100 = write_checkpoint(output, 100, b"step-100")
    trainer = make_preserving_trainer(
        output,
        global_step=100,
        best_model_checkpoint=checkpoint_100,
        model_payload=b"step-100",
    )

    trainer._load_best_model()
    metadata = finalize_adapter_artifacts(
        trainer,
        output,
        load_best_model_at_end=True,
    )

    assert adapter_payload(output / "final_adapter") == b"step-100"
    assert adapter_payload(output / "best_adapter") == b"step-100"
    assert metadata["final_global_step"] == metadata["best_global_step"] == 100


def test_without_best_reload_finalizer_saves_current_last_state(tmp_path: Path):
    output = tmp_path / "run"
    trainer = make_preserving_trainer(
        output,
        global_step=17,
        best_model_checkpoint=None,
        model_payload=b"last-step-17",
    )

    metadata = finalize_adapter_artifacts(
        trainer,
        output,
        load_best_model_at_end=False,
    )

    assert adapter_payload(output / "final_adapter") == b"last-step-17"
    assert adapter_payload(output / "best_adapter") == b"last-step-17"
    assert metadata["final_global_step"] == 17
    assert metadata["best_global_step"] is None
    assert metadata["final_adapter_source"] == "in_memory_after_train_without_best_model_reload"
    assert metadata["best_adapter_source"] == "final_adapter_fallback_no_best_checkpoint"


def test_missing_pre_reload_capture_never_silently_labels_best_as_final(tmp_path: Path):
    output = tmp_path / "run"
    best = write_checkpoint(output, 50, b"best-step-50")
    trainer = FakeTrainer(
        global_step=123,
        best_model_checkpoint=best,
        model_payload=b"already-reloaded-best-step-50",
        save_strategy="steps",
        save_total_limit=1,
    )

    with pytest.raises(RuntimeError, match="not captured before best-model reload"):
        finalize_adapter_artifacts(trainer, output, load_best_model_at_end=True)
    assert not (output / "final_adapter").exists()


def test_interrupted_run_resumes_periodic_checkpoint_and_recaptures_final(tmp_path: Path):
    output = tmp_path / "run"
    checkpoint_50 = write_checkpoint(output, 50, b"safe-step-50")
    write_adapter(output / "final_adapter", b"stale-from-earlier-completion")

    # Auto-resume deliberately considers only complete Trainer checkpoints,
    # never an adapter-only export.
    resume = resolve_resume_checkpoint(parse_args(["--auto-resume"]), output)
    assert resume == checkpoint_50.resolve()
    assert find_latest_checkpoint(output) == checkpoint_50.resolve()

    resumed = make_preserving_trainer(
        output,
        global_step=73,
        best_model_checkpoint=checkpoint_50,
        model_payload=b"resumed-true-final-step-73",
    )
    resumed._load_best_model()
    finalize_adapter_artifacts(resumed, output, load_best_model_at_end=True)
    assert adapter_payload(output / "final_adapter") == b"resumed-true-final-step-73"
    assert adapter_payload(output / "best_adapter") == b"safe-step-50"


def test_adapter_metadata_output_is_explicit_and_unbuffered(capsys):
    print_adapter_metadata({
        "final_global_step": 123,
        "final_adapter_source": "in_memory_before_best_model_reload",
        "best_global_step": None,
        "best_adapter_source": "final_adapter_fallback_no_best_checkpoint",
    })
    assert capsys.readouterr().out.splitlines() == [
        "FINAL_GLOBAL_STEP=123",
        "FINAL_ADAPTER_SOURCE=in_memory_before_best_model_reload",
        "BEST_GLOBAL_STEP=NONE",
        "BEST_ADAPTER_SOURCE=final_adapter_fallback_no_best_checkpoint",
    ]


def test_adapter_artifact_validation_checks_config_and_weights(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert not adapter_artifacts_exist(adapter)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert not adapter_artifacts_exist(adapter)
    (adapter / "adapter_model.bin").write_bytes(b"weights")
    assert adapter_artifacts_exist(adapter)


@pytest.mark.parametrize("strategy", ["steps", "epoch"])
def test_training_arguments_preserve_best_loading_and_save_policy(
    tmp_path: Path,
    monkeypatch,
    strategy: str,
):
    args = parse_args([
        "--eval-strategy", strategy,
        "--save-strategy", strategy,
        "--save-total-limit", "2",
    ])
    validate_args(args)

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(TrainingArguments=FakeTrainingArguments),
    )
    training_args = create_training_arguments(
        args,
        output_dir=tmp_path,
        precision=SimpleNamespace(bf16=False, fp16=False),
    )
    assert training_args.load_best_model_at_end is True
    assert training_args.eval_strategy == strategy
    assert training_args.save_strategy == strategy
    assert training_args.save_total_limit == 2
