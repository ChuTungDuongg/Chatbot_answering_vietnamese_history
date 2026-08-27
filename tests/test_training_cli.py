from __future__ import annotations

import subprocess
import sys


def run_help(module: str):
    return subprocess.run([sys.executable, "-m", module, "--help"], check=True, capture_output=True, text=True)


def test_training_cli_help_smoke():
    modules = [
        "training.history_answerer.train",
        "training.history_answerer.evaluate",
        "training.history_answerer.train_instruction_sft",
        "training.history_answerer.prepare_dataset",
        "training.research_agent.prepare_dataset",
        "training.research_agent.build_history_trajectories",
        "training.research_agent.train",
        "training.research_agent.evaluate",
        "training.research_agent.validate_dataset",
        "training.research_agent.preflight",
        "training.evidence_agent.prepare_dataset",
        "training.evidence_agent.train",
        "training.evidence_agent.evaluate",
        "training.scripts.build_corpus",
        "scripts.upload_modal_volume",
    ]
    for module in modules:
        completed = run_help(module)
        assert "usage:" in completed.stdout


def test_all_trainers_expose_required_override_flags():
    required = {
        "--batch-size",
        "--eval-batch-size",
        "--gradient-accumulation-steps",
        "--learning-rate",
        "--lora-r",
        "--resume-from-checkpoint",
    }
    for module in (
        "training.history_answerer.train",
        "training.research_agent.train",
        "training.evidence_agent.train",
    ):
        help_text = run_help(module).stdout
        assert all(flag in help_text for flag in required)


def test_history_rag_sft_starts_from_vanilla_base():
    help_text = run_help("training.history_answerer.train").stdout
    assert "--phase1-adapter" not in help_text
    assert "--merged-base-dir" not in help_text



