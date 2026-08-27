from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest

from training.evidence_agent.config import EvidenceAgentConfig
from training.evidence_agent.train import build_parser, initialize_peft_adapter, main


def test_cli_exposes_init_adapter_separately_from_resume():
    help_text = build_parser().format_help()
    assert "--init-adapter" in help_text
    assert "--resume-from-checkpoint" in help_text


def test_existing_adapter_is_loaded_trainable(monkeypatch):
    calls = []

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            calls.append((model, path, kwargs))
            return "trainable-adapter"

    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeftModel))
    result = initialize_peft_adapter("base", argparse.Namespace(init_adapter="adapter-root"), EvidenceAgentConfig())
    assert result == "trainable-adapter"
    assert calls == [("base", "adapter-root", {"is_trainable": True})]


def test_init_adapter_cannot_resume_old_optimizer_state():
    with pytest.raises(ValueError, match="cannot be combined"):
        main([
            "--dataset", "does-not-need-to-exist.jsonl",
            "--init-adapter", "adapter-root",
            "--resume-from-checkpoint", "checkpoint-root",
            "--dry-run",
        ])
