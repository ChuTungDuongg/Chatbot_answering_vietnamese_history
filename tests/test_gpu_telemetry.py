from __future__ import annotations

import logging

from app.services.rag_service import RAGService


def test_gpu_telemetry_handles_no_cuda(monkeypatch, caplog):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with caplog.at_level(logging.INFO):
        RAGService()._log_gpu_profile()

    records = [record for record in caplog.records if record.message == "GPU_PROFILE"]
    assert records
    assert getattr(records[-1], "gpu_name") is None
    assert getattr(records[-1], "bnb_4bit") is True
