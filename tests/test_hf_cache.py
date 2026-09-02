from __future__ import annotations

import json
from pathlib import Path

from app.agents.common.hf_cache import hf_cache_status, resolve_hf_hub_cache_dir, seed_hf_cache
from app.agents.common.model_registry import CENTRAL_BASE_MODEL_ID
from scripts import hf_cache as hf_cache_cli


def test_hf_cache_dir_prefers_explicit_then_hub_then_home(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    assert resolve_hf_hub_cache_dir(explicit) == explicit

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert resolve_hf_hub_cache_dir() == tmp_path / "hub"

    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert resolve_hf_hub_cache_dir() == tmp_path / "home" / "hub"


def test_hf_cache_status_reports_miss_without_download(monkeypatch, tmp_path):
    def fake_snapshot_download(**_kwargs):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    status = hf_cache_status(CENTRAL_BASE_MODEL_ID, cache_dir=tmp_path)

    assert status["cache_hit"] is False
    assert status["cache_root"] == str(tmp_path)
    assert "FileNotFoundError" in status["error"]


def test_hf_cache_status_reports_hit(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen3-8B" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_kwargs: str(snapshot))

    status = hf_cache_status(CENTRAL_BASE_MODEL_ID, cache_dir=tmp_path)

    assert status["cache_hit"] is True
    assert status["snapshot_path"] == str(snapshot)
    assert status["required_files"] == {"config.json": True, "tokenizer_config.json": True}


def test_seed_hf_cache_is_idempotent_over_same_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen3-8B" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    first = seed_hf_cache(CENTRAL_BASE_MODEL_ID, cache_dir=tmp_path)
    second = seed_hf_cache(CENTRAL_BASE_MODEL_ID, cache_dir=tmp_path)

    assert first["cache_hit"] is True
    assert second["cache_hit"] is True
    assert len(calls) == 4
    assert all(call["cache_dir"] == str(tmp_path) for call in calls)


def test_hf_cache_cli_validate_only_uses_requested_cache(monkeypatch, tmp_path, capsys):
    def fake_status(model_id, *, cache_dir=None):
        return {"model_id": model_id, "cache_root": str(cache_dir), "cache_hit": True}

    monkeypatch.setattr(hf_cache_cli, "hf_cache_status", fake_status)

    exit_code = hf_cache_cli.main([
        "--validate-only",
        "--model-id", CENTRAL_BASE_MODEL_ID,
        "--cache-dir", str(tmp_path),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["models"][0]["cache_root"] == str(tmp_path)


def test_hf_cache_cli_seed_can_include_shared_4b(monkeypatch, tmp_path, capsys):
    seeded: list[str] = []

    def fake_seed(model_id, *, cache_dir=None, local_files_only=False):
        seeded.append(model_id)
        return {"model_id": model_id, "cache_root": str(cache_dir), "cache_hit": True}

    monkeypatch.setattr(hf_cache_cli, "seed_hf_cache", fake_seed)

    exit_code = hf_cache_cli.main(["--cache-dir", str(tmp_path), "--include-shared-4b"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert seeded == ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B-Instruct-2507"]
