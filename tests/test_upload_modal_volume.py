from __future__ import annotations

from argparse import Namespace
import json

import pytest

from app.agents.model_registry import CENTRAL_BASE_MODEL_ID, SHARED_BASE_MODEL_ID, registry_manifest
from app.artifact_contract import (
    build_artifact_lock,
    inference_config_payload,
    manifest_payload,
    write_artifact_lock,
)
from scripts.upload_modal_volume import (
    LocalFile,
    RemoteFile,
    SyncPlan,
    build_sync_plan,
    collect_local_files,
    collect_uploads,
    execute_sync_plan,
    main as upload_main,
    validate_canonical_bundle,
)
from scripts.validate_artifact_bundle import main as validate_bundle_main


def _canonical_bundle(tmp_path):
    bundle = tmp_path / "vn_history_deployment"
    for role in ("research", "evidence", "history"):
        adapter = bundle / "adapters" / role
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": SHARED_BASE_MODEL_ID}), encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(f"{role}-weights".encode())
    central_adapter_path = "adapters/central-v2"
    central = bundle / central_adapter_path
    central.mkdir(parents=True)
    (central / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": CENTRAL_BASE_MODEL_ID}), encoding="utf-8",
    )
    (central / "adapter_model.safetensors").write_bytes(b"central-weights")
    (bundle / "corpus").mkdir()
    (bundle / "corpus" / "vn_history_rag_chunks_enriched.jsonl").write_text(
        '{"chunk_id":"c1"}\n', encoding="utf-8",
    )
    (bundle / "retrieval" / "faiss").mkdir(parents=True)
    (bundle / "retrieval" / "faiss" / "chunks.index").write_bytes(b"index")
    (bundle / "retrieval" / "faiss" / "manifest.json").write_text('{"count":1}', encoding="utf-8")
    (bundle / "retrieval" / "bm25s_index").mkdir()
    (bundle / "retrieval" / "bm25s_index" / "phase9_manifest.json").write_text(
        '{"count":1}', encoding="utf-8",
    )
    (bundle / "config").mkdir()
    (bundle / "config" / "inference_config.json").write_text(
        json.dumps(inference_config_payload(central_adapter_path=central_adapter_path)), encoding="utf-8",
    )
    (bundle / "config" / "model_registry.json").write_text(
        json.dumps(registry_manifest(central_adapter_path=central_adapter_path)), encoding="utf-8",
    )
    provisional = build_artifact_lock(bundle)
    (bundle / "manifest.json").write_text(
        json.dumps(
            manifest_payload(
                corpus_count=1,
                deployment_id=provisional["deployment_id"],
                central_adapter_path=central_adapter_path,
            )
        ),
        encoding="utf-8",
    )
    write_artifact_lock(bundle)
    (bundle / "EXPORT_SUCCESS.txt").write_text("ok\n", encoding="utf-8")
    return bundle


def _bundle_args(bundle, **overrides):
    values = {
        "local_dir": str(bundle), "remote_dir": "/", "history_model": None,
        "history_adapter": None, "research_agent": None, "evidence_agent": None,
        "central_agent": None, "retrieval_dir": None, "corpus": None,
        "config_dir": None, "manifest": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_bundle_upload_targets_volume_root_without_extra_wrapper(tmp_path):
    bundle = _canonical_bundle(tmp_path)
    uploads = collect_uploads(_bundle_args(bundle))

    remotes = [upload.remote for upload in uploads]
    assert "/artifact_lock.json" in remotes
    assert remotes[-2:] == ["/manifest.json", "/artifact_lock.json"]


def test_modal_sync_dry_run_classification(tmp_path):
    bundle = _canonical_bundle(tmp_path)
    local_files = collect_local_files(bundle, "/")
    by_remote = {item.remote: item for item in local_files}
    remote = {
        "/config/inference_config.json": RemoteFile(
            remote="/config/inference_config.json",
            size=3,
            sha256="different",
        ),
        "/stale.txt": RemoteFile(remote="/stale.txt", size=3, sha256="old"),
    }

    plan = build_sync_plan(local_files, remote, remote_dir="/")

    assert by_remote["/artifact_lock.json"] in plan.upload
    assert [item.remote for item in plan.replace] == ["/config/inference_config.json"]
    assert [item.remote for item in plan.delete_stale] == ["/stale.txt"]


def test_local_bundle_is_validated_before_upload_and_manifest_id_must_match(tmp_path):
    bundle = _canonical_bundle(tmp_path)
    root, lock = validate_canonical_bundle(bundle)
    assert root == bundle.resolve()
    assert lock["central"] is not None

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deployment_id"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest deployment_id"):
        validate_canonical_bundle(bundle)


def test_local_validation_cli_needs_no_modal_or_gpu(tmp_path, capsys):
    bundle = _canonical_bundle(tmp_path)

    assert validate_bundle_main([str(bundle)]) == 0
    output = capsys.readouterr().out
    assert "ARTIFACT_BUNDLE_VALID" in output
    assert '"central": true' in output


def test_local_validation_cli_returns_nonzero_with_compact_lock_diff(tmp_path, capsys):
    bundle = _canonical_bundle(tmp_path)
    (bundle / "adapters" / "central-v2" / "adapter_model.safetensors").write_bytes(b"stale")

    assert validate_bundle_main([str(bundle)]) == 1
    error = capsys.readouterr().err
    assert "ARTIFACT_BUNDLE_INVALID" in error
    assert "central.adapter_model_sha256" in error


def test_unsafe_component_upload_is_rejected_instead_of_leaving_stale_lock(tmp_path):
    with pytest.raises(ValueError, match="Unsafe component upload is disabled"):
        collect_uploads(_bundle_args(None, local_dir=None, central_agent=str(tmp_path / "central")))
    with pytest.raises(ValueError, match="Unsafe component upload is disabled"):
        upload_main([
            "--volume", "vn-history-artifacts",
            "--central-agent", str(tmp_path / "central"),
        ])


def test_invalid_local_bundle_fails_before_any_modal_subprocess(tmp_path, monkeypatch):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    called = False

    def forbidden_subprocess(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Modal subprocess must not run")

    monkeypatch.setattr("scripts.upload_modal_volume.subprocess.run", forbidden_subprocess)
    with pytest.raises(FileNotFoundError, match="Canonical deployment bundle is incomplete"):
        upload_main([
            "--volume", "vn-history-artifacts", "--local-dir", str(incomplete), "--dry-run",
        ])
    assert called is False


def test_upload_order_places_manifest_near_end_and_lock_last(tmp_path, capsys):
    regular = LocalFile(tmp_path / "config.json", "/config/inference_config.json", 1, "a")
    manifest = LocalFile(tmp_path / "manifest.json", "/manifest.json", 1, "b")
    lock = LocalFile(tmp_path / "artifact_lock.json", "/artifact_lock.json", 1, "c")
    plan = SyncPlan(unchanged=[], upload=[lock, manifest, regular], replace=[], delete_stale=[])

    assert execute_sync_plan(
        volume="vn-history-artifacts", plan=plan, dry_run=True,
        allow_replace_adapter_weights=False,
    ) == 0

    commands = [line for line in capsys.readouterr().out.splitlines() if line.startswith("modal volume put")]
    assert "/config/inference_config.json" in commands[0]
    assert "/manifest.json" in commands[-2]
    assert "/artifact_lock.json" in commands[-1]


def test_sync_never_deletes_old_lock_before_replacement():
    plan = SyncPlan(
        unchanged=[], upload=[], replace=[],
        delete_stale=[RemoteFile("/artifact_lock.json", 1, "old")],
    )
    with pytest.raises(RuntimeError, match="Refusing to delete"):
        execute_sync_plan(
            volume="vn-history-artifacts", plan=plan, dry_run=True,
            allow_replace_adapter_weights=False,
        )


def test_unchanged_remote_file_not_uploaded(tmp_path, capsys):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("same", encoding="utf-8")
    local_files = collect_local_files(bundle, "/")
    remote = {
        "/manifest.json": RemoteFile(
            remote="/manifest.json",
            size=local_files[0].size,
            sha256=local_files[0].sha256,
        )
    }
    plan = build_sync_plan(local_files, remote, remote_dir="/")

    rc = execute_sync_plan(
        volume="vn-history-artifacts",
        plan=plan,
        dry_run=True,
        allow_replace_adapter_weights=False,
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert plan.unchanged == local_files
    assert "modal volume put" not in output


def test_adapter_weight_replacement_requires_explicit_stop(tmp_path, capsys):
    bundle = tmp_path / "bundle"
    adapter = bundle / "adapters" / "research"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"new")
    local_files = collect_local_files(bundle, "/")
    remote = {
        "/adapters/research/adapter_model.safetensors": RemoteFile(
            remote="/adapters/research/adapter_model.safetensors",
            size=3,
            sha256="different",
        )
    }
    plan = build_sync_plan(local_files, remote, remote_dir="/")

    rc = execute_sync_plan(
        volume="vn-history-artifacts",
        plan=plan,
        dry_run=True,
        allow_replace_adapter_weights=False,
    )

    assert rc == 2
    assert "STOP: adapter_model.safetensors replacement detected" in capsys.readouterr().out
