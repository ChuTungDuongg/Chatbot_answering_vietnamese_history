from __future__ import annotations

from argparse import Namespace

from scripts.upload_modal_volume import (
    RemoteFile,
    build_sync_plan,
    collect_local_files,
    collect_uploads,
    execute_sync_plan,
)


def test_bundle_upload_targets_volume_root_without_extra_wrapper(tmp_path):
    bundle = tmp_path / "vn_history_deployment"
    (bundle / "history_answerer").mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    args = Namespace(
        local_dir=str(bundle),
        remote_dir="/",
        history_model=None,
        history_adapter=None,
        research_agent=None,
        evidence_agent=None,
        retrieval_dir=None,
        corpus=None,
        config_dir=None,
        manifest=None,
    )
    uploads = collect_uploads(args)
    assert {upload.remote for upload in uploads} == {"/history_answerer", "/manifest.json"}


def test_modal_sync_dry_run_classification(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "config").mkdir(parents=True)
    (bundle / "config" / "inference_config.json").write_text("new", encoding="utf-8")
    (bundle / "artifact_lock.json").write_text("lock", encoding="utf-8")
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

    assert [item.remote for item in plan.upload] == ["/artifact_lock.json"]
    assert [item.remote for item in plan.replace] == ["/config/inference_config.json"]
    assert [item.remote for item in plan.delete_stale] == ["/stale.txt"]
    assert by_remote["/artifact_lock.json"] in plan.upload


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
