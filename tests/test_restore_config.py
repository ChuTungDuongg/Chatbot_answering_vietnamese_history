"""Path contracts for the restored runtime; no large artifact/model dependency."""

from pathlib import Path

from app.config import REPO_ROOT, Settings


def test_default_v1_uses_strict_runtime_metadata(monkeypatch, tmp_path):
    for name in ("ARTIFACT_ROOT", "CORPUS_PATH", "RETRIEVAL_ROOT", "INFERENCE_CONFIG_PATH", "RUNTIME_MANIFEST_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None)
    assert settings.corpus_path == REPO_ROOT / "artifacts/corpus_v1/chunks.jsonl"
    assert settings.retrieval_root == REPO_ROOT / "artifacts/corpus_v1/retrieval"
    assert settings.manifest_path == REPO_ROOT / "artifacts/corpus_v1/runtime/manifest.json"
    assert settings.inference_config_path == REPO_ROOT / "artifacts/corpus_v1/runtime/inference_config.json"


def test_explicit_v0_retains_legacy_contract():
    settings = Settings(_env_file=None, artifact_root=Path("artifacts/vn_history_deployment"),
                        corpus_path_override=None, retrieval_root=None,
                        runtime_manifest_path=None, inference_config_path_override=None)
    assert settings.corpus_path == REPO_ROOT / "artifacts/vn_history_deployment/corpus/vn_history_rag_chunks_enriched.jsonl"
    assert settings.retrieval_root is None
    assert settings.manifest_path == REPO_ROOT / "artifacts/vn_history_deployment/manifest.json"


def test_modal_mount_paths_remain_absolute():
    settings = Settings(_env_file=None, artifact_root=Path("/artifacts/corpus_v1"),
                        corpus_path_override=Path("/artifacts/corpus_v1/chunks.jsonl"),
                        retrieval_root=Path("/artifacts/corpus_v1/retrieval"),
                        runtime_manifest_path=None, inference_config_path_override=None)
    # Windows Path regards a POSIX rooted path differently; the container uses POSIX Path.
    if Path("/artifacts").is_absolute():
        assert settings.corpus_path == Path("/artifacts/corpus_v1/chunks.jsonl")
        assert settings.manifest_path == Path("/artifacts/corpus_v1/runtime/manifest.json")
