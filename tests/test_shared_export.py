from __future__ import annotations

import json

from app.agents.model_registry import SHARED_BASE_MODEL_ID
from training.scripts.export_artifacts import main


def test_export_contains_three_adapters_and_no_base_weight_copy(tmp_path):
    adapters = {}
    for role in ("research", "evidence", "history"):
        path = tmp_path / f"source-{role}"
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": SHARED_BASE_MODEL_ID}), encoding="utf-8"
        )
        (path / "adapter_model.safetensors").write_bytes(b"fixture")
        adapters[role] = path
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"c1","text":"x"}\n', encoding="utf-8")
    retrieval = tmp_path / "retrieval"
    (retrieval / "faiss").mkdir(parents=True)
    (retrieval / "bm25s_index").mkdir()
    (retrieval / "faiss" / "chunks.index").write_bytes(b"idx")
    (retrieval / "faiss" / "manifest.json").write_text('{"count":1}', encoding="utf-8")
    (retrieval / "bm25s_index" / "index").write_bytes(b"idx")
    (retrieval / "bm25s_index" / "phase9_manifest.json").write_text('{"count":1}', encoding="utf-8")
    output = tmp_path / "deployment"
    (output / "stale.txt").parent.mkdir(parents=True)
    (output / "stale.txt").write_text("old", encoding="utf-8")
    (output / "adapters" / "research" / "checkpoint-1").mkdir(parents=True)

    assert main([
        "--research-agent", str(adapters["research"]),
        "--evidence-agent", str(adapters["evidence"]),
        "--history-agent", str(adapters["history"]),
        "--corpus", str(corpus),
        "--retrieval-dir", str(retrieval),
        "--output-root", str(output),
    ]) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shared_base_model_id"] == SHARED_BASE_MODEL_ID
    assert manifest["base_weights_bundled"] is False
    assert set(manifest["roles"]) == {"research", "evidence", "history"}
    assert all((output / "adapters" / role / "adapter_config.json").is_file() for role in manifest["roles"])
    assert (output / "artifact_lock.json").is_file()
    assert not (output / "stale.txt").exists()
    assert not (output / "adapters" / "research" / "checkpoint-1").exists()
