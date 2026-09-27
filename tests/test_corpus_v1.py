"""Small offline fixture exercises the real staged builder."""

import json
from pathlib import Path

import pytest

from scripts.corpus_v1.audit import audit
from scripts.corpus_v1.chunking import chunk_text
from scripts.corpus_v1.history_filter import classify
from scripts.corpus_v1.normalize import normalize_text
from scripts.corpus_v1.pipeline import build
from scripts.corpus_v1.provenance import digest_file
from scripts.colab.bootstrap import bootstrap
from scripts.retrieval.build_index import ordered_chunk_id_sha256
from app.services.rag_service import RAGService


TEXT = (
    "Lịch sử triều đại được lưu giữ tại thành cổ. "
    "Một vị vua đã lãnh đạo đất nước và để lại di sản văn hóa. "
    "Các tài liệu khảo cổ cho thấy thành phố có nhiều giai đoạn phát triển. "
    "Người dân ghi chép sự kiện trong nhiều biên niên sử."
)


def _data():
    return [
        {"id": "ruler", "title": "Vua Đại Việt", "text": TEXT, "url": "https://vi.wikipedia.org/wiki/Vua"},
        {"id": "world", "title": "Hoàng đế La Mã", "text": "Hoàng đế cai trị đế quốc trong thời cổ đại. " + TEXT},
        {"id": "place", "title": "Thành phố cổ", "text": "Thủ đô này có nhiều công trình và di tích lịch sử. " + TEXT},
        {"id": "war", "title": "Trận đánh lịch sử", "text": "Chiến tranh và trận đánh diễn ra ở nhiều vùng. " + TEXT},
        {"id": "heritage", "title": "Di sản khảo cổ", "text": "Di sản văn hóa và khu khảo cổ. " + TEXT},
        {"id": "person", "title": "Nhà văn cổ", "text": "Một nhà văn sinh tại thành phố. " + TEXT},
        {"id": "review", "title": "Một địa danh", "text": "Đây là địa danh có nhiều thế hệ cư dân và những câu chuyện được ghi chép qua thời gian. " * 2},
        {"id": "drop", "title": "Công thức nấu ăn", "text": "Công thức nấu ăn hôm nay dùng nguyên liệu tươi. " * 3},
        {"id": "duplicate", "title": "Bản sao", "text": TEXT},
    ]


def _config():
    return {"dataset_id": "synthetic/wiki", "dataset_revision": "fixed-fixture", "split": "train",
            "streaming": True, "chunk_tokens": 50, "chunk_overlap": 8,
            "fields": {"title": "title", "text": "text", "id": "id", "url": "url"}}


def _counter(value: str) -> int:
    return len(value.split())


def test_normalization_preserves_vietnamese():
    value = "  Việt Nam\r\n\r\n\r\nTriều đại  "
    normalized = normalize_text(value)
    assert normalized == "Việt Nam\n\nTriều đại"
    assert normalize_text(normalized) == normalized


@pytest.mark.parametrize("index", range(6))
def test_broad_historical_categories_are_not_dropped(index):
    row = _data()[index]
    assert classify(row["title"], row["text"])[1] in ("KEEP", "REVIEW")


def test_review_and_clear_drop():
    rows = _data()
    assert classify(rows[6]["title"], rows[6]["text"])[1] == "REVIEW"
    assert classify(rows[7]["title"], rows[7]["text"])[1] == "DROP"


def test_chunk_overlap_never_exceeds_budget():
    text = "Một đoạn văn ngắn có ý nghĩa. " + "Lịch sử " * 15 + ". " + "Di sản " * 10
    chunks = list(chunk_text(text, _counter, budget=16, overlap=6))
    assert len(chunks) > 1
    assert all(count <= 16 for _, _, count in chunks)


def test_pipeline_resume_hashes_and_read_only_audit(tmp_path: Path):
    root = tmp_path / "corpus_v1" / "run-001"
    first = build(_config(), root, source_factory=_data, token_counter=_counter)
    assert first["filter_counts"]["DROP"] == 1
    assert first["filter_counts"]["REVIEW"] >= 1
    assert first["duplicate_document_count"] == 1
    assert first["document_count"] == 7
    docs = [json.loads(line) for line in (root / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = [json.loads(line) for line in (root / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({row["document_id"] for row in docs}) == len(docs)
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    assert all(row["schema_version"] == 1 for row in docs + chunks)
    assert all(row["source_id"] for row in docs + chunks)
    assert all(row["token_count"] <= 50 for row in chunks)
    assert any(row["historical_filter_decision"] == "REVIEW" for row in docs)
    hashes = json.loads((root / "hashes.json").read_text(encoding="utf-8"))
    assert all(digest_file(root / name) == digest for name, digest in hashes.items())
    before = digest_file(root / "chunks.jsonl")
    report = audit(root)
    assert report["counts"]["chunk_count"] == first["chunk_count"]
    assert not report["hash_mismatches"]
    assert digest_file(root / "chunks.jsonl") == before
    second = build(_config(), root, resume=True, source_factory=lambda: pytest.fail("reloaded source"),
                   token_counter=lambda _: pytest.fail("reloaded tokenizer"))
    assert second["hashes"] == first["hashes"]
    assert digest_file(root / "chunks.jsonl") == before
    changed = _config()
    changed["chunk_tokens"] = 60
    with pytest.raises(RuntimeError, match="configuration differs"):
        build(changed, root, resume=True, source_factory=_data, token_counter=_counter)
    with (root / "intermediate" / "20_filtered_documents.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("corrupt")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        build(_config(), root, resume=True, source_factory=_data, token_counter=_counter)


def test_v0_path_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="protected V0"):
        build(_config(), tmp_path / "vn_history_deployment" / "corpus_v1",
              source_factory=_data, token_counter=_counter)


def test_incomplete_stage_is_rejected(tmp_path: Path):
    root = tmp_path / "corpus_v1" / "run-partial"
    stage_dir = root / "intermediate"
    stage_dir.mkdir(parents=True)
    (stage_dir / "10_normalized_documents.jsonl.partial").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete temporary stage"):
        build(_config(), root, resume=True, source_factory=_data, token_counter=_counter)


def test_index_row_alignment_rejects_reordered_corpus(tmp_path: Path):
    service = RAGService()
    service.chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    service.ordered_chunk_id_sha256 = ordered_chunk_id_sha256(service.chunks)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"count": 2,
                                    "ordered_chunk_id_sha256": ordered_chunk_id_sha256(
                                        list(reversed(service.chunks)))}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="row order mismatch"):
        service._validate_index_manifest(manifest)


def test_bootstrap_local_paths_without_colab(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    result = bootstrap(tmp_path / "project")
    assert set(result["paths"]) == {"raw", "cache", "intermediate", "corpus_v1", "logs", "reports"}
    assert all(Path(path).is_dir() for path in result["paths"].values())
    with pytest.raises(RuntimeError, match="Google Colab"):
        bootstrap(tmp_path / "other", mount_drive=True)
