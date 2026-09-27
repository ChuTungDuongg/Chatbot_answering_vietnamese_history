"""Offline UVW-shaped coverage of Corpus V1."""

import json
from pathlib import Path

import pytest

from scripts.corpus_v1.audit import audit
from scripts.corpus_v1.chunking import chunk_text
from scripts.corpus_v1.cli import main as corpus_cli
from scripts.corpus_v1.history_filter import classify
from scripts.corpus_v1.pipeline import _source_iterator, build
from scripts.corpus_v1.provenance import digest_file
from scripts.corpus_v1.source import article_url, detect_fields, inspect, preset, resolve_source
from scripts.colab.bootstrap import bootstrap


SHA = "a" * 40
HISTORY = "Lịch sử triều đại Đại Việt và di sản văn hóa. " * 4
FIELDS = ["id", "title", "content", "num_chars", "num_sentences", "quality_score",
          "wikidata_id", "main_category"]


def row(identifier, title, content, category, quality=1):
    return {"id": identifier, "title": title, "content": content, "num_chars": len(content),
            "num_sentences": 2, "quality_score": quality, "wikidata_id": "Q1",
            "main_category": category}


def fixture_rows():
    return {
        "train": [row("Vua_Đại_Việt", "Vua Đại Việt", HISTORY, "người", 1),
                  row("Taxon", "Loài cây", "Mô tả lá và hoa của loài cây. " * 5, "đơn vị phân loại", 10)],
        "validation": [row("Vua_Đại_Việt", "Bản sao", HISTORY, "người", 9),
                       row("Hà_Nội", "Hà Nội", "Thành phố có di tích lịch sử. " * 5,
                           "thành phố trực thuộc trung ương của Việt Nam", 2)],
        "test": [row("Phần_mềm", "Gói phần mềm", "Thư viện xử lý chuỗi dữ liệu. " * 5,
                     "gói phần mềm", 10),
                 row("Nhà_văn", "Nhà văn", "Nhà văn viết tác phẩm về văn hóa. " * 5,
                     "nhà văn", 2)],
    }


def source_info():
    return {"dataset_id": "undertheseanlp/UVW-2026", "dataset_config": "default",
            "requested_revision": "main", "resolved_revision_sha": SHA,
            "available_splits": ["train", "validation", "test"],
            "expected_split_sizes": {name: len(values) for name, values in fixture_rows().items()},
            "features": FIELDS, "license": "cc-by-sa-4.0",
            "source_url": "https://huggingface.co/datasets/undertheseanlp/UVW-2026"}


def config(**overrides):
    result = preset("uvw-2026")
    result.update(shard_size=1, chunk_tokens=24, chunk_overlap=4)
    result.update(overrides)
    return result


def counter(value):
    return len(value.split())


def source(split):
    return fixture_rows()[split]


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_preset_schema_url():
    selected = preset("uvw-2026")
    assert selected["split"] == "all"
    assert detect_fields(FIELDS, {}) == {"title": "title", "text": "content", "id": "id", "url": None}
    assert article_url(fixture_rows()["train"][0], selected["fields"], "uvw-2026").endswith(
        "/Vua_%C4%90%E1%BA%A1i_Vi%E1%BB%87t")


def test_requested_revision_resolves_to_sha_and_inspection_trims(monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub

    class Card:
        def to_dict(self):
            return {"license": "cc-by-sa-4.0", "configs": [{"config_name": "default"}],
                    "dataset_info": {"features": [{"name": name} for name in FIELDS],
                                     "splits": [{"name": split, "num_examples": 2}
                                                for split in ("train", "validation", "test")]}}

    class Api:
        def dataset_info(self, dataset_id, revision):
            assert dataset_id == "undertheseanlp/UVW-2026" and revision == "main"
            return SimpleNamespace(sha=SHA, card_data=Card())

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    info = resolve_source(config())
    assert info["requested_revision"] == "main" and info["resolved_revision_sha"] == SHA
    assert info["expected_split_sizes"] == dict.fromkeys(("train", "validation", "test"), 2)
    preview = inspect(config(), 1, source_info=info, loader=lambda split: fixture_rows()[split])
    assert preview["field_mapping"]["text"] == "content"
    assert all(len(preview["examples"][split][0]["content"]) <= 241 for split in preview["source_splits"])

    class BadApi:
        def dataset_info(self, dataset_id, revision):
            return SimpleNamespace(sha="main", card_data=Card())

    monkeypatch.setattr(huggingface_hub, "HfApi", BadApi)
    with pytest.raises(RuntimeError, match="immutable commit SHA"):
        resolve_source(config())


@pytest.mark.parametrize("title,category", [
    ("Vua Đại Việt", "người"), ("Hoàng đế La Mã", "người"),
    ("Nhân vật lịch sử", "người"), ("Chính trị gia", "chính trị gia"),
    ("Tướng quân", "quân nhân"), ("Nhà văn", "nhà văn"),
    ("Học giả", "học giả"), ("Văn minh cổ đại", "nền văn minh"),
    ("Chiến tranh", "chiến tranh"), ("Trận đánh", "trận đánh"),
    ("Hiệp ước", "hiệp ước"), ("Quốc gia", "quốc gia"),
    ("Thành phố", "thành phố"), ("Vùng lịch sử", "vùng"),
    ("Di chỉ khảo cổ", "địa điểm khảo cổ"), ("Di sản", "di sản"),
    ("Chùa cổ", "chùa"), ("Sông lịch sử", "sông")])
def test_broad_history_not_dropped(title, category):
    assert classify(title, "Tư liệu về địa danh và con người qua nhiều thời kỳ. " * 2,
                    category)[1] in ("KEEP", "REVIEW")


@pytest.mark.parametrize("title,category", [
    ("Thư viện lập trình", "gói phần mềm"), ("Điện thoại mẫu", "thiết bị điện tử"),
    ("Trận đấu hôm nay", "trận đấu bóng đá"), ("Tài liệu API", "phần mềm"),
    ("Loài cây", "đơn vị phân loại")])
def test_unrelated_category_can_drop(title, category):
    assert classify(title, "Mô tả kỹ thuật và thông số hiện tại. " * 3, category)[1] == "DROP"


def test_unknown_category_and_chunk_budget():
    text = "Mô tả đối tượng có nguồn tư liệu địa phương. " * 3
    assert classify("Đối tượng", text, None)[1] == "REVIEW"
    assert classify("Đối tượng", text, "đơn vị phân loại")[1] == "DROP"
    assert "person_place_heritage_category" in classify("Nhân vật", text, "người")[2]
    long_text = "Lịch sử Việt Nam. " * 15 + chr(10) * 2 + "a" * 200
    chunks = list(chunk_text(long_text, counter, budget=16, overlap=3))
    assert chunks and all(size <= 16 for _, _, size in chunks)
    assert "".join(value for _, value, _ in chunk_text("a" * 200, len, budget=16, overlap=0)) == "a" * 200
    odd = lambda value: len(value.split()) + (7 if " Việt Nam" in value else 0)
    assert all(size <= 16 for _, _, size in chunk_text(long_text, odd, budget=16, overlap=3))


def test_resume_uses_iterable_skip_when_available():
    class Stream:
        def __init__(self, values):
            self.values = values

        def skip(self, offset):
            assert offset == 2
            return iter(self.values[offset:])

        def __iter__(self):
            pytest.fail("sequential iteration used instead of skip")

    assert list(_source_iterator(Stream([0, 1, 2, 3]), 2)) == [2, 3]


def test_full_multisplit_resume_and_provenance(tmp_path):
    root = tmp_path / "corpus_v1" / "full"
    first = build(config(), root, source_factory=source, source_info=source_info(), token_counter=counter)
    assert first["source_complete"] is True
    assert first["observed_split_sizes"] == first["expected_split_sizes"] == dict.fromkeys(
        ("train", "validation", "test"), 2)
    assert first["filter_counts"]["DROP"] == 2
    assert first["duplicate_document_count"] == 1
    train_source = records(root / "intermediate" / "shards" / "train" / "part-000000.records.jsonl")[0]["source_id"]
    validation_source = records(root / "intermediate" / "shards" / "validation" / "part-000000.records.jsonl")[0]["source_id"]
    assert train_source != validation_source
    docs, chunks = records(root / "documents.jsonl"), records(root / "chunks.jsonl")
    assert first["document_count"] == len(docs) == 3
    assert len({doc["document_id"] for doc in docs}) == len(docs)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert {doc["source_split"] for doc in docs} == {"train", "validation", "test"}
    assert all(doc["schema_version"] == 2 and doc["source_revision_sha"] == SHA for doc in docs)
    assert all(chunk["schema_version"] == 2 and chunk["source_article_id"] for chunk in chunks)
    assert all(chunk["token_count"] <= 24 for chunk in chunks)
    assert docs[0]["source_metadata"]["quality_score"] == 1
    assert docs[0]["source_metadata"]["wikidata_id"] == "Q1"
    assert docs[0]["url"].endswith("/Vua_%C4%90%E1%BA%A1i_Vi%E1%BB%87t")
    assert all(digest_file(root / name) == digest for name, digest in first["hashes"].items())
    report = audit(root)
    assert not report["hash_mismatches"]
    assert report["source_split_distribution"]["document"] == {"train": 1, "validation": 1, "test": 1}
    assert report["quality_score_distribution"]["1"] == 1
    second = build(config(), root, resume=True, source_factory=lambda _: pytest.fail("source reloaded"),
                   token_counter=lambda _: pytest.fail("tokenizer reloaded"))
    assert first == second
    build(config(), tmp_path / "corpus_v1" / "repeat", source_factory=source,
          source_info=source_info(), token_counter=counter)
    assert [d["document_id"] for d in docs] == [d["document_id"] for d in records(
        tmp_path / "corpus_v1" / "repeat" / "documents.jsonl")]
    changed_info = source_info()
    changed_info["resolved_revision_sha"] = "b" * 40
    build(config(), tmp_path / "corpus_v1" / "new_revision", source_factory=source,
          source_info=changed_info, token_counter=counter)
    assert docs[0]["source_id"] != records(tmp_path / "corpus_v1" / "new_revision" / "documents.jsonl")[0]["source_id"]
    with pytest.raises(RuntimeError, match="configuration differs"):
        build(config(chunk_tokens=32), root, resume=True, source_factory=source, token_counter=counter)
    with (root / "intermediate" / "shards" / "train" / "part-000000.records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("corrupt")
    with pytest.raises(RuntimeError, match="Corrupt completed shard"):
        build(config(), root, resume=True, source_factory=source, token_counter=counter)


def test_interrupted_shard_recovery(tmp_path, monkeypatch):
    import scripts.corpus_v1.pipeline as pipeline
    root = tmp_path / "corpus_v1" / "interrupted"
    original = pipeline.commit_shard
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            directory = root / "intermediate" / "shards" / "train"
            (directory / "part-000001.records.jsonl.partial").write_text("partial", encoding="utf-8")
            raise KeyboardInterrupt("simulated disconnect")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "commit_shard", interrupt)
    with pytest.raises(KeyboardInterrupt):
        build(config(), root, source_factory=source, source_info=source_info(), token_counter=counter)
    first_hash = digest_file(root / "intermediate" / "shards" / "train" / "part-000000.records.jsonl")
    monkeypatch.setattr(pipeline, "commit_shard", original)
    final = build(config(), root, resume=True, source_factory=source, token_counter=counter)
    assert final["source_complete"] is True
    assert digest_file(root / "intermediate" / "shards" / "train" / "part-000000.records.jsonl") == first_hash
    assert not (root / "intermediate" / "shards" / "train" / "part-000001.records.jsonl.partial").exists()


def test_initial_manifest_disconnect_recovery(tmp_path, monkeypatch):
    import scripts.corpus_v1.pipeline as pipeline
    root = tmp_path / "corpus_v1" / "before_source_manifest"
    original = pipeline.atomic_json

    def interrupt(path, data):
        if path.name == "source_manifest.json":
            raise KeyboardInterrupt("simulated early disconnect")
        return original(path, data)

    monkeypatch.setattr(pipeline, "atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        build(config(), root, source_factory=source, source_info=source_info(), token_counter=counter)
    monkeypatch.setattr(pipeline, "atomic_json", original)
    assert build(config(), root, resume=True, source_factory=source, source_info=source_info(),
                 token_counter=counter)["source_complete"] is True


def test_pilot_cli_and_count_mismatch(tmp_path, capsys):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "metadata.json").write_text(json.dumps(source_info()), encoding="utf-8")
    for split, values in fixture_rows().items():
        (fixture / f"{split}.jsonl").write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in values) + "\n", encoding="utf-8")
    root = tmp_path / "corpus_v1" / "pilot"
    assert corpus_cli(["build", "--preset", "uvw-2026", "--offline-fixture-dir", str(fixture),
                       "--max-records-per-split", "1", "--shard-size", "1", "--output", str(root)]) == 0
    capsys.readouterr()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_scope"] == "pilot" and manifest["source_complete"] is False
    assert manifest["observed_split_sizes"] == dict.fromkeys(("train", "validation", "test"), 1)
    assert json.loads((root / "filter_audit.json").read_text(encoding="utf-8"))["quality_score_distribution"]
    assert corpus_cli(["audit", "--corpus", str(root)]) == 0
    capsys.readouterr()
    info = source_info()
    info["expected_split_sizes"]["test"] = 3
    with pytest.raises(RuntimeError, match="Incomplete source split test"):
        build(config(), tmp_path / "corpus_v1" / "short", source_factory=source,
              source_info=info, token_counter=counter)


def test_v0_path_and_bootstrap(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="protected V0"):
        build(config(), tmp_path / "vn_history_deployment" / "corpus_v1",
              source_factory=source, source_info=source_info(), token_counter=counter)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    result = bootstrap(tmp_path / "project")
    assert set(result["paths"]) == {"raw", "cache", "intermediate", "corpus_v1", "logs", "reports"}
    assert all(Path(path).is_dir() for path in result["paths"].values())
    assert result["free_disk_bytes"] > 0
    with pytest.raises(RuntimeError, match="Google Colab"):
        bootstrap(tmp_path / "other", mount_drive=True)
