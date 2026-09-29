"""Model-free citation V2 attribution and frozen V1 compatibility tests."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.rag.prompting import SYSTEM_PROMPT
from app.rag.response_modes import mode_instruction, sft_system_instruction
from evaluation.silver_citation_v2 import (
    _context_message, cite_answer, fact_support, jsonl_bytes, load_inputs, run, strip_citations, transform_row,
    validate_row,
)
from evaluation.silver_v1 import sft_row


def fixture(facts, answer, *, supports=None, mode="standard", answerable=True,
            gold=None, paragraphs=None):
    chunks = {
        "chk_a": {"chunk_id": "chk_a", "source_id": "src_a", "title": "Nguồn A",
                  "text": "Triều Lý có sự kiện năm 1010. Nhà Trần có sự kiện năm 1225."},
        "chk_b": {"chunk_id": "chk_b", "source_id": "src_b", "title": "Nguồn B",
                  "text": "Nhà Trần có sự kiện năm 1225. Dữ kiện khác."},
    }
    record = {"id": "q1", "question": "Chuyện gì xảy ra?", "gold_answer": answer,
              "required_facts": facts, "relevant_chunk_ids": list(chunks),
              "relevant_source_ids": ["src_a", "src_b"],
              "gold_citation_source_ids": gold if gold is not None else ["src_a", "src_b"],
              "factual_paragraph_indices": paragraphs if paragraphs is not None else [0],
              "answerable": answerable}
    evidence = []
    for chunk_id, indices in (supports or {}).items():
        evidence.append({"chunk_id": chunk_id, "quote": chunks[chunk_id]["text"][:20],
                         "decision": "relevant", "supports_fact_indices": indices})
    audit = {"evidence": evidence, "pass_b": "checked"}
    v1 = {"id": f"q1:{mode}", "canonical_id": "q1", "response_mode": mode,
          "messages": [{"role": "system", "content": sft_system_instruction(mode)},
                       {"role": "user", "content": _context_message(record, chunks)},
                       {"role": "assistant", "content": answer}],
          "difficulty": "medium", "category": "chronology", "question_type": "chronology",
          "relevant_chunk_ids": list(chunks), "relevant_source_ids": ["src_a", "src_b"],
          "annotation_origin": "automatic", "dataset_tier": "SILVER"}
    return record, audit, chunks, v1


def test_one_fact_unicode_and_exact_stripping():
    fact = "Triều Lý có sự kiện năm 1010."
    record, audit, chunks, v1 = fixture([fact], fact, supports={"chk_a": [0]})
    v2, info = transform_row(v1, record, audit, chunks)
    assert v2["messages"][2]["content"] == fact + " [chk_a]"
    assert strip_citations(v2["messages"][2]["content"], {"chk_a"}) == fact
    assert v2["messages"][0]["content"] == SYSTEM_PROMPT + " " + mode_instruction("standard")
    assert info["status"] == "cited"
    assert v2["messages"][1] == v1["messages"][1]


def test_multiple_supports_choose_first_canonical_chunk():
    fact = "Nhà Trần có sự kiện năm 1225."
    record, audit, chunks, v1 = fixture([fact], fact, supports={"chk_b": [0], "chk_a": [0]})
    v2, info = transform_row(v1, record, audit, chunks)
    assert info["cited_chunks"] == ["chk_a"]
    assert v2["messages"][2]["content"] == fact + " [chk_a]"


def test_different_facts_get_their_own_support():
    first = "Triều Lý có sự kiện năm 1010."
    second = "Nhà Trần có sự kiện năm 1225."
    record, audit, chunks, v1 = fixture([first, second], first + " " + second,
                                         supports={"chk_a": [0], "chk_b": [1]})
    v2, info = transform_row(v1, record, audit, chunks)
    assert v2["messages"][2]["content"] == first + " [chk_a] " + second + " [chk_b]"
    assert info["cited_chunks"] == ["chk_a", "chk_b"]
    assert strip_citations(v2["messages"][2]["content"], set(chunks)) == v1["messages"][2]["content"]


def test_detailed_target_cites_exact_added_fact_but_reports_uncited_prose():
    fact = "Triều Lý có sự kiện năm 1010."
    answer = "Bối cảnh được kể theo cách khác.\n\nChi tiết liên quan: " + fact
    record, audit, chunks, v1 = fixture([fact], answer, supports={"chk_a": [0]}, mode="detailed")
    v2, info = transform_row(v1, record, audit, chunks)
    assert v2["messages"][2]["content"] == answer + " [chk_a]"
    assert info["status"] == "partial" and info["uncited_factual_paragraph_indices"] == [0]


def test_multiple_factual_paragraphs_with_exact_facts():
    first = "Triều Lý có sự kiện năm 1010."
    second = "Nhà Trần có sự kiện năm 1225."
    record, audit, chunks, v1 = fixture([first, second], first + "\n\n" + second,
                                         supports={"chk_a": [0], "chk_b": [1]}, paragraphs=[0, 1])
    v2, info = transform_row(v1, record, audit, chunks)
    assert v2["messages"][2]["content"] == first + " [chk_a]\n\n" + second + " [chk_b]"
    assert info["status"] == "cited"


def test_unmatched_second_factual_paragraph_is_exposed():
    fact = "Triều Lý có sự kiện năm 1010."
    record, audit, chunks, v1 = fixture([fact], fact + "\n\nMột đoạn diễn giải khác.",
                                         supports={"chk_a": [0]}, paragraphs=[0, 1])
    _, info = transform_row(v1, record, audit, chunks)
    assert info["status"] == "partial"
    assert info["uncited_factual_paragraph_indices"] == [1]


def test_single_audited_chunk_can_cite_whole_one_paragraph():
    record, audit, chunks, v1 = fixture(["Fact A.", "Fact B."], "A free-form summary.",
                                         supports={"chk_a": [0, 1]}, gold=["src_a"])
    record["relevant_chunk_ids"] = ["chk_a"]
    record["relevant_source_ids"] = ["src_a"]
    v1["relevant_chunk_ids"] = ["chk_a"]
    v1["relevant_source_ids"] = ["src_a"]
    v1["messages"][1]["content"] = _context_message(record, chunks)
    v2, info = transform_row(v1, record, audit, chunks)
    assert v2["messages"][2]["content"] == "A free-form summary. [chk_a]"
    assert info["status"] == "paragraph_cited"


def test_unanswerable_stays_uncited():
    answer = "Nguồn chưa đủ để xác nhận tiền đề."
    record, audit, chunks, v1 = fixture(["Fact A."], answer, supports={"chk_a": [0]}, answerable=False)
    v2, info = transform_row(v1, record, audit, chunks)
    assert v2["messages"][2]["content"] == answer
    assert info["status"] == "unanswerable_unchanged"


def test_missing_audit_mapping_is_unresolved():
    fact = "Triều Lý có sự kiện năm 1010."
    record, audit, chunks, v1 = fixture([fact], fact)
    v2, info = transform_row(v1, record, None, chunks)
    assert v2["messages"][2]["content"] == fact
    assert info["status"] == "unresolved" and info["unresolved_facts"] == [0]


def test_outside_gold_source_never_cited_and_validator_rejects_it():
    fact = "Nhà Trần có sự kiện năm 1225."
    record, audit, chunks, v1 = fixture([fact], fact, supports={"chk_b": [0]}, gold=["src_a"])
    v2, info = transform_row(v1, record, audit, chunks)
    assert info["status"] == "unresolved"
    corrupted = deepcopy(v2)
    corrupted["messages"][2]["content"] += " [chk_b]"
    with pytest.raises(ValueError, match="gold sources"):
        validate_row(v1, corrupted, record, chunks)


def test_nonexistent_audit_chunk_and_invalid_alias_rejected():
    fact = "Triều Lý có sự kiện năm 1010."
    record, audit, chunks, v1 = fixture([fact], fact, supports={"chk_a": [0]})
    bad_audit = deepcopy(audit)
    bad_audit["evidence"][0]["chunk_id"] = "chk_missing"
    with pytest.raises(ValueError, match="Invalid audit evidence"):
        fact_support(record, bad_audit, chunks)
    v2, _ = transform_row(v1, record, audit, chunks)
    for alias in ("S1", "chk_missing"):
        corrupted = deepcopy(v2)
        corrupted["messages"][2]["content"] = fact + f" [{alias}]"
        with pytest.raises(ValueError):
            validate_row(v1, corrupted, record, chunks)


def test_v1_derivation_remains_bytewise_in_behavior():
    record, audit, chunks, _ = fixture(["Triều Lý có sự kiện năm 1010."],
                                       "Triều Lý có sự kiện năm 1010.", supports={"chk_a": [0]})
    record.update(difficulty="medium", category="chronology", question_type="chronology")
    class Corpus:
        def chunk(self, chunk_id):
            return chunks.get(chunk_id)
    before = sft_row(record, "standard", Corpus())
    original = deepcopy(before)
    transformed, _ = transform_row(before, record, audit, chunks)
    assert before == original == sft_row(record, "standard", Corpus())
    assert transformed["messages"][0]["content"] != before["messages"][0]["content"]
    assert before["messages"][0]["content"] == sft_system_instruction("standard")


def test_atomic_output_validation_and_training_loader_compatibility(tmp_path, monkeypatch):
    import evaluation.silver_citation_v2 as v2_module
    from training.train_qwen3 import build_parser, prepare

    fact = "Triều Lý có sự kiện năm 1010."
    record, audit, chunks, v1 = fixture([fact], fact, supports={"chk_a": [0]})
    train, _ = transform_row(v1, record, audit, chunks)
    validation = deepcopy(train)
    validation["id"] = "q2:standard"
    validation["canonical_id"] = "q2"
    output = {"train": [train], "validation": [validation]}
    hashes = {part: hashlib.sha256(jsonl_bytes(rows)).hexdigest() for part, rows in output.items()}
    split = {"ids": {"train": ["q1"], "validation": ["q2"], "test": []}}
    split_dir = tmp_path / "source_split"
    split_dir.mkdir()
    split_bytes = json.dumps(split).encode()
    (split_dir / "split_manifest.json").write_bytes(split_bytes)
    split_sha = hashlib.sha256(split_bytes).hexdigest()
    manifest = {"split_manifest_sha256": split_sha, "train_sft_sha256": hashes["train"],
                "validation_sft_sha256": hashes["validation"], "corpus_sha256": "fixture",
                "train_sft_v2_sha256": hashes["train"],
                "validation_sft_v2_sha256": hashes["validation"],
                "provenance": {"audit_file_sha256": {}}}
    stats = {"train_rows": 1, "validation_rows": 1}
    monkeypatch.setattr(v2_module, "build", lambda args: (output, stats, manifest, []))
    args = SimpleNamespace(action="derive", dry_run=False, output_dir=tmp_path / "v2",
                           train_split=split_dir / "train.jsonl")
    run(args)
    assert sorted(path.name for path in args.output_dir.iterdir()) == [
        "manifest.json", "split_manifest.json", "stats.json", "train_sft.jsonl", "validation_sft.jsonl"]
    with pytest.raises(FileExistsError):
        run(args)
    args.action = "validate"
    run(args)
    training_args = build_parser().parse_args(["--train-file", str(args.output_dir / "train_sft.jsonl"),
        "--validation-file", str(args.output_dir / "validation_sft.jsonl"),
        "--output-dir", str(tmp_path / "model"), "--dry-run"])
    loaded_train, loaded_validation, _, _ = prepare(training_args)
    assert len(loaded_train) == len(loaded_validation) == 1


def test_output_cannot_be_nested_in_frozen_v1(tmp_path):
    args = SimpleNamespace(train_sft_v1=tmp_path / "v1" / "train_sft.jsonl",
                           output_dir=tmp_path / "v1" / "citation_v2")
    with pytest.raises(ValueError, match="outside the frozen V1"):
        load_inputs(args)
