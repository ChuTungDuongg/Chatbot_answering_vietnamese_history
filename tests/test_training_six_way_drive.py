"""Tiny local tests; no model, network, corpus scan or Drive API."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.models.qwen import QwenRuntime
from app.rag.prompting import build_messages, build_no_rag_messages
from app.rag.response_modes import MODE_INSTRUCTIONS
from app.schemas import ChatRequest
from evaluation.schema import Question
from evaluation.six_way import (SYSTEMS, SYSTEM_NAMES, _append, _read_jsonl,
                                aggregate, score_prediction, select_systems)
from tools.drive_cli import copy_verified, sha256
from training.train_qwen3 import (SFT_LOSS_OPTIONS, assert_split_isolation, build_parser,
                                  effective_batch, lora_kwargs, prepare, read_sft,
                                  to_prompt_completion, validate_prompt_lengths)


@pytest.mark.parametrize("mode", ["concise", "standard", "detailed"])
def test_response_mode_crosses_schema_and_prompt_without_retrieval_change(mode):
    payload = ChatRequest(conversation_id=uuid4(), question="Chiến thắng Bạch Đằng?", response_mode=mode)
    assert payload.response_mode == mode
    assert MODE_INSTRUCTIONS[mode] in build_messages(payload.question, [], response_mode=mode)[0]["content"]
    no_rag = build_no_rag_messages(payload.question, response_mode=mode)
    assert MODE_INSTRUCTIONS[mode] in no_rag[0]["content"]
    assert len(no_rag) == 2 and "Nguồn được truy xuất" not in str(no_rag)


def test_response_default_and_invalid():
    assert ChatRequest(conversation_id=uuid4(), question="Lịch sử Việt Nam?").response_mode == "standard"
    with pytest.raises(ValidationError):
        ChatRequest(conversation_id=uuid4(), question="Lịch sử Việt Nam?", response_mode="verbose")
    with pytest.raises(ValueError):
        Settings(model_variant="sft", model_adapter_path=None)


def test_merged_model_requires_explicit_local_opt_in(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        QwenRuntime(model_id=str(merged))
    assert QwenRuntime(model_id=str(merged), allow_local_model=True).model_id == str(merged)


def _row(canonical, mode):
    return {"canonical_id": canonical, "response_mode": mode,
            "messages": [{"role": "system", "content": MODE_INSTRUCTIONS[mode]},
                         {"role": "user", "content": "Question"},
                         {"role": "assistant", "content": "Answer"}]}


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_training_parsing_batch_lora_and_split(tmp_path, monkeypatch):
    train = tmp_path / "train_sft.jsonl"
    valid = tmp_path / "validation_sft.jsonl"
    _write(train, [_row("a", "concise"), _row("a", "detailed")])
    _write(valid, [_row("b", "standard")])
    split = {"ids": {"train": ["a"], "validation": ["b"], "test": ["c"]}}
    (tmp_path / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
    args = build_parser().parse_args(["--train-file", str(train), "--validation-file", str(valid),
                                      "--output-dir", str(tmp_path / "run"), "--dry-run"])
    parsed_train, parsed_valid, manifest, resume = prepare(args)
    assert len(parsed_train) == 2 and len(parsed_valid) == 1 and resume is None
    assert manifest["train_sha256"] == sha256(train)
    assert effective_batch(2, 8, 1) == 16
    assert "q_proj" in lora_kwargs(args)["target_modules"]
    with pytest.raises(ValueError):
        assert_split_isolation(parsed_train, [_row("a", "standard")], split)
    _write(train, [_row("a", "concise"), _row("a", "concise")])
    with pytest.raises(ValueError, match="Duplicate"):
        read_sft(train)


def test_qlora_train_and_validation_use_conversational_prompt_completion(tmp_path):
    train = tmp_path / "train_sft.jsonl"
    valid = tmp_path / "validation_sft.jsonl"
    train_rows = [_row("a", "concise"), _row("a", "detailed")]
    valid_rows = [_row("b", "standard")]
    _write(train, train_rows)
    _write(valid, valid_rows)
    (tmp_path / "split_manifest.json").write_text(json.dumps({"ids": {
        "train": ["a"], "validation": ["b"], "test": ["c"]}}), encoding="utf-8")
    args = build_parser().parse_args(["--train-file", str(train), "--validation-file", str(valid),
                                      "--output-dir", str(tmp_path / "run"), "--dry-run"])
    parsed_train, parsed_valid, _, _ = prepare(args)
    for original, converted in zip(parsed_train, to_prompt_completion(parsed_train), strict=True):
        assert converted == {"prompt": original["messages"][:2],
                             "completion": [original["messages"][2]]}
        assert MODE_INSTRUCTIONS[original["response_mode"]] in converted["prompt"][0]["content"]
    assert to_prompt_completion(parsed_valid) == [{
        "prompt": valid_rows[0]["messages"][:2],
        "completion": [valid_rows[0]["messages"][2]],
    }]
    assert train_rows == parsed_train and valid_rows == parsed_valid
    with pytest.raises(ValueError, match="TEST ID"):
        assert_split_isolation(parsed_train, [_row("c", "standard")], {
            "ids": {"train": ["a"], "validation": ["c"], "test": ["c"]}})


def test_qlora_completion_mask_does_not_need_generation_markers():
    assert SFT_LOSS_OPTIONS == {"assistant_only_loss": False, "completion_only_loss": True}
    class TokenizerWithoutGenerationMask:
        chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        calls = []
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            assert "return_assistant_tokens_mask" not in kwargs
            return [1, 2, 3]
    tokenizer = TokenizerWithoutGenerationMask()
    row = _row("a", "concise")
    assert "{% generation %}" not in tokenizer.chat_template
    validate_prompt_lengths([row], tokenizer, 128)
    assert tokenizer.calls == [(row["messages"][:2], {
        "tokenize": True, "add_generation_prompt": True, "enable_thinking": False})]
    with pytest.raises(ValueError, match="before assistant answer"):
        validate_prompt_lengths([row], tokenizer, 34)


def test_qlora_dry_run_still_works_with_tiny_splits(tmp_path, capsys):
    from training.train_qwen3 import main
    train = tmp_path / "train_sft.jsonl"
    valid = tmp_path / "validation_sft.jsonl"
    _write(train, [_row("a", "concise")])
    _write(valid, [_row("b", "standard")])
    (tmp_path / "split_manifest.json").write_text(json.dumps({"ids": {
        "train": ["a"], "validation": ["b"], "test": ["c"]}}), encoding="utf-8")
    assert main(["--train-file", str(train), "--validation-file", str(valid),
                 "--output-dir", str(tmp_path / "run"), "--dry-run"]) == 0
    assert '"dry_run": true' in capsys.readouterr().out


def test_training_resume_requires_matching_config(tmp_path):
    train = tmp_path / "train_sft.jsonl"; valid = tmp_path / "validation_sft.jsonl"
    _write(train, [_row("a", "concise")]); _write(valid, [_row("b", "standard")])
    (tmp_path / "split_manifest.json").write_text(json.dumps({"ids": {"train": ["a"],
        "validation": ["b"], "test": []}}), encoding="utf-8")
    output = tmp_path / "run"
    args = build_parser().parse_args(["--train-file", str(train), "--validation-file", str(valid),
                                      "--output-dir", str(output)])
    _, _, manifest, _ = prepare(args)
    output.mkdir(); (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args.resume_from_checkpoint = "latest"
    assert prepare(args)[3] is None  # Interrupted before first checkpoint: restart same config.
    checkpoint = output / "checkpoints" / "checkpoint-1"; checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert prepare(args)[3] == checkpoint
    args.learning_rate = 1e-4
    with pytest.raises(ValueError, match="mismatch"):
        prepare(args)


def test_six_registry_and_metric_applicability():
    assert len(SYSTEM_NAMES) == 6
    assert [system.dense_backend for system in select_systems("all")] == [None, "faiss", "qdrant", None, "faiss", "qdrant"]
    assert SYSTEMS["vanilla_faiss"].model_type == "vanilla"
    assert SYSTEMS["sft_faiss"].model_type == "sft"
    question = Question(id="q", question="Khi nào?", category="chronology", gold_answer="Năm 938.",
                        relevant_chunk_ids=["c"], relevant_source_ids=["s"], required_facts=["Năm 938."])
    prediction = {"question_id": "q", "success": True, "answer": "Năm 938.", "sources": []}
    no_rag = score_prediction(question, prediction, SYSTEMS["vanilla_no_rag"])
    rag = score_prediction(question, prediction, SYSTEMS["vanilla_faiss"])
    assert no_rag["retrieval"]["chunk"]["hit_rate@1"] is None
    assert all(value is None for value in no_rag["citations"].values())
    assert no_rag["answer_metrics"]["required_fact_phrase_recall"] == 1
    assert rag["retrieval"]["chunk"]["hit_rate@1"] == 0
    assert aggregate([question], {"q": prediction}, SYSTEMS["vanilla_no_rag"])["overall"]["answer"]["exact_match"]["value"] == 1


def test_append_resume_truncates_only_partial_final_row(tmp_path):
    path = tmp_path / "predictions.jsonl"
    _append(path, {"question_id": "q1", "success": True})
    with path.open("ab") as stream:
        stream.write(b'{"question_id":"q2"')
    rows = _read_jsonl(path)
    assert set(rows) == {"q1"}
    _append(path, {"question_id": "q2", "success": False})
    assert set(_read_jsonl(path)) == {"q1", "q2"}


def test_drive_copy_refuses_mismatch_and_verifies(tmp_path):
    left = tmp_path / "source.jsonl"; right = tmp_path / "drive" / "source.jsonl"
    left.write_text("first\n", encoding="utf-8")
    assert copy_verified(left, right) == "copied"
    assert copy_verified(left, right) == "unchanged"
    left.write_text("second\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        copy_verified(left, right)
    assert copy_verified(left, right, overwrite=True) == "copied"
    assert sha256(left) == sha256(right)


def test_six_way_fake_generation_persists_and_resumes(tmp_path, monkeypatch):
    import evaluation.six_way as six
    from app.config import settings
    import app.models.qwen as qwen
    monkeypatch.setattr(settings, "app_mode", settings.app_mode)
    monkeypatch.setattr(settings, "corpus_path_override", settings.corpus_path_override)
    monkeypatch.setattr(settings, "retrieval_root", settings.retrieval_root)
    question = {"id": "q1", "question": "Năm nào?", "category": "chronology", "gold_answer": "Năm 938."}
    test_file = tmp_path / "test.jsonl"
    _write(test_file, [question])
    monkeypatch.setattr(six, "FROZEN_CANONICAL_SHA", "tiny-test-only")
    monkeypatch.setattr(six, "FROZEN_TEST_SHA", sha256(test_file))
    (tmp_path / "split_manifest.json").write_text(json.dumps({"seed": 42,
        "canonical_dataset_sha256": "tiny-test-only", "split_sha256": {"test": sha256(test_file)},
        "ids": {"test": ["q1"]}}), encoding="utf-8")
    calls = []
    class FakeModel:
        def __init__(self, **kwargs):
            calls.append("load")
        async def generate(self, messages, *, max_new_tokens):
            calls.append("generate")
            return "Năm 938.", SimpleNamespace(input_tokens=5, output_tokens=3,
                model_id="fake", model_revision="fixture")
    monkeypatch.setattr(qwen, "QwenRuntime", FakeModel)
    args = six.build_parser().parse_args(["--test-file", str(test_file),
        "--systems", "vanilla_no_rag", "--output-dir", str(tmp_path / "out"),
        "--corpus-path", str(tmp_path / "no-corpus.jsonl"),
        "--retrieval-root", str(tmp_path / "no-index")])
    result = asyncio.run(six.run(args))
    assert calls == ["load", "generate"]
    assert result["systems"]["vanilla_no_rag"]["question_count"] == 1
    assert set(_read_jsonl(tmp_path / "out" / "vanilla_no_rag" / "predictions.jsonl")) == {"q1"}
    args.resume = True
    asyncio.run(six.run(args))
    assert calls == ["load", "generate"]
    args.max_new_tokens = 12
    with pytest.raises(ValueError, match="Resume identity"):
        asyncio.run(six.run(args))


def test_qdrant_not_ready_fails_without_faiss_fallback(tmp_path, monkeypatch):
    import evaluation.six_way as six
    question = {"id": "q1", "question": "Năm nào?", "category": "chronology"}
    test_file = tmp_path / "test.jsonl"; _write(test_file, [question])
    monkeypatch.setattr(six, "FROZEN_CANONICAL_SHA", "tiny-test-only")
    monkeypatch.setattr(six, "FROZEN_TEST_SHA", sha256(test_file))
    (tmp_path / "split_manifest.json").write_text(json.dumps({"seed": 42,
        "canonical_dataset_sha256": "tiny-test-only", "split_sha256": {"test": sha256(test_file)},
        "ids": {"test": ["q1"]}}), encoding="utf-8")
    args = six.build_parser().parse_args(["--test-file", str(test_file),
        "--systems", "vanilla_qdrant", "--output-dir", str(tmp_path / "out"),
        "--retrieval-root", str(tmp_path / "index"), "--dry-run"])
    (tmp_path / "index" / "qdrant.partial").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not finalized"):
        asyncio.run(six.run(args))


def test_six_way_reuses_identical_faiss_retrieval_for_vanilla_and_sft(tmp_path, monkeypatch):
    import evaluation.six_way as six
    import app.models.qwen as qwen
    import app.rag.retrieval as retrieval
    import app.services.rag_service as rag_service
    from app.config import settings
    for key in ("app_mode", "corpus_path_override", "retrieval_root", "retrieval_dense_backend"):
        monkeypatch.setattr(settings, key, getattr(settings, key))
    test_file = tmp_path / "test.jsonl"
    _write(test_file, [{"id": "q1", "question": "Năm nào?", "category": "chronology",
                        "relevant_chunk_ids": ["c"], "relevant_source_ids": ["s"]}])
    monkeypatch.setattr(six, "FROZEN_CANONICAL_SHA", "tiny-test-only")
    monkeypatch.setattr(six, "FROZEN_TEST_SHA", sha256(test_file))
    (tmp_path / "split_manifest.json").write_text(json.dumps({"seed": 42,
        "canonical_dataset_sha256": "tiny-test-only", "split_sha256": {"test": sha256(test_file)},
        "ids": {"test": ["q1"]}}), encoding="utf-8")
    corpus = tmp_path / "corpus"; (corpus / "runtime").mkdir(parents=True)
    (corpus / "runtime" / "manifest.json").write_text(json.dumps({"corpus": {
        "corpus_sha256": "tiny", "ordered_chunk_id_sha256": "tiny", "count": 1}}), encoding="utf-8")
    index = tmp_path / "index"; (index / "faiss").mkdir(parents=True)
    (index / "bm25s_index").mkdir()
    (index / "faiss" / "manifest.json").write_text("{}", encoding="utf-8")
    (index / "bm25s_index" / "phase9_manifest.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter"; adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls = []
    class FakeModel:
        def __init__(self, **kwargs):
            calls.append(("model", kwargs["adapter_path"] is not None))
        async def generate(self, messages, *, max_new_tokens):
            return "Năm 938 [c].", SimpleNamespace(input_tokens=6, output_tokens=4,
                model_id="fake", model_revision="fixture")
    class FakeService:
        def load(self):
            calls.append(("service", "load"))
        def shutdown(self):
            pass
    class FakeRetriever:
        def __init__(self, service):
            pass
        def retrieve(self, question, final_k):
            calls.append(("retrieve", question))
            return {"final_context": [{"chunk_id": "c", "source_id": "s",
                    "title": "Fixture", "text": "Năm 938."}], "query_variants": [question]}
    monkeypatch.setattr(qwen, "QwenRuntime", FakeModel)
    monkeypatch.setattr(rag_service, "RAGService", FakeService)
    monkeypatch.setattr(retrieval, "HybridRetriever", FakeRetriever)
    args = six.build_parser().parse_args(["--test-file", str(test_file),
        "--systems", "vanilla_faiss,sft_faiss", "--adapter-path", str(adapter),
        "--output-dir", str(tmp_path / "out"), "--corpus-path", str(corpus / "chunks.jsonl"),
        "--retrieval-root", str(index)])
    result = asyncio.run(six.run(args))
    assert calls.count(("retrieve", "Năm nào?")) == 1
    assert ("model", False) in calls and ("model", True) in calls
    assert result["systems"]["sft_faiss"]["question_count"] == 1
