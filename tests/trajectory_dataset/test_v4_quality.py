from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.trajectory_dataset import cli
from training.trajectory_dataset.adapters.agent_flan import normalize_agent_flan
from training.trajectory_dataset.audit import audit_rows, tokenizer_audit
from training.trajectory_dataset.builders.custom_history import (
    SYNTHETIC_MARKER,
    CustomBuildConfig,
    QueryPlan,
    build_custom_trajectories,
    classify_subject,
    compact_observation,
    load_seed_records,
    task_eligible,
)
from training.trajectory_dataset.io_utils import atomic_write_jsonl, read_jsonl
from training.trajectory_dataset.preprocess import analyze_truncation, build_canonical_sft_example
from training.trajectory_dataset.retrieval import FixtureRetriever, apply_rerank_batch_override
from training.trajectory_dataset.schema import DEFAULT_SYSTEM_PROMPT, SEARCH_HISTORY_TOOL, make_trajectory
from training.trajectory_dataset.split import source_groups, split_trajectories
from training.trajectory_dataset.teacher.base import TeacherResponse
from training.trajectory_dataset.teacher.enhance import enhance_rows
from training.trajectory_dataset.validate import validate_rows


def record(chunk_id: str, title: str, subject_type: str, text: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text or f"{title} diễn ra năm 1945. Sự kiện có kết quả và ý nghĩa lịch sử quan trọng.",
        "url": f"https://example.test/{chunk_id}",
        "metadata": {
            "subject_type": subject_type,
            "content_facets": ["nguyên nhân", "kết quả"],
            "countries": ["Việt Nam"],
        },
    }


def write_corpus(path: Path, rows: list[dict]) -> Path:
    atomic_write_jsonl(path, rows)
    return path


def config(**counts: int) -> CustomBuildConfig:
    return CustomBuildConfig(
        task_counts=counts,
        top_k=7,
        max_corpus_records=100,
        seed=9,
        observation_char_budget=2_000,
        max_result_text_chars=500,
    )


def tool_arguments(row: dict) -> list[dict]:
    return [
        call["function"]["arguments"]
        for message in row["messages"]
        for call in message.get("tool_calls", [])
    ]


def test_subject_classification_and_analytical_eligibility_are_semantic():
    person = record("p", "Võ Nguyên Giáp", "person")
    location = record("l", "Cần Thơ", "location")
    date = record("d", "2 tháng 9 năm 1945", "date")
    topic = record("t", "Một chủ đề chung", "topic")
    event = record("e", "Cách mạng tháng Tám", "event")

    assert [classify_subject(row) for row in (person, location, date, topic, event)] == [
        "person", "location", "date", "topic", "event",
    ]
    assert not any(task_eligible(row, "cause") for row in (person, location, date, topic))
    assert not any(task_eligible(row, "summary") for row in (location, date, topic))
    assert task_eligible(person, "summary")
    assert task_eligible(event, "cause") and task_eligible(event, "summary")
    assert classify_subject({"title": "Trần Hưng Đạo", "text": "Danh tướng nhà Trần."}) == "person"


def test_compare_pairs_are_same_type_distinct_unique_and_questions_are_deduped(tmp_path: Path):
    rows = [
        record("a1", "Khởi nghĩa A", "event"),
        record("a2", "Khởi nghĩa A", "event"),
        record("b", "Khởi nghĩa B", "event"),
        record("c", "Khởi nghĩa C", "event"),
        record("where", "Cần Thơ", "location"),
    ]
    corpus = write_corpus(tmp_path / "corpus.jsonl", rows)
    built = list(build_custom_trajectories(corpus, FixtureRetriever(rows), config=config(compare=2)))

    assert len(built) == 2
    assert len({row["provenance"]["compare_pair_key"] for row in built}) == 2
    assert len({next(m["content"] for m in row["messages"] if m["role"] == "user") for row in built}) == 2
    for row in built:
        provenance = row["provenance"]
        assert provenance["subject_type"] == provenance["secondary_subject_type"] == "event"
        assert provenance["primary_title"] != provenance["secondary_title"]
        assert len(provenance["source_groups"]) == 2


def test_duplicate_title_chunks_do_not_underfill_or_emit_duplicate_questions(tmp_path: Path):
    rows = [
        record("a1", "Chiến dịch A", "event"),
        record("a2", "Chiến dịch A", "event"),
        record("b", "Chiến dịch B", "event"),
    ]
    corpus = write_corpus(tmp_path / "corpus.jsonl", rows)
    built = list(build_custom_trajectories(corpus, FixtureRetriever(rows), config=config(factual=2)))
    questions = [next(m["content"] for m in row["messages"] if m["role"] == "user") for row in built]
    assert len(built) == len(set(questions)) == 2


def test_claims_queries_and_answers_obey_v4_semantics(tmp_path: Path):
    rows = [
        record(
            "a", "Chiến dịch A", "event",
            "Chiến dịch A bắt đầu năm 1950 do bối cảnh và điều kiện lịch sử. "
            "Chiến dịch A gặp khó khăn, hạn chế và bất lợi. "
            "Kết quả của Chiến dịch A mở ra một giai đoạn mới và có ý nghĩa quan trọng.",
        ),
        record(
            "b", "Chiến dịch B", "event",
            "Chiến dịch B hình thành do bối cảnh và nguyên nhân lịch sử. "
            "Chiến dịch B gặp khó khăn, suy yếu và thất bại. "
            "Kết quả của Chiến dịch B có tác động và ý nghĩa lịch sử.",
        ),
    ]
    corpus = write_corpus(tmp_path / "corpus.jsonl", rows)
    built = list(build_custom_trajectories(
        corpus,
        FixtureRetriever(rows),
        config=config(verification=1, insufficient_evidence=1, multihop=1, hard_negative=1, summary=1),
    ))
    assert validate_rows(built).ok

    verification = next(row for row in built if row["task_type"] == "verification")
    claim = verification["provenance"]["concrete_claim"]
    assert claim and claim in next(m["content"] for m in verification["messages"] if m["role"] == "user")
    insufficient = next(row for row in built if row["task_type"] == "insufficient_evidence")
    assert SYNTHETIC_MARKER in insufficient["provenance"]["synthetic_claim"]
    assert "hỗ trợ trực tiếp" not in insufficient["messages"][-1]["content"]
    assert "[" not in insufficient["messages"][-1]["content"]

    multihop_args = tool_arguments(next(row for row in built if row["task_type"] == "multihop"))
    assert len(multihop_args) == 2
    assert "nguyên nhân" in multihop_args[0]["query"]
    assert "hệ quả" in multihop_args[1]["query"]
    hard_args = tool_arguments(next(row for row in built if row["task_type"] == "hard_negative"))
    assert "thắng lợi" in hard_args[0]["query"]
    assert "thất bại" in hard_args[1]["query"]
    assert len(tool_arguments(next(row for row in built if row["task_type"] == "summary"))) == 2


@pytest.mark.parametrize(
    ("returned", "task_type"),
    [([], "insufficient_evidence"), ([record("one", "Chiến dịch Gốc", "event")], "factual")],
)
def test_requested_top_k_is_preserved_for_empty_and_short_results(
    tmp_path: Path, returned: list[dict], task_type: str,
):
    seeds = [record("seed", "Chiến dịch Gốc", "event")]

    class Retriever:
        def search(self, query, *, top_k):
            assert top_k == 7
            return returned

    corpus = write_corpus(tmp_path / "corpus.jsonl", seeds)
    row = list(build_custom_trajectories(corpus, Retriever(), config=config(**{task_type: 1})))[0]
    assert tool_arguments(row) == [{"query": tool_arguments(row)[0]["query"], "top_k": 7}]


def test_compact_observation_budget_and_citations_are_observed(tmp_path: Path):
    huge = record(
        "huge", "Chiến dịch Dài", "event",
        "Nguyên nhân dẫn đến chiến dịch là điều kiện lịch sử. " + "Thông tin nội bộ không cần thiết. " * 300,
    )
    huge["metadata"]["debug_blob"] = "x" * 20_000
    plan = QueryPlan("search_history", "Chiến dịch Dài nguyên nhân điều kiện", 6, "context_cause")
    compact = compact_observation(
        [huge], plan, task_type="cause", observation_char_budget=500, max_result_text_chars=250,
    )
    assert len(json.dumps(compact, ensure_ascii=False, sort_keys=True)) <= 500
    assert "debug_blob" not in json.dumps(compact)

    corpus = write_corpus(tmp_path / "corpus.jsonl", [huge])
    row = list(build_custom_trajectories(
        corpus,
        FixtureRetriever([huge]),
        config=CustomBuildConfig(task_counts={"cause": 1}, top_k=6, max_corpus_records=1,
                                 observation_char_budget=500, max_result_text_chars=250),
    ))[0]
    payload_ids = {
        item["chunk_id"]
        for message in row["messages"] if message["role"] == "tool"
        for item in json.loads(message["content"])
    }
    assert set(row["provenance"]["evidence_ids"]).issubset(payload_ids)


def test_whole_corpus_sampler_is_deterministic_and_not_prefix_only(tmp_path: Path):
    rows = [record(f"c{i:03d}", f"Chiến dịch {i}", "event") for i in range(80)]
    corpus = write_corpus(tmp_path / "corpus.jsonl", rows)
    first = load_seed_records(corpus, limit=8, seed=17)
    second = load_seed_records(corpus, limit=8, seed=17)
    ids = [row["chunk_id"] for row in first]
    assert ids == [row["chunk_id"] for row in second]
    assert any(int(chunk_id[1:]) >= 8 for chunk_id in ids)
    build_config = CustomBuildConfig(task_counts={"factual": 3}, max_corpus_records=8, seed=17)
    assert list(build_custom_trajectories(corpus, FixtureRetriever(rows), config=build_config)) == list(
        build_custom_trajectories(corpus, FixtureRetriever(rows), config=build_config)
    )


def test_agent_flan_singular_conversation_and_provenance_bloat_fix():
    raw = {
        "id": "agent-1",
        "conversation": [
            {"from": "human", "value": "Xin chào"},
            {"from": "gpt", "value": "Chào bạn"},
        ],
    }
    row = normalize_agent_flan(raw, split="agent_instruct_react")
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
    assert "conversation" not in row["provenance"]["source_metadata"]
    assert row["provenance"]["original_split"] == "agent_instruct_react"
    assert cli.PUBLIC_SOURCE_DEFAULT_SPLITS["agent_flan"] == "agent_instruct_react"


def test_build_all_uses_agent_pool_and_other_source_default_splits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    corpus = write_corpus(tmp_path / "corpus.jsonl", [record("seed", "Chiến dịch Seed", "event")])
    seen: dict[str, str] = {}

    def fake_normalize(args):
        seen[args.source] = args.split or cli.PUBLIC_SOURCE_DEFAULT_SPLITS[args.source]
        canonical = {
            "agent_flan": "agent_flan",
            "multihop": "multi_hop_function_calling",
            "vietnam_history": "vietnam_history_200k",
        }[args.source]
        row = minimal_row(f"public-{args.source}", [f"group-{args.source}"])
        row["source_dataset"] = canonical
        atomic_write_jsonl(args.output, [row])
        return {"written": 1}

    def fake_custom(args):
        output = Path(args.output_dir) / "custom_history.jsonl"
        atomic_write_jsonl(output, [minimal_row("custom", ["group-custom"])])
        return {"written": 1}

    monkeypatch.setattr(cli, "_normalize_public", fake_normalize)
    monkeypatch.setattr(cli, "_build_custom", fake_custom)
    args = cli.build_parser().parse_args([
        "build-all", "--corpus-path", str(corpus), "--output-dir", str(tmp_path / "all"),
        "--retrieval-backend", "fixture",
    ])
    cli._build_all(args)
    assert seen == {
        "agent_flan": "auto",
        "multihop": "train",
        "vietnam_history": "train",
    }


def test_public_max_samples_counts_successful_writes_and_is_attempt_bounded(tmp_path: Path):
    raw = [
        {"id": "bad-1", "conversation": []},
        {"id": "ok-1", "conversation": [{"from": "human", "value": "Q1"}, {"from": "gpt", "value": "A1"}]},
        {"id": "bad-2", "conversation": []},
        {"id": "ok-2", "conversation": [{"from": "human", "value": "Q2"}, {"from": "gpt", "value": "A2"}]},
    ]
    source = write_corpus(tmp_path / "raw.jsonl", raw)
    parser = cli.build_parser()
    args = parser.parse_args([
        "normalize-public", "--source", "agent_flan", "--input-jsonl", str(source),
        "--output", str(tmp_path / "normalized.jsonl"), "--max-samples", "2", "--max-attempts", "4",
    ])
    result = cli._normalize_public(args)
    assert result["attempted"] == 4 and result["written"] == 2 and result["rejected"] == 2
    assert not result["hit_max_attempts"]
    assert len(read_jsonl(args.output)) == 2

    bounded_args = parser.parse_args([
        "normalize-public", "--source", "agent_flan", "--input-jsonl", str(source),
        "--output", str(tmp_path / "bounded.jsonl"), "--max-samples", "2", "--max-attempts", "2",
    ])
    bounded = cli._normalize_public(bounded_args)
    assert bounded["written"] == 1 and bounded["hit_max_attempts"]


def minimal_row(row_id: str, groups: list[str]) -> dict:
    return make_trajectory(
        trajectory_id=row_id,
        source_dataset="custom_history",
        task_type="no_tool",
        messages=[{"role": "user", "content": f"Question {row_id}"}, {"role": "assistant", "content": "Answer"}],
        tools=[],
        difficulty="easy",
        provenance={"source_group": groups[0], "source_groups": groups, "requires_final_answer": True},
    )


def test_multi_source_union_split_prevents_compare_leakage_and_legacy_still_works():
    rows = [minimal_row("ab", ["A", "B"]), minimal_row("b", ["B"]), minimal_row("c", ["C"]), minimal_row("d", ["D"])]
    splits = split_trajectories(rows, train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, seed=3)
    membership = {row["id"]: name for name in ("train", "validation", "test") for row in getattr(splits, name)}
    assert membership["ab"] == membership["b"]
    group_sets = [
        {group for row in getattr(splits, name) for group in source_groups(row)}
        for name in ("train", "validation", "test")
    ]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    legacy = minimal_row("legacy", ["legacy-group"])
    del legacy["provenance"]["source_groups"]
    assert source_groups(legacy) == ["legacy-group"]


def test_rerank_override_is_in_memory_and_does_not_write_config(tmp_path: Path):
    config_path = tmp_path / "inference_config.json"
    config_path.write_text('{"retrieval":{"rerank_batch_size":16}}', encoding="utf-8")
    before = config_path.read_bytes()
    service = SimpleNamespace(config=json.loads(config_path.read_text(encoding="utf-8")))
    apply_rerank_batch_override(service, 4)
    assert service.config["retrieval"]["rerank_batch_size"] == 4
    assert config_path.read_bytes() == before


def _one_cause_row(tmp_path: Path) -> dict:
    rows = [record("cause", "Khởi nghĩa Mẫu", "event", "Khởi nghĩa Mẫu nổ ra do áp bức kéo dài. Cuộc khởi nghĩa có kết quả quan trọng.")]
    corpus = write_corpus(tmp_path / "teacher-corpus.jsonl", rows)
    return list(build_custom_trajectories(corpus, FixtureRetriever(rows), config=config(cause=1)))[0]


def test_teacher_enhancement_is_answer_only_and_rejects_unknown_citations(tmp_path: Path):
    row = _one_cause_row(tmp_path)
    original_messages = copy.deepcopy(row["messages"])
    observed_id = row["provenance"]["observed_evidence_ids"][0]

    class GoodTeacher:
        def generate(self, requests):
            assert requests[0].question == next(m["content"] for m in row["messages"] if m["role"] == "user")
            assert observed_id in requests[0].allowed_evidence_ids
            return [TeacherResponse(answer=f"Câu trả lời chỉ dùng bằng chứng quan sát. [{observed_id}]")]

    enhanced = enhance_rows([row], GoodTeacher(), task_types={"cause"})
    assert enhanced.enhanced == 1
    assert enhanced.rows[0]["messages"][:-1] == original_messages[:-1]
    assert enhanced.rows[0]["messages"][-1]["content"] != original_messages[-1]["content"]

    class BadTeacher:
        def generate(self, requests):
            return [TeacherResponse(answer="Thông tin không hợp lệ. [unknown-id]")]

    fallback = enhance_rows([row], BadTeacher(), task_types={"cause"}, failure_policy="fallback")
    assert fallback.fallback == 1 and fallback.rows[0]["messages"] == original_messages
    rejected = enhance_rows([row], BadTeacher(), task_types={"cause"}, failure_policy="reject")
    assert not rejected.rows and len(rejected.rejected) == 1


def test_build_custom_closes_retriever_before_teacher_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    corpus = write_corpus(tmp_path / "corpus.jsonl", [record("e", "Chiến dịch E", "event")])
    events: list[str] = []

    class Retriever:
        def close(self):
            events.append("retriever_closed")

    class Teacher:
        def __init__(self, *args, **kwargs):
            events.append("teacher_loaded")
            assert events == ["retriever_closed", "teacher_loaded"]

        def generate(self, requests):
            return []

    monkeypatch.setattr(cli, "_make_retriever", lambda args, corpus: Retriever())
    monkeypatch.setattr(cli, "LocalHFTeacher", Teacher)
    monkeypatch.setattr(cli, "build_custom_trajectories", lambda *args, **kwargs: iter(()))
    code = cli.main([
        "build-custom", "--corpus-path", str(corpus), "--output-dir", str(tmp_path / "out"),
        "--teacher-backend", "local_hf", "--teacher-model", "fake", "--no-include-no-tool",
    ])
    assert code == 0 and events == ["retriever_closed", "teacher_loaded"]


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        text = "".join(f"<{m['role']}>{json.dumps(m, ensure_ascii=False)}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_audit_detects_semantic_violations_without_loading_a_model_and_truncation_is_safe():
    invalid = minimal_row("invalid", ["doc"])
    invalid["task_type"] = "cause"
    invalid["provenance"].update({"subject_type": "person", "grounded": True})
    report = audit_rows([invalid], strict_custom=True)
    assert not report["valid"] and report["issues"]["cause_invalid_subject"] == 1

    long_row = make_trajectory(
        trajectory_id="long",
        source_dataset="custom_history",
        task_type="factual",
        tools=[SEARCH_HISTORY_TOOL],
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT + "x" * 200},
            {"role": "user", "content": "Câu hỏi ban đầu"},
            {"role": "assistant", "content": "Câu trả lời"},
        ],
        difficulty="easy",
        provenance={"source_group": "long", "requires_final_answer": True},
    )
    truncation = analyze_truncation(CharacterTokenizer(), long_row, max_length=50)
    assert truncation["initial_user_lost"]
    token_report = tokenizer_audit([long_row], CharacterTokenizer(), max_seq_length=50)
    assert token_report["rows_initial_user_lost"] == 1
    with pytest.raises(ValueError, match="left truncation"):
        build_canonical_sft_example(CharacterTokenizer(), long_row, max_length=50)


def test_tokenizer_free_cli_audit_does_not_construct_teacher_or_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    source = tmp_path / "audit.jsonl"
    atomic_write_jsonl(source, [minimal_row("audit-row", ["audit-group"])])

    def forbidden(*args, **kwargs):
        raise AssertionError("a tokenizer/model constructor must not run")

    monkeypatch.setattr(cli, "LocalHFTeacher", forbidden)
    args = cli.build_parser().parse_args(["audit", "--input", str(source)])
    report = cli._audit_command(args)
    assert report["rows"] == 1 and "tokenizer" not in report
