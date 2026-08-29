from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.trajectory_dataset.adapters.agent_flan import normalize_agent_flan
from training.trajectory_dataset.adapters.multihop import normalize_multihop
from training.trajectory_dataset.adapters.common import AdapterError
from training.trajectory_dataset.adapters.vietnam_history import normalize_vietnam_history
from training.trajectory_dataset.builders.custom_history import CustomBuildConfig, build_custom_trajectories
from training.trajectory_dataset.dedup import deduplicate
from training.trajectory_dataset.drive import mount_google_drive, resolve_corpus_path
from training.trajectory_dataset.io_utils import IncrementalJsonlWriter, read_jsonl
from training.trajectory_dataset.mix import mix_sources
from training.trajectory_dataset.preprocess import IGNORE_INDEX, build_canonical_sft_example
from training.trajectory_dataset.retrieval import FixtureRetriever
from training.trajectory_dataset.schema import RETRIEVE_TOOL, SEARCH_HISTORY_TOOL, canonical_id, make_trajectory, tool_call
from training.trajectory_dataset.split import source_group, split_trajectories
from training.trajectory_dataset.stats import dataset_stats
from training.trajectory_dataset.teacher.base import TeacherResponse
from training.trajectory_dataset.validate import validate_rows


FIXTURES = Path(__file__).with_name("fixtures")


def agent_row() -> dict:
    return {
        "id": "agent-1",
        "tools": [{
            "name": "lookup_weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }],
        "messages": [
            {"role": "user", "content": "Weather in Hanoi?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "w1", "function": {"name": "lookup_weather", "arguments": {"city": "Hanoi"}}}]},
            {"role": "tool", "name": "lookup_weather", "tool_call_id": "w1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
    }


def multihop_row() -> dict:
    return {
        "tag": "fixture",
        "multihop": True,
        "messages": [
            {"role": "user", "content": "Where was the author born?"},
            {"role": "assistant", "content": None, "function_call": {"name": "retrieve", "arguments": "{\"query\": \"Who is the author?\"}"}},
            {"role": "function", "name": "retrieve", "content": "The author is A."},
            {"role": "assistant", "content": None, "function_call": {"name": "retrieve", "arguments": "{\"query\": \"Where was A born?\"}"}},
            {"role": "function", "name": "retrieve", "content": "A was born in Hue."},
            {"role": "assistant", "content": "A was born in Hue."},
        ],
    }


def vietnam_row() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý lịch sử Việt Nam."},
            {"role": "user", "content": "Ý nghĩa của chiến thắng Bạch Đằng năm 938 là gì?"},
            {"role": "assistant", "content": "Chiến thắng chấm dứt ách Bắc thuộc kéo dài, khẳng định nền độc lập và mở ra thời đại tự chủ lâu dài của dân tộc Việt Nam."},
        ]
    }


def no_tool(row_id: str = "no-tool", question: str = "Xin chào") -> dict:
    return make_trajectory(
        trajectory_id=row_id,
        source_dataset="fixture",
        task_type="no_tool",
        messages=[{"role": "user", "content": question}, {"role": "assistant", "content": "Xin chào!"}],
        provenance={"requires_final_answer": True, "source_group": row_id},
    )


def one_tool(row_id: str = "one-tool", *, final: bool = True) -> dict:
    call = tool_call("call-1", "search_history", {"query": "Bạch Đằng", "top_k": 1})
    messages = [
        {"role": "user", "content": "Bạch Đằng năm 938 có ý nghĩa gì?"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "name": "search_history", "tool_call_id": "call-1", "content": json.dumps([{"chunk_id": "c1", "text": "Mở ra độc lập."}])},
    ]
    if final:
        messages.append({"role": "assistant", "content": "Mở ra thời đại độc lập. [c1]"})
    return make_trajectory(
        trajectory_id=row_id,
        source_dataset="custom_history",
        task_type="factual",
        tools=[SEARCH_HISTORY_TOOL],
        messages=messages,
        provenance={"requires_final_answer": True, "grounded": True, "evidence_ids": ["c1"], "source_group": "doc-1"},
    )


def test_three_public_adapters_produce_valid_canonical_trajectories():
    rows = [
        normalize_agent_flan(agent_row()),
        normalize_multihop(multihop_row()),
        normalize_vietnam_history(vietnam_row()),
    ]
    result = validate_rows(rows)
    assert result.ok
    assert rows[0]["source_dataset"] == "agent_flan"
    assert rows[1]["task_type"] == "multi_hop_retrieval"
    assert rows[2]["uses_tools"] is False


def test_public_generic_tool_is_not_renamed_search_history():
    row = normalize_multihop(multihop_row())
    assert {tool["function"]["name"] for tool in row["tools"]} == {"retrieve"}
    assert {call["function"]["name"] for message in row["messages"] for call in message.get("tool_calls", [])} == {"retrieve"}


def test_vietnam_history_quality_and_task_filters_reject_observable_bad_rows():
    malformed = vietnam_row()
    malformed["messages"][-1]["content"] = "Quá ngắn."
    with pytest.raises(AdapterError, match="answer too short"):
        normalize_vietnam_history(malformed)
    non_analytical = vietnam_row()
    non_analytical["messages"][1]["content"] = "Sự kiện diễn ra năm nào?"
    with pytest.raises(AdapterError, match="preferred analytical"):
        normalize_vietnam_history(non_analytical, preferred_only=True)


def test_no_tool_one_tool_and_multi_tool_validation():
    multi = normalize_multihop(multihop_row())
    assert validate_rows([no_tool(), one_tool(), multi]).ok


def test_malformed_tool_call_id_and_missing_final_are_rejected_with_reason():
    malformed = one_tool("bad-id")
    malformed["messages"][2]["tool_call_id"] = "wrong"
    result = validate_rows([malformed, one_tool("missing-final", final=False)])
    assert len(result.rejected) == 2
    assert "unknown tool_call_id" in result.rejected[0]["reason"]
    assert "missing final assistant answer" in result.rejected[1]["reason"]


def test_duplicate_ids_and_normalized_exact_question_dedup():
    duplicate_ids = validate_rows([no_tool("same", "A"), no_tool("same", "B")])
    assert "duplicate trajectory id" in duplicate_ids.rejected[0]["reason"]
    result = deduplicate([no_tool("one", "  Xin, CHÀO! "), no_tool("two", "xin chào")])
    assert len(result.rows) == 1
    assert result.rejected[0]["reason"] == "duplicate normalized user question"


def test_split_is_deterministic_and_source_group_safe():
    rows = []
    for group in range(10):
        for variant in range(2):
            row = no_tool(f"{group}-{variant}", f"Question {group} variant {variant}")
            row["provenance"]["source_group"] = f"doc-{group}"
            rows.append(row)
    first = split_trajectories(rows, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, seed=7)
    second = split_trajectories(rows, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, seed=7)
    assert [[row["id"] for row in getattr(first, name)] for name in ("train", "validation", "test")] == [
        [row["id"] for row in getattr(second, name)] for name in ("train", "validation", "test")
    ]
    groups = [{source_group(row) for row in getattr(first, name)} for name in ("train", "validation", "test")]
    assert all(groups)
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])


def test_official_test_split_is_preserved():
    rows = [no_tool(f"row-{index}", f"q {index}") for index in range(5)]
    rows[0]["provenance"]["original_split"] = "test"
    split = split_trajectories(rows, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, seed=1)
    assert rows[0] in split.test


def test_mix_ratios_and_stats_are_correct():
    sources = {
        "a": [no_tool(f"a-{index}", f"a question {index}") for index in range(80)],
        "b": [no_tool(f"b-{index}", f"b question {index}") for index in range(20)],
    }
    mixed = mix_sources(sources, {"a": 0.8, "b": 0.2}, seed=4, max_total=50)
    counts = {name: sum(row["id"].startswith(name) for row in mixed) for name in ("a", "b")}
    assert counts == {"a": 40, "b": 10}
    stats = dataset_stats([*mixed, one_tool("tool-stats")])
    assert stats["total_rows"] == 51
    assert stats["tool_usage"]["with_tools"] == 1
    assert stats["trajectory_turn_count"]["max"] == 4


def test_local_and_directory_corpus_path_resolution(tmp_path: Path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    corpus = directory / "vn_history_rag_chunks_enriched.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    assert resolve_corpus_path(corpus) == corpus.resolve()
    assert resolve_corpus_path(directory) == corpus.resolve()


def test_mount_drive_outside_colab_is_actionable(monkeypatch: pytest.MonkeyPatch):
    def fail(_: str):
        raise ImportError("not colab")

    monkeypatch.setattr("training.trajectory_dataset.drive.importlib.import_module", fail)
    with pytest.raises(RuntimeError, match="outside Google Colab"):
        mount_google_drive("/content/drive")


def test_fake_drive_mount_and_drive_corpus_resolution(tmp_path: Path):
    calls: list[str] = []
    module = SimpleNamespace(drive=SimpleNamespace(mount=lambda point: calls.append(point)))
    corpus = tmp_path / "drive" / "vn_history_rag_chunks_enriched.jsonl"
    corpus.parent.mkdir()
    corpus.write_text("{}\n", encoding="utf-8")
    resolved = resolve_corpus_path(
        drive_corpus_path=corpus,
        mount_drive=True,
        drive_mount_point=tmp_path / "drive",
        colab_module=module,
    )
    assert resolved == corpus.resolve()
    assert calls == [str(tmp_path / "drive")]


def test_incremental_resume_skips_completed_records(tmp_path: Path):
    path = tmp_path / "resume.jsonl"
    with IncrementalJsonlWriter(path, resume=False, checkpoint_every=1) as writer:
        assert writer.write(no_tool("a"))
    with IncrementalJsonlWriter(path, resume=True, checkpoint_every=1) as writer:
        assert not writer.write(no_tool("a"))
        assert writer.write(no_tool("b", "hello b"))
        assert writer.skipped == 1
    assert [row["id"] for row in read_jsonl(path)] == ["a", "b"]


def test_custom_builder_reads_fixture_without_mutating_and_uses_actual_tool_schema():
    corpus = FIXTURES / "fake_corpus.jsonl"
    before = corpus.read_bytes()
    records = read_jsonl(corpus)
    config = CustomBuildConfig(
        task_counts={"factual": 1, "cause": 1, "significance": 1, "compare": 1, "summary": 0, "multihop": 0, "verification": 0, "hard_negative": 0},
        top_k=2,
        max_corpus_records=3,
        seed=2,
    )
    rows = list(build_custom_trajectories(corpus, FixtureRetriever(records), config=config))
    assert corpus.read_bytes() == before
    assert len(rows) == 4
    assert validate_rows(rows).ok
    assert all(row["tools"] == [SEARCH_HISTORY_TOOL] for row in rows)
    assert all(row["provenance"]["corpus_read_only"] for row in rows)


def test_custom_insufficient_and_external_verification_behaviors_are_valid():
    corpus = FIXTURES / "fake_corpus.jsonl"
    records = read_jsonl(corpus)

    class External:
        def search(self, tool, query, *, top_k):
            assert tool == "search_wikipedia"
            return [{"chunk_id": "wiki-1", "title": "Đối chiếu", "text": "Nguồn đối chiếu lịch sử."}]

    config = CustomBuildConfig(
        task_counts={"verification": 1, "insufficient_evidence": 1},
        top_k=2,
        max_corpus_records=3,
        seed=4,
    )
    rows = list(
        build_custom_trajectories(
            corpus,
            FixtureRetriever(records),
            config=config,
            external_retriever=External(),
        )
    )
    assert validate_rows(rows).ok
    verification = next(row for row in rows if row["task_type"] == "verification")
    assert {tool["function"]["name"] for tool in verification["tools"]} == {"search_history", "search_wikipedia"}
    insufficient = next(row for row in rows if row["task_type"] == "insufficient_evidence")
    assert "chưa đủ bằng chứng" in insufficient["messages"][-1]["content"]
    assert insufficient["provenance"]["evidence_ids"] == []


def test_mock_teacher_receives_real_batches():
    corpus = FIXTURES / "fake_corpus.jsonl"
    records = read_jsonl(corpus)

    class Teacher:
        def __init__(self):
            self.batch_sizes = []

        def generate(self, requests):
            self.batch_sizes.append(len(requests))
            return [TeacherResponse(question=f"Câu hỏi {index}?", answer="Câu trả lời dựa trên bằng chứng đã cho.") for index, _ in enumerate(requests)]

    teacher = Teacher()
    config = CustomBuildConfig(
        task_counts={"factual": 2, "summary": 2},
        top_k=2,
        max_corpus_records=3,
        seed=8,
    )
    rows = list(build_custom_trajectories(corpus, FixtureRetriever(records), config=config, teacher=teacher))
    assert len(rows) == 4
    assert teacher.batch_sizes == [2, 2]
    assert validate_rows(rows).ok


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        assert tokenize is False
        parts = []
        for message in messages:
            role = message["role"]
            payload = json.dumps(
                {key: value for key, value in message.items() if key != "role"},
                ensure_ascii=False,
                sort_keys=True,
            )
            parts.append(f"<{role}>{payload}</{role}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def test_chat_template_masking_trains_calls_and_answers_but_not_tool_observations():
    row = one_tool()
    tokenizer = CharacterTokenizer()
    feature = build_canonical_sft_example(tokenizer, row, max_length=100_000)
    rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, tools=row["tools"])
    labels = feature["labels"]
    observation_start = rendered.index("<tool>")
    observation_end = rendered.index("</tool>") + len("</tool>")
    assert all(label == IGNORE_INDEX for label in labels[observation_start:observation_end])
    call_start = rendered.index("tool_calls")
    assert any(label != IGNORE_INDEX for label in labels[call_start : call_start + 100])
    answer_start = rendered.index("Mở ra thời đại")
    assert all(label != IGNORE_INDEX for label in labels[answer_start : answer_start + len("Mở ra thời đại")])


def test_cli_help_and_dry_run_do_not_load_heavy_components(tmp_path: Path):
    help_result = subprocess.run(
        [sys.executable, "-m", "training.trajectory_dataset.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "build-all" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.trajectory_dataset.cli",
            "build-custom",
            "--corpus-path",
            str(FIXTURES / "fake_corpus.jsonl"),
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["no_model_or_retriever_loaded"] is True
    assert not list(tmp_path.iterdir())


def test_imports_have_no_network_or_model_side_effects():
    modules = [
        "training.trajectory_dataset",
        "training.trajectory_dataset.cli",
        "training.trajectory_dataset.adapters.agent_flan",
        "training.trajectory_dataset.builders.custom_history",
    ]
    for module in modules:
        assert importlib.import_module(module)
