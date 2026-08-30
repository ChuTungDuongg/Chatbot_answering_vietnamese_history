from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training.trajectory_dataset.audit import audit_rows
from training.trajectory_dataset.builders.custom_history import (
    CustomBuildConfig,
    build_custom_trajectories,
    classify_subject,
    load_seed_records,
    task_eligible,
)
from training.trajectory_dataset.io_utils import atomic_write_jsonl
from training.trajectory_dataset.preprocess import analyze_truncation


def record(chunk_id: str, title: str, subject_type: str | None = "event", text: str | None = None) -> dict:
    metadata = {"content_facets": ["bối cảnh", "kết quả"], "countries": ["Việt Nam"]}
    if subject_type:
        metadata["subject_type"] = subject_type
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text or f"{title} diễn ra trong một bối cảnh lịch sử cụ thể. Sự kiện có kết quả và ý nghĩa quan trọng.",
        "url": f"https://example.test/{chunk_id}",
        "metadata": metadata,
    }


def corpus(tmp_path: Path, rows: list[dict], name: str = "corpus.jsonl") -> Path:
    path = tmp_path / name
    atomic_write_jsonl(path, rows)
    return path


def config(task_type: str, count: int = 1, **overrides) -> CustomBuildConfig:
    values = {
        "task_counts": {task_type: count},
        "top_k": 4,
        "seed": 21,
        "max_corpus_records": 100,
        "observation_char_budget": 12_000,
        "trajectory_observation_char_budget": 6_000,
        "max_result_text_chars": 1_600,
        "max_candidate_attempts_per_task": 100,
    }
    values.update(overrides)
    return CustomBuildConfig(**values)


class FakeRetriever:
    def __init__(self, search):
        self._search = search
        self.calls: list[str] = []

    def search(self, query: str, *, top_k: int) -> list[dict]:
        self.calls.append(query)
        results = []
        for raw in self._search(query)[:top_k]:
            result = copy.deepcopy(raw)
            # Generic fixtures represent relevant hits; bind them explicitly to
            # the requested target/facet so the production filter can stay strict.
            result["text"] = f"{query}. {result.get('text') or ''}".strip()
            results.append(result)
        return results


def evidence(chunk_id: str = "chunk_evidence", title: str = "Bằng chứng", text: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text or "Bối cảnh và nguyên nhân được ghi nhận. Kết quả có ý nghĩa và tác động lịch sử.",
        "source_kind": "history",
    }


def user_question(row: dict) -> str:
    return next(message["content"] for message in row["messages"] if message["role"] == "user")


def tool_payloads(row: dict) -> list[list[dict]]:
    return [json.loads(message["content"]) for message in row["messages"] if message["role"] == "tool"]


def test_subject_classification_rejects_capitalization_false_positive_and_detects_location_suffix():
    script = record(
        "script", "Chữ Quốc ngữ", None,
        "Chữ Quốc ngữ là một hệ thống chữ viết dùng ký tự Latinh cho tiếng Việt.",
    )
    township = record(
        "town", "Lịch Hội Thượng (thị trấn)", None,
        "Lịch Hội Thượng là một thị trấn và địa danh hành chính.",
    )
    former_township = record(
        "former-town", "Lịch Hội Thượng (thị trấn cũ)", None,
        "Lịch Hội Thượng từng là một đơn vị hành chính.",
    )
    assert classify_subject(script) == "topic"
    assert not task_eligible(script, "compare")
    assert classify_subject(township) == "location"
    assert classify_subject(former_township) == "location"
    assert not task_eligible(township, "hard_negative")


def test_person_fallback_requires_biographical_text_evidence():
    person = record(
        "person", "Trần Hưng Đạo", None,
        "Trần Hưng Đạo là danh tướng, có cuộc đời và hoạt động gắn với nhà Trần.",
    )
    title_case_topic = record("topic", "Chữ Quốc Ngữ", None, "Đây là một hệ thống chữ viết.")
    substring_topic = record(
        "substring-topic", "Chữ Quốc Ngữ", None,
        "Hệ thống này không làm thay đổi bản chất của ngôn ngữ nói.",
    )
    assert classify_subject(person) == "person"
    assert classify_subject(title_case_topic) == "topic"
    assert classify_subject(substring_topic) == "topic"


@pytest.mark.parametrize(
    ("subject_type", "title", "required_fragments", "forbidden_fragment"),
    [
        ("person", "Nhân vật Mẫu", ("tiểu sử", "dấu mốc hoạt động", "đóng góp"), "kết quả của Nhân vật"),
        ("event", "Chiến dịch Mẫu", ("bối cảnh", "diễn biến chính", "kết quả", "ý nghĩa"), "tiểu sử"),
        ("organization", "Tổ chức Mẫu", ("sự hình thành", "phát triển", "vai trò", "kết quả"), "tiểu sử"),
    ],
)
def test_summary_questions_are_subject_type_aware(
    tmp_path: Path,
    subject_type: str,
    title: str,
    required_fragments: tuple[str, ...],
    forbidden_fragment: str,
):
    seed = record("seed", title, subject_type)
    retriever = FakeRetriever(lambda query: [evidence(title=title)])
    row = list(build_custom_trajectories(corpus(tmp_path, [seed]), retriever, config=config("summary")))[0]
    question = user_question(row)
    assert all(fragment in question for fragment in required_fragments)
    assert forbidden_fragment not in question
    if subject_type == "person":
        assert "Tiểu sử và các dấu mốc hoạt động" in row["messages"][-1]["content"]
        assert "Vai trò và đóng góp lịch sử" in row["messages"][-1]["content"]


def test_required_evidence_failure_skips_candidate_without_counting_it(tmp_path: Path):
    rows = [record("a", "Chiến dịch Không tìm thấy", "event"), record("b", "Chiến dịch Có bằng chứng", "event")]
    ordered = load_seed_records(corpus(tmp_path, rows), limit=2, seed=21)
    empty_title, good_title = ordered[0]["title"], ordered[1]["title"]

    def search(query: str):
        return [] if empty_title in query else [evidence("chunk_good", good_title)]

    retriever = FakeRetriever(search)
    built = list(build_custom_trajectories(corpus(tmp_path, rows, "second.jsonl"), retriever, config=config("factual")))
    assert len(built) == 1
    assert built[0]["provenance"]["primary_title"] == good_title
    assert any(empty_title in query for query in retriever.calls)


def test_required_evidence_attempt_bound_is_explicit(tmp_path: Path):
    rows = [record("a", "Chiến dịch A", "event"), record("b", "Chiến dịch B", "event")]
    retriever = FakeRetriever(lambda query: [evidence()])
    with pytest.raises(ValueError, match="after 1 candidate attempts"):
        list(build_custom_trajectories(
            corpus(tmp_path, rows), retriever,
            config=config("factual", count=2, max_candidate_attempts_per_task=1),
        ))


def test_summary_uses_one_broad_fallback_for_missing_required_facet(tmp_path: Path):
    seed = record("event", "Chiến dịch Fallback", "event")

    def search(query: str):
        if "kết quả ý nghĩa" in query:
            return []
        return [evidence("chunk_fallback", seed["title"])]

    retriever = FakeRetriever(search)
    row = list(build_custom_trajectories(corpus(tmp_path, [seed]), retriever, config=config("summary")))[0]
    queries = row["provenance"]["retrieval_queries"]
    result_plan = next(item for item in queries if item["role"] == "result_significance")
    assert result_plan["query"] == f"{seed['title']} lịch sử"
    assert result_plan["is_fallback"] is True
    assert all(tool_payloads(row))


def test_compare_requires_compact_evidence_for_both_sides(tmp_path: Path):
    rows = [record("a", "Chiến dịch Alpha", "event"), record("b", "Chiến dịch Beta", "event")]
    one_sided = FakeRetriever(lambda query: [] if "Beta" in query else [evidence("chunk_a", "Alpha")])
    with pytest.raises(ValueError, match="0/1 compare"):
        list(build_custom_trajectories(corpus(tmp_path, rows), one_sided, config=config("compare")))

    balanced = FakeRetriever(
        lambda query: [evidence("chunk_b" if "Beta" in query else "chunk_a", "Beta" if "Beta" in query else "Alpha")]
    )
    built = list(build_custom_trajectories(corpus(tmp_path, rows, "balanced.jsonl"), balanced, config=config("compare")))
    assert len(built) == 1 and all(tool_payloads(built[0]))


def test_multihop_requires_both_roles_even_after_one_broad_fallback(tmp_path: Path):
    seed = record("multi", "Chiến dịch Multi", "event")

    def missing_result(query: str):
        if "hệ quả" in query or query == f"{seed['title']} lịch sử":
            return []
        return [evidence("chunk_context")]

    with pytest.raises(ValueError, match="0/1 multihop"):
        list(build_custom_trajectories(
            corpus(tmp_path, [seed]), FakeRetriever(missing_result), config=config("multihop"),
        ))

    complete = FakeRetriever(lambda query: [evidence("chunk_result" if "hệ quả" in query else "chunk_context")])
    row = list(build_custom_trajectories(
        corpus(tmp_path, [seed], "complete.jsonl"), complete, config=config("multihop"),
    ))[0]
    assert all(tool_payloads(row))


def test_verification_requires_direct_evidence_but_allows_empty_corroboration(tmp_path: Path):
    seed = record("verify", "Chiến dịch Kiểm chứng", "event")

    with pytest.raises(ValueError, match="0/1 verification"):
        list(build_custom_trajectories(
            corpus(tmp_path, [seed]), FakeRetriever(lambda query: []), config=config("verification"),
        ))

    calls = 0

    def direct_only(query: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [evidence("chunk_direct", seed["title"], seed["text"])]
        return []

    row = list(build_custom_trajectories(
        corpus(tmp_path, [seed], "verification-direct.jsonl"), FakeRetriever(direct_only),
        config=config("verification"),
    ))[0]
    assert tool_payloads(row)[0]
    assert tool_payloads(row)[1] == []
    assert row["provenance"]["retrieval_queries"][1]["expected_empty"] is True
    report = audit_rows([row], strict_custom=True)
    assert report["expected_empty_tool_results"] == 1
    assert report["unexpected_empty_tool_results"] == 0
    assert report["valid"]


def test_hard_negative_and_insufficient_have_task_aware_empty_policies(tmp_path: Path):
    seed = record("hard", "Chiến dịch Hard", "event")

    wrong_empty = FakeRetriever(
        lambda query: [] if "thành công thắng lợi" in query else [evidence("chunk_corrective")]
    )
    hard_row = list(build_custom_trajectories(
        corpus(tmp_path, [seed]), wrong_empty, config=config("hard_negative"),
    ))[0]
    assert tool_payloads(hard_row)[0] == [] and tool_payloads(hard_row)[1]

    corrective_empty = FakeRetriever(
        lambda query: [evidence("chunk_wrong")] if "thành công thắng lợi" in query else []
    )
    with pytest.raises(ValueError, match="0/1 hard_negative"):
        list(build_custom_trajectories(
            corpus(tmp_path, [seed], "corrective-empty.jsonl"), corrective_empty,
            config=config("hard_negative"),
        ))

    insufficient = list(build_custom_trajectories(
        corpus(tmp_path, [seed], "insufficient.jsonl"), FakeRetriever(lambda query: []),
        config=config("insufficient_evidence"),
    ))[0]
    assert tool_payloads(insufficient) == [[]]


def _make_task_row(tmp_path: Path, task_type: str, *, budget: int = 1_200) -> dict:
    rows = [record("a", "Chiến dịch A", "event"), record("b", "Chiến dịch B", "event")]
    long_text = (
        "Bối cảnh và nguyên nhân lịch sử được giải thích rõ ràng. "
        "Khó khăn, hạn chế và thất bại cũng được ghi nhận. "
        "Kết quả, hệ quả, vai trò và ý nghĩa có tác động quan trọng. " * 80
    )

    def search(query: str):
        query_key = sum(ord(character) for character in query)
        return [
            evidence(f"chunk_{query_key}_{index}", text=long_text)
            for index in range(4)
        ]

    return list(build_custom_trajectories(
        corpus(tmp_path, rows, f"{task_type}.jsonl"), FakeRetriever(search),
        config=config(
            task_type,
            trajectory_observation_char_budget=budget,
            max_result_text_chars=5_000,
        ),
    ))[0]


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        assert messages
        rendered = "".join(
            f"<{message['role']}>{json.dumps(message, ensure_ascii=False)}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


@pytest.mark.parametrize(
    ("task_type", "expected_calls"),
    [("factual", 1), ("compare", 2), ("multihop", 2), ("hard_negative", 2)],
)
def test_trajectory_budget_is_shared_across_calls_and_preserves_4096_supervision(
    tmp_path: Path, task_type: str, expected_calls: int,
):
    row = _make_task_row(tmp_path, task_type)
    provenance = row["provenance"]
    assert len(provenance["observation_chars"]) == expected_calls
    assert provenance["trajectory_observation_chars"] <= 1_200
    assert all(value <= 1_200 // expected_calls for value in provenance["observation_chars"])
    assert all(tool_payloads(row))
    truncation = analyze_truncation(CharacterTokenizer(), row, max_length=4_096)
    assert not truncation["truncated"]
    assert not truncation["initial_user_lost"]
    assert truncation["lost_tool_call_targets"] == 0
    assert not truncation["final_assistant_lost"]


def _empty_role(row: dict, role: str) -> dict:
    mutated = copy.deepcopy(row)
    index = next(
        index for index, query in enumerate(mutated["provenance"]["retrieval_queries"])
        if query["role"] == role
    )
    tool_messages = [message for message in mutated["messages"] if message["role"] == "tool"]
    tool_messages[index]["content"] = "[]"
    mutated["messages"][-1]["content"] = "Chưa đủ bằng chứng để kết luận."
    mutated["provenance"]["evidence_ids"] = []
    return mutated


def test_role_aware_empty_audit_distinguishes_expected_and_required_calls(tmp_path: Path):
    insufficient_seed = record("ins", "Chiến dịch Ins", "event")
    insufficient = list(build_custom_trajectories(
        corpus(tmp_path, [insufficient_seed], "audit-ins.jsonl"), FakeRetriever(lambda query: []),
        config=config("insufficient_evidence"),
    ))[0]
    hard = _make_task_row(tmp_path, "hard_negative")
    hard_wrong_empty = _empty_role(hard, "wrong_facet")
    factual_required_empty = _empty_role(_make_task_row(tmp_path, "factual"), "factual")
    hard_corrective_empty = _empty_role(hard, "corrective_facet")
    compare_target_empty = _empty_role(_make_task_row(tmp_path, "compare"), "target_b")
    multihop_result_empty = _empty_role(_make_task_row(tmp_path, "multihop"), "result_significance")

    expected_report = audit_rows([insufficient, hard_wrong_empty], strict_custom=True)
    assert expected_report["expected_empty_tool_results"] == 2
    assert expected_report["unexpected_empty_tool_results"] == 0
    assert expected_report["valid"]

    invalid_report = audit_rows(
        [factual_required_empty, hard_corrective_empty, compare_target_empty, multihop_result_empty],
        strict_custom=True,
    )
    assert invalid_report["unexpected_empty_tool_results"] == 4
    assert invalid_report["issues"]["unexpected_empty_tool_results"] == 4
    assert not invalid_report["valid"]
    by_role = invalid_report["empty_tool_results_by_retrieval_role"]
    assert by_role["corrective_facet"]["unexpected_empty_tool_results"] == 1
    assert by_role["target_b"]["unexpected_empty_tool_results"] == 1
    assert by_role["result_significance"]["unexpected_empty_tool_results"] == 1
    assert by_role["corrective_facet"]["empty_tool_result_rate"] == 1.0
    assert invalid_report["empty_tool_results_by_task_type"]["compare"]["empty_tool_result_rate"] == 0.5


def test_evidence_gating_remains_deterministic_and_does_not_duplicate_questions(tmp_path: Path):
    rows = [
        record("a1", "Chiến dịch A", "event"),
        record("a2", "Chiến dịch A", "event"),
        record("b", "Chiến dịch B", "event"),
        record("c", "Chiến dịch C", "event"),
    ]

    def build(name: str):
        retriever = FakeRetriever(lambda query: [] if "Chiến dịch A" in query else [evidence()])
        return list(build_custom_trajectories(corpus(tmp_path, rows, name), retriever, config=config("factual", 2)))

    first, second = build("det-a.jsonl"), build("det-b.jsonl")
    assert first == second
    questions = [user_question(row) for row in first]
    assert len(questions) == len(set(questions)) == 2
