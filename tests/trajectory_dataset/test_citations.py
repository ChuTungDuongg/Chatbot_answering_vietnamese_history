from __future__ import annotations

import copy

import pytest

from training.trajectory_dataset.audit import audit_rows
from training.trajectory_dataset.builders.custom_history import CustomBuildConfig, build_custom_trajectories
from training.trajectory_dataset.citations import extract_evidence_citations
from training.trajectory_dataset.io_utils import atomic_write_jsonl
from training.trajectory_dataset.retrieval import FixtureRetriever
from training.trajectory_dataset.teacher.base import TeacherResponse
from training.trajectory_dataset.teacher.enhance import enhance_rows
from training.trajectory_dataset.validate import validate_rows


@pytest.mark.parametrize(
    ("answer", "expected_citations", "expected_unknown"),
    [
        ("Thông tin lịch sử [chunk_abc]", ("chunk_abc",), ()),
        ("Sự kiện diễn ra năm 1945 [1]. [chunk_abc]", ("chunk_abc",), ()),
        ("Dẫn chiếu [1] rồi [2], bằng chứng [chunk_abc]", ("chunk_abc",), ()),
        ("Ghi chú [chú thích], bằng chứng [chunk_abc]", ("chunk_abc",), ()),
        ("Trích dẫn sai [chunk_not_observed]", (), ("chunk_not_observed",)),
    ],
)
def test_canonical_citation_parser_distinguishes_evidence_from_incidental_brackets(
    answer: str,
    expected_citations: tuple[str, ...],
    expected_unknown: tuple[str, ...],
):
    parsed = extract_evidence_citations(answer, {"chunk_abc"})
    assert parsed.citations == expected_citations
    assert parsed.unknown_ids == expected_unknown


def _bracketed_source() -> dict:
    return {
        "chunk_id": "chunk_abc",
        "title": "Chiến dịch Mẫu",
        "text": (
            "Chiến dịch Mẫu diễn ra vào năm 1945 [1]. "
            "Sự kiện này có một số kết quả lịch sử quan trọng [chú thích]."
        ),
        "url": "https://example.test/chunk_abc",
        "metadata": {
            "subject_type": "event",
            "content_facets": ["kết quả"],
            "countries": ["Việt Nam"],
        },
    }


def _build_bracketed_row(tmp_path) -> dict:
    source = _bracketed_source()
    corpus = tmp_path / "bracketed-corpus.jsonl"
    atomic_write_jsonl(corpus, [source])
    config = CustomBuildConfig(
        task_counts={"factual": 1},
        top_k=2,
        max_corpus_records=1,
        seed=3,
    )
    return list(build_custom_trajectories(corpus, FixtureRetriever([source]), config=config))[0]


def test_deterministic_builder_accepts_incidental_brackets_and_keeps_grounding(tmp_path):
    row = _build_bracketed_row(tmp_path)
    answer = row["messages"][-1]["content"]

    assert "[1]" in answer
    assert "[chunk_abc]" in answer
    assert row["provenance"]["evidence_ids"] == ["chunk_abc"]
    assert set(row["provenance"]["evidence_ids"]).issubset(row["provenance"]["observed_evidence_ids"])
    assert validate_rows([row]).ok


def test_teacher_accepts_incidental_brackets_with_valid_evidence(tmp_path):
    row = _build_bracketed_row(tmp_path)

    class Teacher:
        def generate(self, requests):
            return [TeacherResponse(
                answer="Sự kiện diễn ra năm 1945 [1], theo ghi chú [chú thích]. [chunk_abc]"
            )]

    result = enhance_rows([row], Teacher(), task_types={"factual"})
    assert result.enhanced == 1 and result.fallback == 0
    assert result.rows[0]["provenance"]["evidence_ids"] == ["chunk_abc"]
    assert validate_rows(result.rows).ok


def test_teacher_unknown_canonical_id_still_falls_back_or_rejects(tmp_path):
    row = _build_bracketed_row(tmp_path)
    original = copy.deepcopy(row)

    class Teacher:
        def generate(self, requests):
            return [TeacherResponse(answer="Sai nguồn. [chunk_not_observed]")]

    fallback = enhance_rows([row], Teacher(), task_types={"factual"}, failure_policy="fallback")
    assert fallback.fallback == 1
    assert fallback.rows[0]["messages"] == original["messages"]
    rejected = enhance_rows([row], Teacher(), task_types={"factual"}, failure_policy="reject")
    assert rejected.rows == [] and len(rejected.rejected) == 1
    assert "chunk_not_observed" in rejected.rejected[0]["reason"]


def test_audit_ignores_incidental_brackets_but_validator_rejects_unknown_id(tmp_path):
    row = _build_bracketed_row(tmp_path)
    report = audit_rows([row], strict_custom=True)
    assert report["issues"].get("grounded_answer_invalid_observed_citations", 0) == 0

    invalid = copy.deepcopy(row)
    invalid["messages"][-1]["content"] = "Sai nguồn. [chunk_not_observed]"
    invalid_report = audit_rows([invalid], strict_custom=True)
    assert invalid_report["issues"]["grounded_answer_invalid_observed_citations"] == 1
    validation = validate_rows([invalid])
    assert not validation.ok
    assert "unknown evidence IDs" in validation.rejected[0]["reason"]
