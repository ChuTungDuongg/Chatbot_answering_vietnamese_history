from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.trajectory_dataset import audit as audit_module
from training.trajectory_dataset import cli
from training.trajectory_dataset.audit import audit_rows, tokenizer_audit
from training.trajectory_dataset.builders.custom_history import (
    CustomBuildConfig,
    build_custom_trajectories,
    canonical_subject_identity,
    classify_subject,
)
from training.trajectory_dataset.io_utils import atomic_write_jsonl, read_jsonl
from training.trajectory_dataset.preprocess import IGNORE_INDEX, analyze_truncation, build_canonical_sft_example
from training.trajectory_dataset.schema import SEARCH_HISTORY_TOOL, make_trajectory, tool_call


def identity_record(chunk_id: str, title: str, text: str, declared: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text,
        "subject_type": declared,
        "history_score": 25,
        "url": f"https://example.test/{chunk_id}",
        "metadata": {
            "events": ["Biến cố lịch sử"],
            "locations": ["Việt Nam"],
            "people": ["Một nhân vật được nhắc đến"],
            "years": [1945],
        },
    }


REAL_IDENTITIES = (
    identity_record(
        "thein-sein", "Thein Sein",
        "Thein Sein (sinh ngày 20 tháng 4 năm 1945) là Tổng thống dân cử của Myanmar. "
        "Ông là một chính trị gia, từng giữ chức Thủ tướng và tiến hành cải cách lịch sử.",
        "state",
    ),
    identity_record(
        "dang-huu", "Đặng Hữu",
        "Đặng Hữu (sinh ngày 2 tháng 1 năm 1930) là một giáo sư, viện sĩ và học giả Việt Nam. "
        "Quá trình công tác của ông gắn với nhiều chính sách khoa học trong lịch sử hiện đại.",
        "organization",
    ),
    identity_record(
        "kim-ha", "Nguyễn Thị Kim Hà",
        "Nguyễn Thị Kim Hà (sinh ngày 28 tháng 2 năm 2000) là một vận động viên Taekwondo Việt Nam. "
        "Tiểu sử và sự nghiệp của cô ghi nhận nhiều giải đấu.",
        "state",
    ),
    identity_record(
        "tu-ma-thien", "Tư Mã Thiên",
        "Tư Mã Thiên là một sử gia thời Hán. Cha của Tư Mã Thiên là Tư Mã Đàm. "
        "Tư Mã Thiên là người huyện Hạ Dương. "
        "Tư Mã Thiên hoạt động biên soạn sử và có đóng góp quan trọng. "
        "Theo các nguồn, ông sinh vào khoảng năm 145 TCN; từ nhỏ ông đã học sử.",
        "state",
    ),
    identity_record(
        "fiji", "Fiji",
        "Fiji trải qua nhiều cuộc đảo chính trong lịch sử. Tổng thống Fiji có nhiệm kỳ năm năm; "
        "Quốc hội, Chính phủ và Hiến pháp Fiji quy định thể chế của quốc gia này. Đối ngoại Fiji thay đổi sau năm 2006.",
        "person",
    ),
    identity_record(
        "con-dao", "Côn Đảo",
        "Hiện nay Côn Đảo là một huyện thuộc tỉnh Bà Rịa – Vũng Tàu. Địa lý tự nhiên của quần đảo "
        "Côn Đảo gồm nhiều đảo, có diện tích và ranh giới hành chính xác định.",
        "state",
    ),
)


@pytest.mark.parametrize(
    ("record", "expected"),
    list(zip(REAL_IDENTITIES, ("person", "person", "person", "person", "state", "location"))),
)
def test_real_subject_identity_examples_use_target_linked_semantics(record: dict, expected: str):
    canonical = canonical_subject_identity(record)
    assert canonical is not None
    assert classify_subject(record) == classify_subject(canonical) == expected


def test_canonical_identity_keeps_late_target_evidence_and_is_idempotent():
    record = identity_record(
        "late-person", "Thein Sein",
        ("Một đoạn ngữ cảnh dài không xác định chủ thể. " * 120)
        + "Thein Sein (sinh ngày 20 tháng 4 năm 1945) là Tổng thống và một chính trị gia Myanmar.",
        "state",
    )
    canonical = canonical_subject_identity(record)
    assert canonical is not None and len(canonical["text"]) <= 3_200
    assert "Thein Sein" in canonical["text"]
    assert canonical_subject_identity(canonical) == canonical
    assert classify_subject(record) == classify_subject(canonical) == "person"


class TargetRetriever:
    def __init__(self, records: list[dict]):
        self.records = records
        self.calls = 0

    def search(self, query: str, *, top_k: int) -> list[dict]:
        self.calls += 1
        matches = [record for record in self.records if record["title"].casefold() in query.casefold()]
        return matches[:top_k]


def test_full_builder_persists_canonical_identity_and_state_question_template(tmp_path: Path):
    records = [REAL_IDENTITIES[index] for index in (0, 1, 3, 4)]
    corpus = tmp_path / "identity-corpus.jsonl"
    atomic_write_jsonl(corpus, records)
    rows = list(build_custom_trajectories(
        corpus,
        TargetRetriever(records),
        config=CustomBuildConfig(task_counts={"factual": 4}, max_corpus_records=4, seed=31),
    ))
    assert len(rows) == 4
    by_title = {row["provenance"]["primary_title"]: row for row in rows}
    assert by_title["Thein Sein"]["provenance"]["subject_type"] == "person"
    assert by_title["Fiji"]["provenance"]["subject_type"] == "state"
    fiji_question = next(message["content"] for message in by_title["Fiji"]["messages"] if message["role"] == "user")
    assert all(fragment not in fiji_question.casefold() for fragment in ("là ai", "cuộc đời", "nhân vật này"))
    for row in rows:
        provenance = row["provenance"]
        assert classify_subject(provenance["primary_subject_identity"]) == provenance["subject_type"]
    report = audit_rows(rows, strict_custom=True)
    assert report["issues"].get("subject_type_mismatch", 0) == 0


def test_compare_builder_and_audit_use_the_same_persisted_identity_contract(tmp_path: Path):
    an_nam = identity_record(
        "an-nam", "An Nam",
        "Trong lịch sử cận đại, Annam là một quốc gia có bộ máy chính quyền, quốc kỳ và quốc ca. "
        "Bối cảnh thuộc địa của An Nam ảnh hưởng đến diễn biến và kết quả lịch sử của quốc gia này.",
        "topic",
    )
    records = [REAL_IDENTITIES[0], REAL_IDENTITIES[3], REAL_IDENTITIES[4], an_nam]
    corpus = tmp_path / "compare-corpus.jsonl"
    atomic_write_jsonl(corpus, records)
    rows = list(build_custom_trajectories(
        corpus,
        TargetRetriever(records),
        config=CustomBuildConfig(task_counts={"compare": 2}, max_corpus_records=4, seed=17),
    ))
    assert len(rows) == 2
    for row in rows:
        provenance = row["provenance"]
        assert classify_subject(provenance["primary_subject_identity"]) == provenance["subject_type"]
        assert classify_subject(provenance["secondary_subject_identity"]) == provenance["secondary_subject_type"]
        assert provenance["subject_type"] == provenance["secondary_subject_type"]
    report = audit_rows(rows, strict_custom=True)
    assert report["issues"].get("subject_type_mismatch", 0) == 0
    assert report["issues"].get("compare_type_mismatch", 0) == 0


def test_custom_builder_yields_before_whole_task_quota_finishes(tmp_path: Path):
    records = [
        identity_record("state-a", "Quốc gia A", "Quốc gia A là một quốc gia hình thành năm 1945 và có lịch sử lâu dài.", "state"),
        identity_record("state-b", "Quốc gia B", "Quốc gia B là một quốc gia hình thành năm 1946 và có lịch sử lâu dài.", "state"),
    ]
    corpus = tmp_path / "incremental.jsonl"
    atomic_write_jsonl(corpus, records)
    retriever = TargetRetriever(records)
    generator = iter(build_custom_trajectories(
        corpus,
        retriever,
        config=CustomBuildConfig(task_counts={"factual": 2}, max_corpus_records=2, seed=5),
    ))
    first = next(generator)
    assert first["task_type"] == "factual" and retriever.calls == 1
    assert len(list(generator)) == 1 and retriever.calls == 2


class PrefixStableTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        assert tokenize is False
        if add_generation_prompt:
            raise AssertionError("canonical span discovery must not depend on a generation prompt")
        return "".join(
            f"<{message['role']}>"
            + json.dumps({key: value for key, value in message.items() if key != "role"}, ensure_ascii=False, sort_keys=True)
            + f"</{message['role']}>"
            for message in messages
        )

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def canonical_tool_row() -> dict:
    call = tool_call("call_1", "search_history", {"query": "Nhà Mạc", "top_k": 6})
    return make_trajectory(
        trajectory_id="canonical-tool-row",
        source_dataset="custom_history",
        task_type="factual",
        tools=[SEARCH_HISTORY_TOOL],
        messages=[
            {"role": "system", "content": "Bạn là trợ lý lịch sử."},
            {"role": "user", "content": "Nhà Mạc có lịch sử ra sao?"},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "name": "search_history", "tool_call_id": "call_1", "content": "[]"},
            {"role": "assistant", "content": "Chưa đủ bằng chứng."},
        ],
        provenance={"source_group": "canonical", "requires_final_answer": True},
    )


def test_canonical_preprocessing_supervises_tool_call_and_final_but_not_user_or_tool():
    row = canonical_tool_row()
    tokenizer = PrefixStableTokenizer()
    feature = build_canonical_sft_example(tokenizer, row, max_length=100_000)
    report = analyze_truncation(tokenizer, row, max_length=100_000)
    rendered = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, tools=row["tools"])
    labels = feature["labels"]
    for marker in ("Nhà Mạc có lịch sử ra sao?", "<tool>"):
        start = rendered.index(marker)
        assert all(label == IGNORE_INDEX for label in labels[start : start + len(marker)])
    for marker in ("tool_calls", "Chưa đủ bằng chứng."):
        start = rendered.index(marker)
        assert any(label != IGNORE_INDEX for label in labels[start : start + len(marker)])
    assert len(report["assistant_spans"]) == 2
    assert all(span["end"] > span["start"] for span in report["assistant_spans"])


def test_tokenizer_audit_exposes_exception_reasons(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        raise ValueError("chat template exposed no assistant target at message 2")

    monkeypatch.setattr(audit_module, "build_canonical_sft_example", fail)
    report = tokenizer_audit([canonical_tool_row()], PrefixStableTokenizer(), max_seq_length=4_096)
    reason = "ValueError: chat template exposed no assistant target at message 2"
    assert report["preprocessing_errors"] == 1
    assert report["preprocessing_error_counts"] == {reason: 1}
    assert report["preprocessing_error_row_ids"] == ["canonical-tool-row"]
    assert report["preprocessing_error_examples"] == [{
        "row_id": "canonical-tool-row",
        "stage": "build_canonical_sft_example",
        "error": reason,
    }]


def progress_row(row_id: str, task_type: str) -> dict:
    return make_trajectory(
        trajectory_id=row_id,
        source_dataset="custom_history",
        task_type=task_type,
        messages=[
            {"role": "user", "content": f"Câu hỏi {row_id}?"},
            {"role": "assistant", "content": "Câu trả lời."},
        ],
        provenance={"source_group": row_id, "requires_final_answer": True},
    )


def progress_args(corpus: Path, output_dir: Path, *, progress_every: int, resume: bool = False):
    values = [
        "build-custom", "--corpus-path", str(corpus), "--output-dir", str(output_dir),
        "--no-include-no-tool", "--progress-every", str(progress_every),
        "--num-factual", "1", "--num-cause", "1",
        "--num-significance", "0", "--num-compare", "0", "--num-summary", "0",
        "--num-multihop", "0", "--num-verification", "0", "--num-hard-negative", "0",
        "--num-insufficient-evidence", "0",
    ]
    if resume:
        values.append("--resume")
    return cli.build_parser().parse_args(values)


def _progress_payloads(output: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("CUSTOM_PROGRESS "))
        for line in output.splitlines()
        if line.startswith("CUSTOM_PROGRESS ")
    ]


def test_custom_progress_counts_only_valid_written_rows_and_emits_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    corpus = tmp_path / "corpus.jsonl"
    atomic_write_jsonl(corpus, [identity_record("seed", "Quốc gia Mẫu", "Quốc gia Mẫu là một quốc gia lịch sử.", "state")])
    invalid = progress_row("invalid", "factual")
    invalid["messages"] = []
    accepted = [progress_row("factual-1", "factual"), progress_row("cause-1", "cause")]
    monkeypatch.setattr(cli, "_make_retriever", lambda args, path: object())
    monkeypatch.setattr(cli, "build_custom_trajectories", lambda *args, **kwargs: iter([invalid, *accepted]))
    cli._build_custom(progress_args(corpus, tmp_path / "out", progress_every=1))
    payloads = _progress_payloads(capsys.readouterr().out)
    assert [payload["written"] for payload in payloads] == [1, 2, 2]
    assert payloads[0]["task_type"] == "factual" and payloads[0]["task_written"] == 1
    assert payloads[1]["task_type"] == "cause" and payloads[1]["task_written"] == 1
    assert payloads[-1]["target"] == 2 and payloads[-1]["percent"] == 100.0
    assert [row["id"] for row in read_jsonl(tmp_path / "out" / "custom_history.jsonl")] == ["factual-1", "cause-1"]


def test_custom_progress_is_resume_aware_and_does_not_change_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    corpus = tmp_path / "corpus.jsonl"
    atomic_write_jsonl(corpus, [identity_record("seed", "Quốc gia Mẫu", "Quốc gia Mẫu là một quốc gia lịch sử.", "state")])
    rows = [progress_row("factual-1", "factual"), progress_row("cause-1", "cause")]
    monkeypatch.setattr(cli, "_make_retriever", lambda args, path: object())

    def fake_builder(*args, completed_ids=None, **kwargs):
        return iter(row for row in rows if row["id"] not in (completed_ids or set()))

    monkeypatch.setattr(cli, "build_custom_trajectories", fake_builder)
    plain_dir = tmp_path / "plain"
    cli._build_custom(progress_args(corpus, plain_dir, progress_every=0))
    assert not _progress_payloads(capsys.readouterr().out)

    progress_dir = tmp_path / "progress"
    atomic_write_jsonl(progress_dir / "custom_history.jsonl", [rows[0]])
    cli._build_custom(progress_args(corpus, progress_dir, progress_every=1, resume=True))
    payloads = _progress_payloads(capsys.readouterr().out)
    assert payloads[0]["written"] == 2
    assert payloads[0]["task_type"] == "cause" and payloads[0]["task_written"] == 1
    assert payloads[-1]["written"] == 2 and payloads[-1]["target"] == 2
    assert read_jsonl(plain_dir / "custom_history.jsonl") == read_jsonl(progress_dir / "custom_history.jsonl")
