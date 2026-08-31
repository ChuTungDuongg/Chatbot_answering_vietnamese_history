from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.trajectory_dataset.adapters.agent_flan import normalize_agent_flan
from training.trajectory_dataset.adapters.common import AdapterError
from training.trajectory_dataset.adapters.vietnam_history import normalize_vietnam_history
from training.trajectory_dataset.audit import audit_rows
from training.trajectory_dataset.builders.custom_history import (
    CustomBuildConfig,
    build_custom_trajectories,
    classify_subject,
)
from training.trajectory_dataset.final_gate import final_dataset_gate
from training.trajectory_dataset.io_utils import atomic_write_jsonl
from training.trajectory_dataset.mix import DEFAULT_MIX_RATIOS, mix_capacity_report
from training.trajectory_dataset.schema import RETRIEVE_TOOL, make_trajectory, tool_call
from training.trajectory_dataset.split import (
    IMPORTANT_CUSTOM_TASK_TYPES,
    TrajectorySplits,
    split_coverage_report,
    split_trajectories,
)
from training.trajectory_dataset.validate import validate_rows


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, tools=None):
        assert tokenize is False
        rendered = []
        if tools:
            rendered.append(json.dumps(tools, ensure_ascii=False, sort_keys=True))
        for message in messages:
            rendered.append(
                f"<{message['role']}>"
                + json.dumps({key: value for key, value in message.items() if key != "role"}, ensure_ascii=False, sort_keys=True)
                + f"</{message['role']}>"
            )
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _write_corpus(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "corpus.jsonl"
    atomic_write_jsonl(path, rows)
    return path


def _simple_row(row_id: str, *, source: str = "fixture", task: str = "no_tool", group: str | None = None) -> dict:
    return make_trajectory(
        trajectory_id=row_id,
        source_dataset=source,
        task_type=task,
        messages=[
            {"role": "user", "content": f"Câu hỏi {row_id}?"},
            {"role": "assistant", "content": f"Câu trả lời {row_id}."},
        ],
        provenance={"requires_final_answer": True, "source_group": group or row_id},
    )


@pytest.mark.parametrize("analysis_text", [
    "Bước 1: xác định dữ kiện. Bước 2: soạn câu trả lời.",
    "Để trả lời, mình sẽ tổng hợp thông tin cốt lõi trước.",
])
def test_vietnam_history_drops_analysis_channel_and_preserves_exact_final(analysis_text: str):
    final = (
        "Chiến thắng Rạch Gầm – Xoài Mút năm 1785 đã đánh bại quân Xiêm, "
        "bảo vệ vùng đất phía Nam và khẳng định tài chỉ huy của Nguyễn Huệ."
    )
    raw = {
        "id": "vh-analysis",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý lịch sử."},
            {"role": "user", "content": "Ý nghĩa trận Rạch Gầm – Xoài Mút là gì?"},
            {"role": "assistant", "channel": "analysis", "content": analysis_text},
            {"role": "assistant", "channel": "final", "content": final},
        ],
    }
    row = normalize_vietnam_history(raw)
    assert [message["role"] for message in row["messages"]] == ["system", "user", "assistant"]
    assert row["messages"][-1]["content"] == final
    assert all(analysis_text not in str(message.get("content") or "") for message in row["messages"])
    assert row["provenance"]["dropped_assistant_analysis_messages"] == 1
    assert "dropped_assistant_analysis_channel" in row["provenance"]["transformations"]
    assert validate_rows([row]).ok
    assert audit_rows([row], strict_custom=True)["issues"].get(
        "vietnam_history_analysis_message_remaining", 0,
    ) == 0


def test_vietnam_history_legacy_two_assistant_shape_is_a_hard_failure_without_text_guessing():
    row = _simple_row("legacy-vietnam", source="vietnam_history_200k", task="vietnamese_history_style")
    row["messages"].insert(1, {"role": "assistant", "content": "Arbitrary planning content."})
    validation = validate_rows([row])
    assert not validation.ok
    assert "analysis channel remains" in validation.rejected[0]["reason"]
    report = audit_rows([row], strict_custom=True)
    assert report["issues"]["vietnam_history_analysis_message_remaining"] == 1


def test_agent_flan_relations_neighbors_react_converts_to_canonical_tools_without_thought():
    raw = {
        "id": "relations",
        "messages": [
            {"role": "user", "content": "Which answer is correct?"},
            {"role": "assistant", "content": "Action: get_relations(Frank Miller)"},
            {"role": "user", "content": "[directed_by]"},
            {"role": "assistant", "content": "Action: get_neighbors(Frank Miller, directed_by)"},
            {"role": "user", "content": "[Sin City]"},
            {"role": "assistant", "content": "Thought: Variable #2 is the answer.\nFinal Answer: #2"},
        ],
    }
    row = normalize_agent_flan(raw)
    calls = [call for message in row["messages"] for call in message.get("tool_calls", [])]
    assert [call["function"]["name"] for call in calls] == ["get_relations", "get_neighbors"]
    assert {tool["function"]["name"] for tool in row["tools"]} == {"get_relations", "get_neighbors"}
    assert row["messages"][-1] == {"role": "assistant", "content": "#2"}
    assert all("Thought:" not in str(message.get("content") or "") for message in row["messages"])
    assert row["uses_tools"] is True and row["task_type"] == "generic_agent_tool_behavior"
    assert validate_rows([row]).ok


def test_agent_flan_search_click_converts_and_unsafe_environment_action_is_rejected():
    safe = {
        "id": "browser",
        "messages": [
            {"role": "user", "content": "Find the requested page."},
            {"role": "assistant", "content": (
                "Action: search[Vietnam history]\nObservation: [1, 2]\n"
                "Action: click[2]\nObservation: The requested page.\nFinal Answer: The requested page."
            )},
        ],
    }
    row = normalize_agent_flan(safe)
    calls = [call for message in row["messages"] for call in message.get("tool_calls", [])]
    assert [call["function"]["name"] for call in calls] == ["search", "click"]
    assert calls[0]["function"]["arguments"] == {"query": "Vietnam history"}
    assert calls[1]["function"]["arguments"] == {"id": 2}
    assert validate_rows([row]).ok

    unsafe = {
        "id": "environment",
        "messages": [
            {"role": "user", "content": "Enter the room."},
            {"role": "assistant", "content": (
                "Action: go to kitchen\nObservation: You are in the kitchen.\nFinal Answer: done"
            )},
        ],
    }
    with pytest.raises(AdapterError, match="unsafe Agent-FLAN action syntax"):
        normalize_agent_flan(unsafe)


def test_agent_flan_structured_terminal_thought_keeps_only_final_answer():
    raw = {
        "id": "structured-final",
        "tools": [{
            "name": "lookup", "description": "lookup",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }],
        "messages": [
            {"role": "user", "content": "Lookup."},
            {"role": "assistant", "function_call": {"name": "lookup", "arguments": {}}},
            {"role": "observation", "content": "#3"},
            {"role": "assistant", "content": "Thought: Variable #3 is correct.\nFinal Answer: #3"},
        ],
    }
    row = normalize_agent_flan(raw, include_reasoning=True)
    assert row["messages"][-1]["content"] == "#3"
    assert audit_rows([row], strict_custom=True)["issues"].get("agent_flan_thought_target", 0) == 0


def test_kop_full_builder_uses_band_identity_and_rejects_incidental_acronym_collision(tmp_path: Path):
    kop = {
        "chunk_id": "kop-band", "title": "KOP", "history_score": 30,
        "text": (
            "Ban nhạc KOP là một ban nhạc rock Việt Nam và là một tổ chức âm nhạc. "
            "KOP được thành lập năm 2011, hoạt động qua nhiều giai đoạn và có vai trò trong dòng nhạc rock."
        ),
        "metadata": {"organizations": ["KOP"], "years": [2011]},
        "url": "https://example.test/kop-band",
    }
    collision = {
        "chunk_id": "raginis", "title": "Władysław Raginis",
        "text": (
            "Władysław Raginis phục vụ trong Quân đoàn Phòng thủ Biên giới (KOP) "
            "và tham gia chiến đấu năm 1939."
        ),
        "metadata": {"people": ["Władysław Raginis"]},
        "url": "https://example.test/raginis",
    }
    contaminated_identity = {
        **kop,
        "text": f"{kop['text']} {collision['text']}",
        "metadata": {"subject_type": "person", "people": ["Władysław Raginis"]},
    }
    assert classify_subject(contaminated_identity) == "organization"

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [kop, collision][:top_k]

    row = list(build_custom_trajectories(
        _write_corpus(tmp_path, [kop]), Retriever(),
        config=CustomBuildConfig(task_counts={"factual": 1}, seed=9, max_corpus_records=1),
    ))[0]
    payload = json.loads(next(message["content"] for message in row["messages"] if message["role"] == "tool"))
    question = next(message["content"] for message in row["messages"] if message["role"] == "user")
    assert row["provenance"]["subject_type"] == "organization"
    assert [result["chunk_id"] for result in payload] == ["kop-band"]
    assert "Władysław Raginis" not in row["messages"][-1]["content"]
    assert all(fragment not in question.casefold() for fragment in ("là ai", "cuộc đời", "nhân vật này"))


def test_vinh_cat_musician_full_builder_uses_person_template(tmp_path: Path):
    musician = {
        "chunk_id": "vinh-cat", "title": "Vĩnh Cát (nhạc sĩ)", "history_score": 20,
        "text": (
            "Vĩnh Cát là một nhạc sĩ Việt Nam, sinh năm 1934 và hoạt động âm nhạc trong nhiều thập niên. "
            "Ông có đóng góp cho đời sống nghệ thuật Việt Nam."
        ),
        "metadata": {"subject_type": "state", "years": [1934]},
        "url": "https://example.test/vinh-cat",
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [musician]

    row = list(build_custom_trajectories(
        _write_corpus(tmp_path, [musician]), Retriever(),
        config=CustomBuildConfig(task_counts={"factual": 1}, seed=8, max_corpus_records=1),
    ))[0]
    question = next(message["content"] for message in row["messages"] if message["role"] == "user")
    assert row["provenance"]["subject_type"] == "person"
    assert "là ai" in question.casefold() and "nhân vật này" in question.casefold()


@pytest.mark.parametrize(("title", "text", "expected"), [
    ("Nhà Nguyễn", "Nhà Nguyễn là một triều đại Việt Nam bắt đầu năm 1802.", "dynasty"),
    ("Nhà Mạc", "Nhà Mạc là một triều đại Việt Nam bắt đầu năm 1527.", "dynasty"),
    (
        "Nhà Tiền Lý",
        "Nhà Tiền Lý (chữ Nôm: 前李, chữ Hán: 前李朝, Hán Việt: Tiền Lý triều, 544–602) "
        "là một triều đại trong lịch sử Việt Nam.",
        "dynasty",
    ),
    ("Nhà Kim", "Nhà Kim (1115–1234) là một triều đại do người Nữ Chân gây dựng.", "dynasty"),
    ("Nhà nguyện đá (Všemily)", "Nhà nguyện đá là một công trình tôn giáo được xây dựng năm 1760.", "location"),
    ("Nhà Lớn Long Sơn", "Nhà Lớn Long Sơn là một quần thể kiến trúc tại Vũng Tàu.", "location"),
    ("Nhà diện 2/IV", "Nhà diện 2/IV là một loại nhà thuộc chính sách quản lý tài sản.", "topic"),
])
def test_nha_title_grammar_requires_real_dynasty_identity(title: str, text: str, expected: str):
    assert classify_subject({"title": title, "text": text, "metadata": {"subject_type": "dynasty"}}) == expected


def test_builder_and_audit_share_bounded_seed_identity_snapshot(tmp_path: Path):
    dynasty = {
        "chunk_id": "early-ly", "title": "Nhà Tiền Lý", "history_score": 35,
        "text": (
            "Nhà Tiền Lý (chữ Nôm: 前李, chữ Hán: 前李朝, Hán Việt: Tiền Lý triều, 544–602) "
            "là một triều đại trong lịch sử Việt Nam, gắn với quốc hiệu Vạn Xuân."
        ),
        # Contextual metadata deliberately names a different dynasty; the
        # exact lead sentence identifies this article subject.
        "metadata": {"dynasties": ["Nhà Ngô"], "years": [544, 602]},
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [dynasty]

    row = list(build_custom_trajectories(
        _write_corpus(tmp_path, [dynasty]), Retriever(),
        config=CustomBuildConfig(task_counts={"factual": 1}, seed=17, max_corpus_records=1),
    ))[0]
    assert row["provenance"]["subject_type"] == "dynasty"
    assert row["provenance"]["primary_subject_identity"]["title"] == "Nhà Tiền Lý"
    assert audit_rows([row], strict_custom=True)["issues"].get("subject_type_mismatch", 0) == 0

    corrupted = json.loads(json.dumps(row, ensure_ascii=False))
    corrupted["provenance"]["subject_type"] = "person"
    assert audit_rows([corrupted], strict_custom=True)["issues"]["subject_type_mismatch"] == 1


def test_non_dynasty_nha_candidates_are_skipped_and_not_compared_as_dynasties(tmp_path: Path):
    dynasty_rows = [
        {
            "chunk_id": "nguyen", "title": "Nhà Nguyễn", "history_score": 30,
            "text": "Nhà Nguyễn là một triều đại Việt Nam từ năm 1802 và có vai trò lịch sử.",
            "metadata": {"dynasties": ["Nhà Nguyễn"]},
        },
        {
            "chunk_id": "mac", "title": "Nhà Mạc", "history_score": 30,
            "text": "Nhà Mạc là một triều đại Việt Nam từ năm 1527 và có vai trò lịch sử.",
            "metadata": {"dynasties": ["Nhà Mạc"]},
        },
    ]
    invalid = [
        {
            "chunk_id": "chapel", "title": "Nhà nguyện đá (Všemily)", "history_score": 30,
            "text": "Nhà nguyện đá là một công trình tôn giáo được xây dựng năm 1760.",
            "metadata": {"subject_type": "dynasty"},
        },
        {
            "chunk_id": "large-house", "title": "Nhà Lớn Long Sơn", "history_score": 30,
            "text": "Nhà Lớn Long Sơn là một quần thể kiến trúc được xây dựng năm 1910.",
            "metadata": {"subject_type": "dynasty"},
        },
        {
            "chunk_id": "house-class", "title": "Nhà diện 2/IV", "history_score": 30,
            "text": "Nhà diện 2/IV là một loại nhà thuộc chính sách tài sản năm 1977.",
            "metadata": {"subject_type": "dynasty"},
        },
    ]
    records = [*invalid, *dynasty_rows]

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [record for record in records if record["title"] in query][:top_k]

    row = list(build_custom_trajectories(
        _write_corpus(tmp_path, records), Retriever(),
        config=CustomBuildConfig(task_counts={"compare": 1}, seed=14, max_corpus_records=len(records)),
    ))[0]
    assert {row["provenance"]["primary_title"], row["provenance"]["secondary_title"]} == {
        "Nhà Nguyễn", "Nhà Mạc",
    }
    assert row["provenance"]["subject_type"] == row["provenance"]["secondary_subject_type"] == "dynasty"


def test_group_safe_stratification_and_split_coverage_report_cover_custom_behaviors():
    rows = []
    for task in IMPORTANT_CUSTOM_TASK_TYPES:
        for variant in range(3):
            rows.append(_simple_row(
                f"{task}-{variant}", source="custom_history", task=task,
                group=f"{task}-group-{variant}",
            ))
    for index in range(3):
        rows.append(_simple_row(f"agent-{index}", source="agent_flan", group=f"agent-group-{index}"))
    splits = split_trajectories(
        rows, train_ratio=1 / 3, validation_ratio=1 / 3, test_ratio=1 / 3, seed=22,
    )
    report = split_coverage_report(splits)
    assert report["source_group_leakage_count"] == 0
    assert all(not report["splits"][name]["missing_custom_task_types"] for name in ("train", "validation", "test"))
    assert all("custom_history" in report["splits"][name]["source_dataset_counts"] for name in ("train", "validation", "test"))


def test_mix_capacity_reports_agent_flan_shortfall_without_duplication():
    sources = {
        "custom_history": [_simple_row(f"c-{i}") for i in range(60)],
        "multi_hop_function_calling": [_simple_row(f"m-{i}") for i in range(20)],
        "agent_flan": [_simple_row("a-0")],
        "vietnam_history_200k": [_simple_row(f"v-{i}") for i in range(20)],
    }
    report = mix_capacity_report(sources, DEFAULT_MIX_RATIOS, requested_total=100)
    assert report["agent_flan_safe_pool_insufficient"] is True
    assert "agent_flan" in report["insufficient_sources"]
    assert report["duplicates_fabricated"] is False


def test_plain_text_tool_observation_is_not_counted_as_empty_retrieval():
    call = tool_call("plain-call", "retrieve", {"query": "history"})
    row = make_trajectory(
        trajectory_id="plain-tool-output", source_dataset="multi_hop_function_calling",
        task_type="single_step_retrieval", tools=[RETRIEVE_TOOL],
        messages=[
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {
                "role": "tool", "name": "retrieve", "tool_call_id": "plain-call",
                "content": "A non-empty plain-text retrieval observation.",
            },
            {"role": "assistant", "content": "Done."},
        ],
        provenance={"requires_final_answer": True, "source_group": "plain-tool-output"},
    )
    report = audit_rows([row], strict_custom=True)
    assert report["empty_tool_results_total"] == 0
    assert report["unexpected_empty_tool_results"] == 0


def test_final_go_train_gate_covers_analysis_actions_linkage_citations_groups_and_tokens():
    clean = TrajectorySplits(
        train=[_simple_row("train")],
        validation=[_simple_row("validation")],
        test=[_simple_row("test")],
    )
    report = final_dataset_gate(clean, tokenizer=CharacterTokenizer(), max_seq_length=100_000)
    assert report["valid"]
    assert report["token_safety_evaluated"]
    assert all(report["gates"].values())

    action = _simple_row("action", source="agent_flan")
    action["messages"][-1]["content"] = "Action: go to kitchen"
    leaking = TrajectorySplits(train=[action], validation=clean.validation, test=clean.test)
    action_report = final_dataset_gate(leaking, tokenizer=CharacterTokenizer(), max_seq_length=100_000)
    assert not action_report["gates"]["agent_action_without_tools"]
    assert action_report["counts"]["agent_flan_literal_action_without_tools"] == 1

    analysis = _simple_row("analysis", source="vietnam_history_200k")
    analysis["messages"][-1]["channel"] = "analysis"
    analysis_report = final_dataset_gate(
        TrajectorySplits(train=[analysis], validation=clean.validation, test=clean.test),
        tokenizer=CharacterTokenizer(), max_seq_length=100_000,
    )
    assert not analysis_report["gates"]["analysis_leakage"]

    shared = _simple_row("shared", group="shared-group")
    group_report = final_dataset_gate(
        TrajectorySplits(train=[shared], validation=[_simple_row("same", group="shared-group")], test=clean.test),
        tokenizer=CharacterTokenizer(), max_seq_length=100_000,
    )
    assert not group_report["gates"]["group_leakage"]

    tiny_report = final_dataset_gate(clean, tokenizer=CharacterTokenizer(), max_seq_length=8)
    assert not tiny_report["gates"]["token_safety"]

    call = tool_call("good-call", "retrieve", {"query": "history"})
    broken_link = make_trajectory(
        trajectory_id="broken-link", source_dataset="fixture", task_type="tool",
        tools=[RETRIEVE_TOOL],
        messages=[
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "name": "retrieve", "tool_call_id": "wrong-call", "content": "[]"},
            {"role": "assistant", "content": "Done."},
        ],
        provenance={"requires_final_answer": True, "source_group": "broken-link"},
    )
    linkage_report = final_dataset_gate(
        TrajectorySplits(train=[broken_link], validation=clean.validation, test=clean.test),
        tokenizer=CharacterTokenizer(), max_seq_length=100_000,
    )
    assert not linkage_report["gates"]["tool_linkage"]
    assert linkage_report["counts"]["tool_call_linkage_errors"] == 1

    citation_call = tool_call("citation-call", "retrieve", {"query": "history"})
    bad_citation = make_trajectory(
        trajectory_id="bad-citation", source_dataset="custom_history", task_type="factual",
        tools=[RETRIEVE_TOOL],
        messages=[
            {"role": "user", "content": "Search history."},
            {"role": "assistant", "content": None, "tool_calls": [citation_call]},
            {
                "role": "tool", "name": "retrieve", "tool_call_id": "citation-call",
                "content": json.dumps([{"chunk_id": "evidence-1", "text": "Historical evidence."}]),
            },
            {"role": "assistant", "content": "Unsupported citation. [missing-id]"},
        ],
        provenance={
            "requires_final_answer": True, "source_group": "bad-citation",
            "grounded": True, "evidence_ids": ["evidence-1"],
        },
    )
    citation_report = final_dataset_gate(
        TrajectorySplits(train=[bad_citation], validation=clean.validation, test=clean.test),
        tokenizer=CharacterTokenizer(), max_seq_length=100_000,
    )
    assert not citation_report["gates"]["citations"]
    assert citation_report["counts"]["custom_citation_errors"] > 0
