from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training.trajectory_dataset.audit import audit_rows
from training.trajectory_dataset.builders.custom_history import (
    CustomBuildConfig,
    QueryPlan,
    build_custom_trajectories,
    classify_subject,
    compact_observation,
    is_result_relevant_to_target,
    is_vietnam_history_relevant,
    load_seed_records,
)
from training.trajectory_dataset.io_utils import atomic_write_jsonl
from training.trajectory_dataset.schema import make_trajectory, tool_call


def person(chunk_id: str, title: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text,
        "url": f"https://example.test/{chunk_id}",
        "metadata": {"subject_type": "person", "people": [title], "countries": ["Việt Nam"]},
    }


def event(chunk_id: str, title: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "title": title,
        "text": text,
        "url": f"https://example.test/{chunk_id}",
        "metadata": {"subject_type": "event", "events": [title], "countries": ["Việt Nam"]},
    }


def write_corpus(tmp_path: Path, rows: list[dict], name: str = "corpus.jsonl") -> Path:
    path = tmp_path / name
    atomic_write_jsonl(path, rows)
    return path


def factual_config(count: int = 1) -> CustomBuildConfig:
    return CustomBuildConfig(
        task_counts={"factual": count},
        top_k=6,
        seed=31,
        max_corpus_records=100,
        observation_char_budget=2_000,
        trajectory_observation_char_budget=2_000,
        max_result_text_chars=600,
        max_candidate_attempts_per_task=100,
    )


@pytest.mark.parametrize(
    "title",
    [
        "Xẩm", "Lý học", "Nho giáo", "Phật giáo", "Chủ nghĩa cộng sản",
        "Chữ Quốc ngữ", "Chữ Nôm", "Tiếng Việt", "Văn hóa Việt Nam",
    ],
)
def test_non_person_language_and_culture_titles_override_weak_or_wrong_person_evidence(title: str):
    row = {
        "title": title,
        "text": f"{title} là một chủ đề lịch sử. Thiết Mộc Chân là một nhân vật có tiểu sử riêng.",
        "metadata": {"subject_type": "person", "people": [title]},
    }
    assert classify_subject(row) == "topic"
    assert classify_subject({"title": title, "text": f"{title} là một khái niệm lịch sử."}) == "topic"


@pytest.mark.parametrize(
    ("title", "text"),
    [
        ("Nguyễn Bình", "Nguyễn Bình (1908–1951) là một tướng lĩnh Việt Nam."),
        ("Po Saong Nyung Ceng", "Po Saong Nyung Ceng lãnh đạo nhiều hoạt động trong lịch sử Chăm."),
        ("Phaolô Nguyễn Văn Bình", "Phaolô Nguyễn Văn Bình sinh năm 1910 và mất năm 1995."),
    ],
)
def test_person_fallback_requires_subject_linked_biographical_evidence(title: str, text: str):
    assert classify_subject({"title": title, "text": text}) == "person"


def test_general_subject_typing_distinguishes_military_organization_dynasty_and_person():
    foreign_air_force = {
        "title": "Không quân Nhân dân Triều Tiên",
        "text": "Không quân Nhân dân Triều Tiên là lực lượng quân sự của Triều Tiên.",
        "metadata": {"subject_type": "dynasty", "dynasties": ["Không quân Nhân dân Triều Tiên"]},
    }
    nha_mac = {
        "title": "Nhà Mạc",
        "text": "Nhà Mạc là một triều đại trong lịch sử Việt Nam.",
        "metadata": {"subject_type": "dynasty", "dynasties": ["Nhà Mạc"]},
    }
    nguyen_binh = {
        "title": "Nguyễn Bình",
        "text": "Nguyễn Bình (1908–1951) là một tướng lĩnh Việt Nam.",
    }
    hoi_an = {
        "title": "Hội An",
        "text": "Hội An là một đô thị lịch sử tại Việt Nam.",
        "metadata": {"subject_type": "location", "locations": ["Hội An"]},
    }
    assert classify_subject(foreign_air_force) == "organization"
    assert classify_subject(nha_mac) == "dynasty"
    assert classify_subject(nguyen_binh) == "person"
    assert classify_subject(hoi_an) == "location"


def test_vietnam_history_domain_eligibility_is_conservative():
    nha_mac = {
        "title": "Nhà Mạc",
        "text": "Nhà Mạc là một triều đại trong lịch sử Việt Nam.",
        "metadata": {"subject_type": "dynasty"},
    }
    nguyen_binh = {
        "title": "Nguyễn Bình",
        "text": "Nguyễn Bình là một tướng lĩnh Việt Nam.",
        "metadata": {"subject_type": "person"},
    }
    border_war = event(
        "border-war",
        "Chiến tranh biên giới Việt Nam–Campuchia",
        "Chiến tranh biên giới Việt Nam–Campuchia có Việt Nam là một chủ thể trực tiếp.",
    )
    foreign_air_force = {
        "title": "Không quân Nhân dân Triều Tiên",
        "text": "Lực lượng này phụ trách phòng không và không quân của Triều Tiên.",
        "metadata": {"subject_type": "organization", "countries": ["Triều Tiên"]},
    }
    assert is_vietnam_history_relevant(nha_mac)
    assert is_vietnam_history_relevant(nguyen_binh)
    assert is_vietnam_history_relevant(border_war)
    assert not is_vietnam_history_relevant(foreign_air_force)


@pytest.mark.parametrize(
    "row",
    [
        {"title": "Võ Trứ", "text": "Võ Trứ lãnh đạo một cuộc khởi nghĩa.", "metadata": {"subject_type": "person"}},
        {"title": "Dương Văn Hiếu", "text": "Dương Văn Hiếu hoạt động trong thế kỷ XX.", "metadata": {"subject_type": "person"}},
        {"title": "Phaolô Nguyễn Văn Bình", "text": "Phaolô Nguyễn Văn Bình sinh năm 1910.", "metadata": {"subject_type": "person"}},
        {"title": "Po Saong Nyung Ceng", "text": "Po Saong Nyung Ceng lãnh đạo nhiều hoạt động trong lịch sử người Chăm.", "metadata": {"subject_type": "person"}},
        {"title": "Chiến tranh Đại Việt–Khmer", "text": "Chiến tranh Đại Việt–Khmer diễn ra trong thế kỷ XII.", "metadata": {"subject_type": "event"}},
        {"title": "Chiến tranh biên giới Việt Nam – Campuchia", "text": "Việt Nam là một chủ thể trực tiếp trong cuộc chiến.", "metadata": {"subject_type": "event"}},
        {"title": "Panduranga", "text": "Panduranga là một chủ thể trong lịch sử Chăm Pa và Đại Việt.", "metadata": {"subject_type": "state"}},
        {"title": "Pháp", "text": "Pháp thiết lập chế độ thuộc địa tại Việt Nam trong thế kỷ XIX.", "metadata": {"subject_type": "state"}},
    ],
)
def test_vietnam_history_domain_accepts_vietnamese_regional_and_direct_foreign_context(row: dict):
    assert is_vietnam_history_relevant(row)


def test_compare_requires_both_targets_to_pass_domain_gate(tmp_path: Path):
    vietnamese_people = [
        person("nguyen-binh", "Nguyễn Bình", "Nguyễn Bình giữ vai trò chỉ huy và đóng góp tại Nam Bộ."),
        person("vo-tru", "Võ Trứ", "Võ Trứ lãnh đạo hoạt động và có đóng góp trong lịch sử Việt Nam."),
    ]
    foreign = {
        "chunk_id": "foreign-person",
        "title": "Kim Mẫu",
        "text": "Kim Mẫu giữ vai trò chỉ huy tại Triều Tiên.",
        "metadata": {"subject_type": "person", "people": ["Kim Mẫu"], "countries": ["Triều Tiên"]},
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [*vietnamese_people, foreign][:top_k]

    config = CustomBuildConfig(task_counts={"compare": 1}, seed=29, max_corpus_records=3)
    built = list(build_custom_trajectories(
        write_corpus(tmp_path, [foreign, *vietnamese_people]), Retriever(), config=config,
    ))
    assert len(built) == 1
    assert {
        built[0]["provenance"]["primary_title"],
        built[0]["provenance"]["secondary_title"],
    } == {"Nguyễn Bình", "Võ Trứ"}


def test_entity_relevance_rejects_similar_name_and_accepts_explicit_broader_article():
    exact = person("nguyen-binh", "Nguyễn Bình", "Nguyễn Bình chỉ huy lực lượng ở Nam Bộ.")
    similar = person("nguyen-binh-khiem", "Nguyễn Bỉnh Khiêm", "Nguyễn Bỉnh Khiêm là một danh nhân.")
    broader = {
        "chunk_id": "nam-bo",
        "title": "Lịch sử Nam Bộ",
        "text": "Trong giai đoạn này, Nguyễn Bình chỉ huy lực lượng kháng chiến tại Nam Bộ.",
        "metadata": {"people": ["Nguyễn Bình"]},
    }
    kwargs = {
        "target_title": "Nguyễn Bình",
        "target_subject_type": "person",
        "retrieval_role": "factual",
    }
    assert is_result_relevant_to_target(exact, **kwargs)
    assert not is_result_relevant_to_target(similar, **kwargs)
    assert is_result_relevant_to_target(broader, **kwargs)


def test_event_relevance_disambiguates_medieval_conflict_from_modern_cambodia():
    medieval = event(
        "medieval",
        "Chiến tranh Đại Việt–Khmer",
        "Chiến tranh Đại Việt–Khmer diễn ra trong thế kỷ XII giữa Đại Việt và Khmer.",
    )
    modern = event(
        "modern",
        "Chiến tranh biên giới Việt Nam – Campuchia",
        "Cuộc chiến hiện đại liên quan đến Khmer Đỏ trong thập niên 1970.",
    )
    kwargs = {
        "target_title": "Chiến tranh Đại Việt–Khmer",
        "target_subject_type": "event",
        "retrieval_role": "factual",
    }
    assert is_result_relevant_to_target(medieval, **kwargs)
    assert not is_result_relevant_to_target(modern, **kwargs)


def test_compaction_is_entity_and_facet_aware_before_answers_and_citations(tmp_path: Path):
    target = person(
        "nguyen-binh",
        "Nguyễn Bình",
        "Nguyễn Bình giữ vai trò chỉ huy và có đóng góp cho kháng chiến Nam Bộ.",
    )
    similar = person(
        "nguyen-binh-khiem",
        "Nguyễn Bỉnh Khiêm",
        "Nguyễn Bỉnh Khiêm để lại nhiều trước tác và lời sấm.",
    )

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [similar, target][:top_k]

    row = list(build_custom_trajectories(
        write_corpus(tmp_path, [target]), Retriever(), config=factual_config(),
    ))[0]
    payload = json.loads(next(message["content"] for message in row["messages"] if message["role"] == "tool"))
    assert [result["chunk_id"] for result in payload] == ["nguyen-binh"]
    answer = row["messages"][-1]["content"]
    assert "[nguyen-binh]" in answer
    assert "nguyen-binh-khiem" not in answer
    assert row["provenance"]["evidence_ids"] == ["nguyen-binh"]


def test_person_summary_drops_unrelated_person_sentence_and_requires_each_facet(tmp_path: Path):
    target = person(
        "nguyen-binh",
        "Nguyễn Bình",
        "Nguyễn Bình sinh năm 1908 và hoạt động ở Nam Bộ. Nguyễn Bình chỉ huy lực lượng và góp phần vào kháng chiến.",
    )
    noisy = copy.deepcopy(target)
    noisy["chunk_id"] = "mixed"
    noisy["text"] = (
        "Thiết Mộc Chân thống nhất các bộ lạc Mông Cổ. "
        "Nguyễn Bình sinh năm 1908 và hoạt động ở Nam Bộ. "
        "Nguyễn Bình giữ vai trò chỉ huy và góp phần vào kháng chiến."
    )

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [noisy]

    config = CustomBuildConfig(task_counts={"summary": 1}, seed=5, max_corpus_records=1)
    row = list(build_custom_trajectories(write_corpus(tmp_path, [target]), Retriever(), config=config))[0]
    observations = " ".join(
        result["text"]
        for message in row["messages"] if message["role"] == "tool"
        for result in json.loads(message["content"])
    )
    assert "Nguyễn Bình" in observations
    assert "Thiết Mộc Chân" not in observations
    assert "Nguyễn Bình" in row["messages"][-1]["content"]


def test_corrective_facet_rejects_definition_only_sentence_and_accepts_failure_evidence():
    plan = QueryPlan(
        "search_history", "Chiến dịch Mẫu khó khăn hạn chế thất bại", 4, "corrective_facet",
    )
    definition = event(
        "definition",
        "Chiến dịch Mẫu",
        "Chiến dịch Mẫu là một loạt hoạt động quân sự diễn ra từ năm 1123 đến năm 1150.",
    )
    limitation = event("limit", "Chiến dịch Mẫu", "Chiến dịch Mẫu gặp khó khăn, hạn chế và nhiều bất lợi.")
    compact = compact_observation(
        [definition, limitation],
        plan,
        task_type="hard_negative",
        observation_char_budget=2_000,
        max_result_text_chars=500,
        target_title="Chiến dịch Mẫu",
        target_subject_type="event",
    )
    assert [result["chunk_id"] for result in compact] == ["limit"]


@pytest.mark.parametrize(
    ("role", "task_type", "query", "invalid_text"),
    [
        (
            "result_significance",
            "significance",
            "Chiến dịch Mẫu kết quả ý nghĩa tác động vai trò",
            "Chiến dịch Mẫu nổ ra do điều kiện chính trị và bối cảnh khu vực.",
        ),
        (
            "context_cause",
            "cause",
            "Chiến dịch Mẫu bối cảnh nguyên nhân điều kiện",
            "Chiến dịch Mẫu kết thúc với thắng lợi và để lại kết quả quan trọng.",
        ),
    ],
)
def test_required_cause_and_significance_facets_reject_wrong_facet_sentences(
    role: str, task_type: str, query: str, invalid_text: str,
):
    plan = QueryPlan("search_history", query, 4, role)
    invalid = event("invalid", "Chiến dịch Mẫu", invalid_text)
    compact = compact_observation(
        [invalid],
        plan,
        task_type=task_type,
        observation_char_budget=2_000,
        max_result_text_chars=500,
        target_title="Chiến dịch Mẫu",
        target_subject_type="event",
    )
    assert compact == []


def test_significance_rejects_weak_opening_sentence_without_impact_or_result():
    plan = QueryPlan(
        "search_history", "Nhà Mạc kết quả ý nghĩa tác động vai trò", 4, "result_significance",
    )
    weak = {
        "chunk_id": "weak",
        "title": "Nhà Mạc",
        "text": "Những người đắc lực giúp ông mở ra nhà Mạc.",
        "metadata": {"subject_type": "dynasty", "countries": ["Việt Nam"]},
    }
    compact = compact_observation(
        [weak],
        plan,
        task_type="significance",
        observation_char_budget=2_000,
        max_result_text_chars=500,
        target_title="Nhà Mạc",
        target_subject_type="dynasty",
    )
    assert compact == []


def test_cause_rejects_late_equipment_and_parade_sentence():
    title = "Không quân Nhân dân Triều Tiên"
    plan = QueryPlan(
        "search_history", f"{title} bối cảnh nguyên nhân điều kiện hình thành", 4, "context_cause",
    )
    late_event = {
        "chunk_id": "pechora",
        "title": title,
        "text": f"{title} trang bị hệ thống Pechora và trình diễn trong các cuộc duyệt binh năm 2011 và 2012.",
        "metadata": {"subject_type": "organization"},
    }
    compact = compact_observation(
        [late_event],
        plan,
        task_type="cause",
        observation_char_budget=2_000,
        max_result_text_chars=500,
        target_title=title,
        target_subject_type="organization",
    )
    assert compact == []


@pytest.mark.parametrize(
    "text",
    [
        "Chiến dịch Mẫu có nguyên nhân từ những xung đột kéo dài trước đó.",
        "Bối cảnh lúc này khiến Chiến dịch Mẫu nổ ra trên toàn khu vực.",
        "Được thành lập vào năm 1951 trong tình thế cấp bách của khu vực.",
    ],
)
def test_cause_facet_accepts_actual_cause_background_and_formation(text: str):
    plan = QueryPlan("search_history", "Chiến dịch Mẫu bối cảnh nguyên nhân", 4, "context_cause")
    compact = compact_observation(
        [event("cause", "Chiến dịch Mẫu", text)], plan,
        task_type="cause", observation_char_budget=2_000, max_result_text_chars=500,
        target_title="Chiến dịch Mẫu", target_subject_type="event",
    )
    assert [result["chunk_id"] for result in compact] == ["cause"]


@pytest.mark.parametrize(
    "text",
    [
        "Không quân Nhân dân Triều Tiên duy trì năng lực quân sự tương đương bằng không quân như một lực lượng ngăn chặn.",
        "Không quân Nhân dân Triều Tiên trang bị hệ thống Pechora và máy bay chiến đấu mới.",
        "Không quân Nhân dân Triều Tiên kết thúc chiến dịch với hệ quả đáng kể cho khu vực.",
    ],
)
def test_cause_facet_rejects_later_capability_equipment_and_consequence(text: str):
    title = "Không quân Nhân dân Triều Tiên"
    plan = QueryPlan("search_history", f"{title} bối cảnh nguyên nhân", 4, "context_cause")
    compact = compact_observation(
        [{"chunk_id": "late", "title": title, "text": text, "metadata": {"subject_type": "organization"}}],
        plan, task_type="cause", observation_char_budget=2_000, max_result_text_chars=500,
        target_title=title, target_subject_type="organization",
    )
    assert compact == []


@pytest.mark.parametrize(
    "text",
    [
        "Nhà Mạc có vai trò tích cực nhất định trong lịch sử Việt Nam.",
        "Nhà Mạc góp phần làm thay đổi cục diện chính trị đương thời.",
        "Nhà Mạc có tác động lâu dài, với hệ quả là một thời kỳ phân tranh.",
    ],
)
def test_significance_facet_accepts_role_contribution_impact_and_consequence(text: str):
    plan = QueryPlan("search_history", "Nhà Mạc ý nghĩa tác động", 4, "result_significance")
    compact = compact_observation(
        [{"chunk_id": "impact", "title": "Nhà Mạc", "text": text, "metadata": {"subject_type": "dynasty"}}],
        plan, task_type="significance", observation_char_budget=2_000, max_result_text_chars=500,
        target_title="Nhà Mạc", target_subject_type="dynasty",
    )
    assert [result["chunk_id"] for result in compact] == ["impact"]


@pytest.mark.parametrize(
    "text",
    [
        "Không quân Nhân dân Triều Tiên cũng có một ít máy bay MiG-29 hiện đại hơn.",
        "Không quân Nhân dân Triều Tiên sở hữu nhiều loại máy bay và trang thiết bị.",
        "Không quân Nhân dân Triều Tiên được thành lập vào tháng 1 năm 1951.",
    ],
)
def test_significance_facet_rejects_equipment_inventory_and_plain_biography(text: str):
    title = "Không quân Nhân dân Triều Tiên"
    plan = QueryPlan("search_history", f"{title} ý nghĩa tác động", 4, "result_significance")
    compact = compact_observation(
        [{"chunk_id": "weak", "title": title, "text": text, "metadata": {"subject_type": "organization"}}],
        plan, task_type="significance", observation_char_budget=2_000, max_result_text_chars=500,
        target_title=title, target_subject_type="organization",
    )
    assert compact == []


@pytest.mark.parametrize(
    "text",
    [
        "Chiến dịch Mẫu thất bại và chịu nhiều tổn thất.",
        "Chiến dịch Mẫu suy yếu trong khủng hoảng kéo dài.",
        "Chiến dịch Mẫu gặp khó khăn vì lực lượng không đủ.",
    ],
)
def test_corrective_facet_accepts_substantive_failure_weakness_and_difficulty(text: str):
    plan = QueryPlan("search_history", "Chiến dịch Mẫu khó khăn thất bại", 4, "corrective_facet")
    compact = compact_observation(
        [event("negative", "Chiến dịch Mẫu", text)], plan,
        task_type="hard_negative", observation_char_budget=2_000, max_result_text_chars=500,
        target_title="Chiến dịch Mẫu", target_subject_type="event",
    )
    assert [result["chunk_id"] for result in compact] == ["negative"]


def test_wrong_facet_remains_optional_and_is_not_hard_gated():
    plan = QueryPlan(
        "search_history", "Chiến dịch Mẫu thành công thắng lợi", 4,
        "wrong_facet", required=False, expected_empty=True,
    )
    neutral = event(
        "neutral", "Chiến dịch Mẫu",
        "Chiến dịch Mẫu diễn ra qua nhiều giai đoạn từ năm 1123 đến năm 1150.",
    )
    compact = compact_observation(
        [neutral], plan, task_type="hard_negative", observation_char_budget=2_000,
        max_result_text_chars=500, target_title="Chiến dịch Mẫu", target_subject_type="event",
    )
    assert [result["chunk_id"] for result in compact] == ["neutral"]


def test_required_facet_empty_after_filter_and_fallback_skips_candidate(tmp_path: Path):
    title = "Chiến tranh Đại Việt–Khmer"
    seed = event(
        "war",
        title,
        f"{title} là một loạt các cuộc chiến tranh diễn ra từ năm 1123 đến năm 1150.",
    )

    class DefinitionOnlyRetriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [seed]

    config = CustomBuildConfig(task_counts={"hard_negative": 1}, seed=7, max_corpus_records=1)
    with pytest.raises(ValueError, match="0/1 hard_negative"):
        list(build_custom_trajectories(
            write_corpus(tmp_path, [seed]), DefinitionOnlyRetriever(), config=config,
        ))


def test_unrelated_foreign_subject_is_skipped_before_quota_counting(tmp_path: Path):
    foreign = {
        "chunk_id": "foreign-air-force",
        "title": "Không quân Nhân dân Triều Tiên",
        "text": "Không quân Nhân dân Triều Tiên là lực lượng quân sự của Triều Tiên.",
        "metadata": {"subject_type": "dynasty", "countries": ["Triều Tiên"]},
    }
    vietnamese = {
        "chunk_id": "nha-mac",
        "title": "Nhà Mạc",
        "text": "Nhà Mạc là một triều đại trong lịch sử Việt Nam và có nhiều dấu mốc lịch sử.",
        "metadata": {"subject_type": "dynasty", "countries": ["Việt Nam"]},
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [foreign, vietnamese][:top_k]

    config = CustomBuildConfig(task_counts={"factual": 1}, seed=13, max_corpus_records=2)
    built = list(build_custom_trajectories(
        write_corpus(tmp_path, [foreign, vietnamese]), Retriever(), config=config,
    ))
    assert len(built) == 1
    assert built[0]["provenance"]["primary_title"] == "Nhà Mạc"


def test_strict_audit_catches_organization_as_dynasty_and_foreign_domain():
    title = "Không quân Nhân dân Triều Tiên"
    row = make_trajectory(
        trajectory_id="bad-foreign-subject",
        source_dataset="custom_history",
        task_type="factual",
        messages=[
            {"role": "user", "content": f"Các dấu mốc của {title} là gì?"},
            {"role": "assistant", "content": "Chưa đủ bằng chứng."},
        ],
        tools=[],
        difficulty="medium",
        provenance={
            "subject_type": "dynasty",
            "primary_title": title,
            "source_group": "foreign",
            "requires_final_answer": True,
        },
    )
    report = audit_rows([row], strict_custom=True)
    assert report["issues"]["subject_type_mismatch"] == 1
    assert report["issues"]["domain_mismatch"] == 1
    assert not report["valid"]


def test_strict_audit_separates_facet_mismatch_from_entity_mismatch(tmp_path: Path):
    title = "Chiến tranh Đại Việt–Khmer"
    seed = event(
        "war",
        title,
        f"{title} gặp khó khăn, chịu nhiều bất lợi và cuối cùng thất bại.",
    )

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [] if "thành công thắng lợi" in query else [seed]

    config = CustomBuildConfig(task_counts={"hard_negative": 1}, seed=7, max_corpus_records=1)
    corrupted = list(build_custom_trajectories(
        write_corpus(tmp_path, [seed]), Retriever(), config=config,
    ))[0]
    definition = event(
        "definition",
        title,
        f"{title} là một loạt các cuộc chiến tranh diễn ra từ năm 1123 đến năm 1150.",
    )
    tool_messages = [message for message in corrupted["messages"] if message["role"] == "tool"]
    tool_messages[1]["content"] = json.dumps([definition], ensure_ascii=False)
    corrupted["messages"][-1]["content"] = f"{definition['text']} [definition]"
    corrupted["provenance"]["observed_evidence_ids"] = ["definition"]
    corrupted["provenance"]["evidence_ids"] = ["definition"]

    report = audit_rows([corrupted], strict_custom=True)
    assert report["issues"]["observation_facet_mismatch"] == 1
    assert report["issues"]["final_answer_facet_mismatch"] == 1
    assert report["issues"].get("observation_target_mismatch", 0) == 0
    assert not report["valid"]


@pytest.mark.parametrize(
    ("task_type", "role", "query", "bad_text"),
    [
        (
            "cause", "context_cause", "Chiến tranh Đại Việt–Khmer bối cảnh nguyên nhân",
            "Chiến tranh Đại Việt–Khmer duy trì năng lực quân sự bằng lực lượng ngăn chặn.",
        ),
        (
            "significance", "result_significance", "Chiến tranh Đại Việt–Khmer ý nghĩa tác động",
            "Chiến tranh Đại Việt–Khmer có một ít trang bị quân sự hiện đại hơn.",
        ),
    ],
)
def test_strict_audit_rejects_required_observation_and_answer_with_wrong_facet(
    task_type: str, role: str, query: str, bad_text: str,
):
    evidence = event("bad-facet", "Chiến tranh Đại Việt–Khmer", bad_text)
    row = make_trajectory(
        trajectory_id=f"bad-{role}",
        source_dataset="custom_history",
        task_type=task_type,
        messages=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": "", "tool_calls": [
                tool_call("call-1", "search_history", {"query": query, "top_k": 4}),
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": json.dumps([evidence], ensure_ascii=False)},
            {"role": "assistant", "content": f"{bad_text} [bad-facet]"},
        ],
        provenance={
            "subject_type": "event",
            "primary_title": "Chiến tranh Đại Việt–Khmer",
            "primary_aliases": [],
            "vietnam_history_relevant": True,
            "vietnam_history_relevance_signals": ["title:dai viet"],
            "retrieval_queries": [{"query": query, "role": role, "required": True}],
            "observed_evidence_ids": ["bad-facet"],
            "evidence_ids": ["bad-facet"],
            "requires_final_answer": True,
        },
    )
    report = audit_rows([row], strict_custom=True)
    assert report["issues"]["observation_facet_mismatch"] == 1
    assert report["issues"]["final_answer_facet_mismatch"] == 1
    assert not report["valid"]


def test_compare_target_isolation_rejects_wrong_person_for_either_side():
    plan_a = QueryPlan("search_history", "Nguyễn Bình vai trò đóng góp", 4, "target_a")
    plan_b = QueryPlan("search_history", "Phaolô Nguyễn Văn Bình vai trò đóng góp", 4, "target_b")
    wrong_a = person("wrong-a", "Nguyễn Bỉnh Khiêm", "Nguyễn Bỉnh Khiêm có ảnh hưởng văn hóa.")
    right_b = person(
        "right-b",
        "Phaolô Nguyễn Văn Bình",
        "Phaolô Nguyễn Văn Bình giữ vai trò tổng giám mục và có nhiều đóng góp.",
    )
    common = {
        "task_type": "compare",
        "observation_char_budget": 2_000,
        "max_result_text_chars": 500,
    }
    left = compact_observation(
        [wrong_a], plan_a, target_title="Nguyễn Bình", target_subject_type="person",
        excluded_titles=("Phaolô Nguyễn Văn Bình",), **common,
    )
    right = compact_observation(
        [right_b], plan_b, target_title="Phaolô Nguyễn Văn Bình", target_subject_type="person",
        excluded_titles=("Nguyễn Bình",), **common,
    )
    assert left == []
    assert [result["chunk_id"] for result in right] == ["right-b"]


def test_strict_semantic_audit_rejects_subject_and_observation_corruption(tmp_path: Path):
    subject_row = make_trajectory(
        trajectory_id="bad-subject",
        source_dataset="custom_history",
        task_type="factual",
        messages=[
            {"role": "user", "content": "Chữ Quốc ngữ là ai?"},
            {"role": "assistant", "content": "Không phù hợp."},
        ],
        tools=[],
        difficulty="medium",
        provenance={
            "subject_type": "person",
            "primary_title": "Chữ Quốc ngữ",
            "source_group": "script",
            "requires_final_answer": True,
        },
    )
    subject_report = audit_rows([subject_row], strict_custom=True)
    assert subject_report["issues"]["subject_type_mismatch"] == 1
    assert not subject_report["valid"]

    target = person("nguyen-binh", "Nguyễn Bình", "Nguyễn Bình chỉ huy lực lượng tại Nam Bộ.")

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [target]

    corrupted = list(build_custom_trajectories(
        write_corpus(tmp_path, [target]), Retriever(), config=factual_config(),
    ))[0]
    wrong = person("wrong", "Nguyễn Bỉnh Khiêm", "Nguyễn Bỉnh Khiêm là một danh nhân văn hóa.")
    tool_message = next(message for message in corrupted["messages"] if message["role"] == "tool")
    tool_message["content"] = json.dumps([wrong], ensure_ascii=False)
    corrupted["messages"][-1]["content"] = "Nguyễn Bỉnh Khiêm là một danh nhân. [wrong]"
    corrupted["provenance"]["observed_evidence_ids"] = ["wrong"]
    corrupted["provenance"]["evidence_ids"] = ["wrong"]
    report = audit_rows([corrupted], strict_custom=True)
    assert report["issues"]["observation_target_mismatch"] == 1
    assert report["issues"]["final_answer_target_mismatch"] == 1
    assert not report["valid"]


def test_low_quality_candidates_do_not_count_and_build_remains_deterministic(tmp_path: Path):
    rows = [
        person("a", "Nguyễn Bình", "Nguyễn Bình sinh năm 1908 và hoạt động tại Nam Bộ."),
        person("b", "Phaolô Nguyễn Văn Bình", "Phaolô Nguyễn Văn Bình sinh năm 1910 và hoạt động mục vụ."),
    ]
    corpus = write_corpus(tmp_path, rows)
    first_seed, second_seed = load_seed_records(corpus, limit=2, seed=31)
    wrong = person("wrong", "Nguyễn Bỉnh Khiêm", "Nguyễn Bỉnh Khiêm là một danh nhân.")

    class Retriever:
        def __init__(self):
            self.calls: list[str] = []

        def search(self, query: str, *, top_k: int) -> list[dict]:
            self.calls.append(query)
            if first_seed["title"] in query:
                return [wrong]
            return [second_seed]

    one = Retriever()
    two = Retriever()
    first = list(build_custom_trajectories(corpus, one, config=factual_config()))
    second = list(build_custom_trajectories(corpus, two, config=factual_config()))
    assert first == second
    assert first[0]["provenance"]["primary_title"] == second_seed["title"]
    assert any(first_seed["title"] in query for query in one.calls)
