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
    compare_subjects_compatible,
    is_custom_history_eligible,
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


@pytest.mark.parametrize(
    ("title", "text", "declared", "expected"),
    [
        ("Asian Idol", "Asian Idol là một cuộc thi ca hát theo dạng Pop Idol.", "person", "topic"),
        ("Kỷ Permi", "Kỷ Permi là một kỷ địa chất trong đại Cổ sinh.", "person", "topic"),
        (
            "Tam Pháp Ty (nhà Nguyễn)",
            "Tam Pháp Ty là cơ quan nhận đơn khiếu nại dưới triều Nguyễn.",
            "dynasty", "organization",
        ),
        ("Srivijaya", "Srivijaya là một đế quốc hàng hải từng tồn tại ở Đông Nam Á.", "person", "state"),
        ("México", "México là một quốc gia tại Bắc Mỹ.", "person", "state"),
        ("Nhà Mạc", "Nhà Mạc là một triều đại Việt Nam.", "dynasty", "dynasty"),
        ("Nhà Nguyễn", "Nhà Nguyễn là một triều đại Việt Nam.", "dynasty", "dynasty"),
        ("Ngô Xuân Lịch", "Ngô Xuân Lịch sinh năm 1954 và là một tướng lĩnh.", "person", "person"),
        (
            "Giải bóng đá Vô địch U-21 Quốc gia 2015",
            "Giải bóng đá Vô địch U-21 Quốc gia 2015 có vòng bảng và trận chung kết.",
            "state", "event",
        ),
        (
            "Việt Nam tại Đại hội Thể thao Đông Nam Á 2007",
            "Việt Nam tại Đại hội Thể thao Đông Nam Á 2007 có bảng tổng sắp huy chương.",
            "organization", "event",
        ),
        ("Nhã nhạc cung đình Huế", "Nhã nhạc cung đình Huế là một loại hình âm nhạc.", "dynasty", "topic"),
        (
            "Quân hàm Lực lượng vũ trang Cách mạng Cuba",
            "Quân hàm Lực lượng vũ trang Cách mạng Cuba là hệ thống cấp bậc quân sự.",
            "event", "topic",
        ),
        ("Tràn ngập (quân sự)", "Tràn ngập là một khái niệm quân sự.", "event", "topic"),
        (
            "Trụ sở Ủy ban nhân dân Thành phố Hồ Chí Minh",
            "Trụ sở Ủy ban nhân dân Thành phố Hồ Chí Minh là một công trình kiến trúc.",
            "organization", "location",
        ),
        (
            "Hội nghị thượng đỉnh Triều Tiên–Hoa Kỳ tại Hà Nội 2019",
            "Hội nghị thượng đỉnh Triều Tiên–Hoa Kỳ tại Hà Nội diễn ra năm 2019.",
            "event", "event",
        ),
        (
            "Hội nghị thành lập Đảng Cộng sản Việt Nam",
            "Hội nghị thành lập Đảng Cộng sản Việt Nam diễn ra đầu năm 1930.",
            "event", "event",
        ),
        ("Hội nghị Fontainebleau 1946", "Hội nghị Fontainebleau diễn ra năm 1946.", "event", "event"),
        (
            "Đảo chính Việt Nam Cộng hòa 1963",
            "Đảo chính Việt Nam Cộng hòa diễn ra vào năm 1963.",
            "event", "event",
        ),
    ],
)
def test_subject_classifier_uses_shared_semantic_contract(
    title: str, text: str, declared: str, expected: str,
):
    assert classify_subject({
        "title": title,
        "text": text,
        "metadata": {"subject_type": declared},
    }) == expected


@pytest.mark.parametrize(
    ("title", "text", "declared", "eligible"),
    [
        ("Asian Idol", "Asian Idol là một cuộc thi ca hát tổ chức năm 2007.", "person", False),
        (
            "Giải bóng đá Vô địch U-21 Quốc gia 2015",
            "Giải bóng đá Vô địch U-21 Quốc gia 2015 có vòng bảng và mỗi trận thắng được 3 điểm.",
            "state", False,
        ),
        (
            "Việt Nam tại Đại hội Thể thao Đông Nam Á 2007",
            "Việt Nam tại Đại hội Thể thao Đông Nam Á 2007 có bảng tổng sắp huy chương.",
            "organization", False,
        ),
        ("Hội nghị Fontainebleau 1946", "Hội nghị Fontainebleau diễn ra năm 1946.", "event", True),
        (
            "Hội nghị thành lập Đảng Cộng sản Việt Nam",
            "Hội nghị thành lập Đảng Cộng sản Việt Nam diễn ra đầu năm 1930.",
            "event", True,
        ),
        (
            "Hội nghị thượng đỉnh Triều Tiên–Hoa Kỳ tại Hà Nội 2019",
            "Hội nghị thượng đỉnh Triều Tiên–Hoa Kỳ diễn ra tại Hà Nội năm 2019.",
            "event", True,
        ),
        ("Chiến tranh Triều Tiên", "Chiến tranh Triều Tiên diễn ra từ năm 1950.", "event", True),
    ],
)
def test_history_eligibility_requires_strong_historical_semantics(
    title: str, text: str, declared: str, eligible: bool,
):
    row = {"title": title, "text": text, "history_score": 99, "metadata": {
        "subject_type": declared, "years": [2007], "events": [title],
    }}
    assert is_custom_history_eligible(row) is eligible


def test_compare_type_contract_allows_semantic_peers_and_rejects_dynasty_institution():
    nha_mac = {
        "title": "Nhà Mạc", "text": "Nhà Mạc là một triều đại từ năm 1527.",
        "metadata": {"subject_type": "dynasty"},
    }
    nha_nguyen = {
        "title": "Nhà Nguyễn", "text": "Nhà Nguyễn là một triều đại từ năm 1802.",
        "metadata": {"subject_type": "dynasty"},
    }
    tam_phap_ty = {
        "title": "Tam Pháp Ty (nhà Nguyễn)",
        "text": "Tam Pháp Ty là cơ quan tư pháp được lập năm 1832 dưới triều Nguyễn.",
        "metadata": {"subject_type": "dynasty"},
    }
    nguyen_binh = person("nb", "Nguyễn Bình", "Nguyễn Bình sinh năm 1908 và là tướng lĩnh.")
    vo_nguyen_giap = person("vng", "Võ Nguyên Giáp", "Võ Nguyên Giáp sinh năm 1911 và là đại tướng.")
    first_event = event("e1", "Hội nghị Mẫu 1946", "Hội nghị Mẫu diễn ra năm 1946.")
    second_event = event("e2", "Hội nghị Khác 1954", "Hội nghị Khác diễn ra năm 1954.")

    assert compare_subjects_compatible(nha_mac, nha_nguyen)
    assert compare_subjects_compatible(nguyen_binh, vo_nguyen_giap)
    assert compare_subjects_compatible(first_event, second_event)
    assert not compare_subjects_compatible(nha_mac, tam_phap_ty)


@pytest.mark.parametrize(
    ("title", "metadata", "text", "expected"),
    [
        ("Nhà Nguyễn", {"dynasties": ["Nhà Nguyễn"]}, "Nhà Nguyễn là một triều đại.", "dynasty"),
        ("Nhà Mạc", {"dynasties": ["Nhà Mạc"]}, "Nhà Mạc là một triều đại.", "dynasty"),
        (
            "Hoàng tộc nhà Nguyễn", {"dynasties": ["Nhà Nguyễn"]},
            "Hoàng tộc nhà Nguyễn gồm các thành viên hoàng gia.", "topic",
        ),
        (
            "Quan chế nhà Nguyễn", {"dynasties": ["Nhà Nguyễn"]},
            "Quan chế nhà Nguyễn quy định hệ thống quan lại.", "topic",
        ),
        (
            "Vạc đồng (nhà Nguyễn)", {"dynasties": ["Nhà Nguyễn"]},
            "Vạc đồng là các hiện vật được đúc dưới thời các chúa Nguyễn.", "topic",
        ),
        (
            "Hành cung nhà Nguyễn", {"dynasties": ["Nhà Nguyễn"]},
            "Hành cung nhà Nguyễn là các kiến trúc dành cho vua nghỉ lại.", "location",
        ),
        (
            "Thành nhà Hồ", {"dynasties": ["Nhà Hồ"], "locations": ["Thành nhà Hồ"]},
            "Thành nhà Hồ là một công trình thành lũy lịch sử.", "location",
        ),
        (
            "Công chúa nhà Đinh", {"dynasties": ["Nhà Đinh"]},
            "Công chúa nhà Đinh là tên gọi chung cho các công chúa của triều Đinh.", "topic",
        ),
        (
            "Hoàng hậu nhà Đinh", {"dynasties": ["Nhà Đinh"]},
            "Hoàng hậu nhà Đinh gồm năm hoàng hậu được sử sách ghi lại.", "topic",
        ),
        (
            "Mộc bản triều Nguyễn", {"dynasties": ["Nhà Nguyễn"]},
            "Mộc bản triều Nguyễn là những văn bản được khắc trên gỗ.", "document",
        ),
    ],
)
def test_whole_title_identity_precedes_contained_dynasty_metadata(
    title: str, metadata: dict, text: str, expected: str,
):
    assert classify_subject({"title": title, "text": text, "metadata": metadata}) == expected


def test_full_builder_resolves_identity_before_templates_and_candidate_selection(tmp_path: Path):
    mexico = {
        "chunk_id": "mexico-history",
        "title": "México",
        "text": (
            "México đối mặt với nhiều biến động sau cuộc cách mạng năm 1910. "
            "Đất nước México tái lập nền cộng hòa, và Benito Juárez trở thành tổng thống."
        ),
        "history_score": 24,
        "metadata": {
            "people": ["Benito Juárez"], "locations": ["Mexico"],
            "events": ["Cách mạng Mexico"], "years": [1867, 1910],
        },
    }
    nha_nguyen = {
        "chunk_id": "nha-nguyen", "title": "Nhà Nguyễn", "history_score": 50,
        "text": "Nhà Nguyễn là một triều đại bắt đầu năm 1802 và có vai trò lịch sử.",
        "metadata": {"dynasties": ["Nhà Nguyễn"], "years": [1802]},
    }
    nha_mac = {
        "chunk_id": "nha-mac", "title": "Nhà Mạc", "history_score": 50,
        "text": "Nhà Mạc là một triều đại bắt đầu năm 1527 và có vai trò lịch sử.",
        "metadata": {"dynasties": ["Nhà Mạc"], "years": [1527]},
    }
    ngo_xuan_lich = {
        "chunk_id": "ngo-xuan-lich", "title": "Ngô Xuân Lịch", "history_score": 40,
        "text": "Ngô Xuân Lịch sinh năm 1954, là một tướng lĩnh và giữ chức vụ lãnh đạo.",
        "metadata": {"people": ["Ngô Xuân Lịch"], "years": [1954]},
    }
    conference = {
        "chunk_id": "fontainebleau", "title": "Hội nghị Fontainebleau 1946",
        "text": "Hội nghị Fontainebleau 1946 diễn ra năm 1946 trong lịch sử ngoại giao.",
        "history_score": 40, "metadata": {"events": ["Hội nghị Fontainebleau 1946"], "years": [1946]},
    }
    rejected = [
        {
            "chunk_id": "royal-family", "title": "Hoàng tộc nhà Nguyễn", "history_score": 50,
            "text": "Hoàng tộc nhà Nguyễn gồm các thành viên hoàng gia vào thế kỷ XIX.",
            "metadata": {"dynasties": ["Nhà Nguyễn"], "years": [1830]},
        },
        {
            "chunk_id": "official-system", "title": "Quan chế nhà Nguyễn", "history_score": 50,
            "text": "Quan chế nhà Nguyễn quy định hệ thống quan lại từ năm 1804.",
            "metadata": {"dynasties": ["Nhà Nguyễn"], "years": [1804]},
        },
        {
            "chunk_id": "bronze-cauldrons", "title": "Vạc đồng (nhà Nguyễn)", "history_score": 50,
            "text": "Vạc đồng là các hiện vật được đúc vào năm 1631.",
            "metadata": {"dynasties": ["Nhà Nguyễn"], "years": [1631]},
        },
        {
            "chunk_id": "royal-palaces", "title": "Hành cung nhà Nguyễn", "history_score": 50,
            "text": "Hành cung nhà Nguyễn là các kiến trúc được xây dựng vào thế kỷ XIX.",
            "metadata": {"dynasties": ["Nhà Nguyễn"], "years": [1840]},
        },
        {
            "chunk_id": "ho-citadel", "title": "Thành nhà Hồ", "history_score": 50,
            "text": "Thành nhà Hồ là một công trình thành lũy xây dựng năm 1397.",
            "metadata": {"dynasties": ["Nhà Hồ"], "locations": ["Thành nhà Hồ"], "years": [1397]},
        },
    ]
    records = [*rejected, mexico, nha_nguyen, nha_mac, ngo_xuan_lich, conference]

    class Retriever:
        def __init__(self):
            self.queries: list[str] = []

        def search(self, query: str, *, top_k: int) -> list[dict]:
            self.queries.append(query)
            matches = [record for record in records if query.startswith(record["title"])]
            return sorted(matches, key=lambda record: len(record["title"]), reverse=True)[:top_k]

    retriever = Retriever()
    config = CustomBuildConfig(
        task_counts={"factual": 5, "compare": 1}, top_k=6, seed=73,
        max_corpus_records=len(records), max_candidate_attempts_per_task=50,
    )
    rows = list(build_custom_trajectories(
        write_corpus(tmp_path, records), retriever, config=config,
    ))
    factual_rows = [row for row in rows if row["task_type"] == "factual"]
    assert len(factual_rows) == 5
    assert not {record["title"] for record in rejected} & {
        row["provenance"]["primary_title"] for row in rows
    }
    assert all(not query.startswith(record["title"]) for query in retriever.queries for record in rejected)

    mexico_row = next(row for row in factual_rows if row["provenance"]["primary_title"] == "México")
    mexico_question = next(message["content"] for message in mexico_row["messages"] if message["role"] == "user")
    assert mexico_row["provenance"]["subject_type"] == "state"
    assert all(fragment not in mexico_question.casefold() for fragment in ("là ai", "cuộc đời", "nhân vật này"))
    factual_types = {
        row["provenance"]["primary_title"]: row["provenance"]["subject_type"]
        for row in factual_rows
    }
    assert factual_types == {
        "México": "state", "Nhà Nguyễn": "dynasty", "Nhà Mạc": "dynasty",
        "Ngô Xuân Lịch": "person", "Hội nghị Fontainebleau 1946": "event",
    }
    ngo_row = next(row for row in factual_rows if row["provenance"]["primary_title"] == "Ngô Xuân Lịch")
    ngo_question = next(message["content"] for message in ngo_row["messages"] if message["role"] == "user")
    assert "là ai" in ngo_question.casefold() and "nhân vật này" in ngo_question.casefold()
    conference_row = next(
        row for row in factual_rows
        if row["provenance"]["primary_title"] == "Hội nghị Fontainebleau 1946"
    )
    conference_question = next(
        message["content"] for message in conference_row["messages"] if message["role"] == "user"
    )
    assert all(fragment not in conference_question.casefold() for fragment in ("là ai", "cuộc đời"))

    compare_row = next(row for row in rows if row["task_type"] == "compare")
    assert {
        compare_row["provenance"]["primary_title"], compare_row["provenance"]["secondary_title"],
    } == {"Nhà Nguyễn", "Nhà Mạc"}
    assert {
        compare_row["provenance"]["subject_type"], compare_row["provenance"]["secondary_subject_type"],
    } == {"dynasty"}
    assert audit_rows(rows, strict_custom=True)["valid"]


def test_builder_skips_incompatible_compare_before_quota_and_finds_dynasty_peer(tmp_path: Path):
    nha_mac = {
        "chunk_id": "mac", "title": "Nhà Mạc",
        "text": "Nhà Mạc là một triều đại trị vì từ năm 1527 và có vai trò lịch sử.",
        "metadata": {"subject_type": "dynasty", "dynasties": ["Nhà Mạc"]},
    }
    tam_phap_ty = {
        "chunk_id": "tam-phap-ty", "title": "Tam Pháp Ty (nhà Nguyễn)",
        "text": "Tam Pháp Ty là cơ quan tư pháp được lập năm 1832 và có vai trò xét xử.",
        "metadata": {"subject_type": "dynasty", "dynasties": ["Nhà Nguyễn"]},
    }
    nha_nguyen = {
        "chunk_id": "nguyen", "title": "Nhà Nguyễn",
        "text": "Nhà Nguyễn là một triều đại trị vì từ năm 1802 và có vai trò lịch sử.",
        "metadata": {"subject_type": "dynasty", "dynasties": ["Nhà Nguyễn"]},
    }
    records = [nha_mac, tam_phap_ty, nha_nguyen]

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            matches = sorted(
                (record for record in records if record["title"] in query),
                key=lambda record: len(record["title"]), reverse=True,
            )
            return matches[:1]

    config = CustomBuildConfig(
        task_counts={"compare": 1}, top_k=6, seed=17, max_corpus_records=3,
        max_candidate_attempts_per_task=20,
    )
    row = list(build_custom_trajectories(
        write_corpus(tmp_path, records), Retriever(), config=config,
    ))[0]
    assert {
        row["provenance"]["primary_title"], row["provenance"]["secondary_title"],
    } == {"Nhà Mạc", "Nhà Nguyễn"}
    assert "Tam Pháp Ty (nhà Nguyễn)" not in {
        row["provenance"]["primary_title"], row["provenance"]["secondary_title"],
    }


@pytest.mark.parametrize(
    "bad_seed",
    [
        {
            "chunk_id": "asian-idol",
            "title": "Asian Idol",
            "text": "Asian Idol là một cuộc thi ca hát tổ chức năm 2007.",
            "history_score": 99,
            "metadata": {"subject_type": "person", "people": ["Asian Idol"], "years": [2007]},
        },
        {
            "chunk_id": "sea-games",
            "title": "Việt Nam tại Đại hội Thể thao Đông Nam Á 2007",
            "text": "Việt Nam tại Đại hội Thể thao Đông Nam Á 2007 có bảng tổng sắp huy chương.",
            "history_score": 99,
            "metadata": {"subject_type": "organization", "years": [2007]},
        },
    ],
)
def test_non_history_seed_is_skipped_before_quota_and_historical_event_is_emitted(
    tmp_path: Path, bad_seed: dict,
):
    summit = event(
        "summit",
        "Hội nghị thượng đỉnh Triều Tiên–Hoa Kỳ tại Hà Nội 2019",
        "Hội nghị thượng đỉnh Triều Tiên–Hoa Kỳ tại Hà Nội diễn ra vào năm 2019.",
    )

    class Retriever:
        def __init__(self):
            self.queries: list[str] = []

        def search(self, query: str, *, top_k: int) -> list[dict]:
            self.queries.append(query)
            return [summit]

    retriever = Retriever()
    row = list(build_custom_trajectories(
        write_corpus(tmp_path, [bad_seed, summit]), retriever, config=factual_config(),
    ))[0]
    assert row["provenance"]["primary_title"] == summit["title"]
    assert all(bad_seed["title"] not in query for query in retriever.queries)
    assert "lịch sử Việt Nam" not in row["messages"][0]["content"]
    assert "trả lời bằng tiếng Việt" in row["messages"][0]["content"]
    report = audit_rows([row], strict_custom=True)
    assert report["issues"].get("subject_type_mismatch", 0) == 0
    assert report["issues"].get("domain_mismatch", 0) == 0


def test_custom_history_scope_accepts_vietnamese_language_history_from_any_country():
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
        "text": "Không quân Nhân dân Triều Tiên được thành lập vào tháng 1 năm 1951 tại Triều Tiên.",
        "metadata": {"subject_type": "organization", "countries": ["Triều Tiên"]},
    }
    assert is_vietnam_history_relevant(nha_mac)
    assert is_vietnam_history_relevant(nguyen_binh)
    assert is_vietnam_history_relevant(border_war)
    assert is_vietnam_history_relevant(foreign_air_force)
    assert is_custom_history_eligible(foreign_air_force)
    assert not is_custom_history_eligible({
        "title": "Example Air Company",
        "text": "This is a modern commercial aviation company.",
        "metadata": {"subject_type": "organization"},
    })


@pytest.mark.parametrize(
    "row",
    [
        {"title": "Võ Trứ", "text": "Võ Trứ lãnh đạo một cuộc khởi nghĩa.", "metadata": {"subject_type": "person"}},
        {"title": "Dương Văn Hiếu", "text": "Dương Văn Hiếu là sĩ quan tình báo hoạt động trong thế kỷ XX.", "metadata": {"subject_type": "person"}},
        {"title": "Phaolô Nguyễn Văn Bình", "text": "Phaolô Nguyễn Văn Bình sinh năm 1910 và là tổng giám mục.", "metadata": {"subject_type": "person"}},
        {"title": "Po Saong Nyung Ceng", "text": "Po Saong Nyung Ceng lãnh đạo nhiều hoạt động trong lịch sử người Chăm.", "metadata": {"subject_type": "person"}},
        {"title": "Chiến tranh Đại Việt–Khmer", "text": "Chiến tranh Đại Việt–Khmer diễn ra trong thế kỷ XII.", "metadata": {"subject_type": "event"}},
        {"title": "Chiến tranh biên giới Việt Nam – Campuchia", "text": "Việt Nam là một chủ thể trực tiếp trong cuộc chiến.", "metadata": {"subject_type": "event"}},
        {"title": "Panduranga", "text": "Panduranga là một chủ thể trong lịch sử Chăm Pa và Đại Việt.", "metadata": {"subject_type": "state"}},
        {"title": "Pháp", "text": "Pháp thiết lập chế độ thuộc địa tại Việt Nam trong thế kỷ XIX.", "metadata": {"subject_type": "state"}},
    ],
)
def test_custom_history_scope_accepts_vietnamese_and_regionally_connected_history(row: dict):
    assert is_vietnam_history_relevant(row)


def test_compare_allows_foreign_vietnamese_language_history_targets(tmp_path: Path):
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
    pair = {
        built[0]["provenance"]["primary_title"],
        built[0]["provenance"]["secondary_title"],
    }
    assert "Kim Mẫu" in pair
    assert pair <= {"Kim Mẫu", "Nguyễn Bình", "Võ Trứ"}


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


@pytest.mark.parametrize(
    ("target_title", "target_text"),
    [
        ("Ngô Nhĩ Khai Hy", "Ngô Nhĩ Khai Hy là nhà lãnh đạo và hoạt động từ năm 1989."),
        ("Ngô Xuân Lịch", "Ngô Xuân Lịch sinh năm 1954 và giữ vai trò tướng lĩnh."),
    ],
)
def test_full_builder_rejects_surname_article_for_multi_token_person(
    tmp_path: Path, target_title: str, target_text: str,
):
    target = person(
        "exact-target", target_title, target_text,
    )
    surname_article = {
        "chunk_id": "ngo-surname",
        "title": "Ngô (họ)",
        "text": (
            "Ngô Ngạn Tổ là diễn viên, Ngô Đôn Nghĩa là chính trị gia và Andrew Ng "
            "là một nhà nghiên cứu khoa học máy tính."
        ),
        "metadata": {"people": ["Ngô Thái Bá", "Ngô Thanh Nguyên"]},
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [target, surname_article][:top_k]

    row = list(build_custom_trajectories(
        write_corpus(tmp_path, [target]), Retriever(), config=factual_config(),
    ))[0]
    payload = json.loads(next(
        message["content"] for message in row["messages"] if message["role"] == "tool"
    ))
    assert [result["chunk_id"] for result in payload] == ["exact-target"]
    assert "Ngô Ngạn Tổ" not in row["messages"][-1]["content"]


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
        (
            "Bằng cách này, Triều Tiên cố gắng để duy trì năng lực quân sự tương đương với Hàn Quốc "
            "bằng cách sử dụng không quân như một lực lượng ngăn chặn, thay vì duy trì công nghệ tương đương."
        ),
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


def test_cause_facet_rejects_decree_quote_and_authorship_as_background():
    title = "Chiếu thoái vị của Bảo Đại"
    plan = QueryPlan("search_history", f"{title} bối cảnh nguyên nhân", 4, "context_cause")
    quotation = {
        "chunk_id": "decree-quote",
        "title": title,
        "text": (
            "Trong non bốn thế kỷ, Liệt Thánh đã trải qua gian lao nguy hiểm, vì nước vì dân. "
            "Câu nói này ở trong Chiếu thoái vị của Bảo Đại và chiếu thư là do Phạm Khắc Hòe soạn."
        ),
        "metadata": {"subject_type": "document"},
    }
    compact = compact_observation(
        [quotation], plan, task_type="cause", observation_char_budget=2_000,
        max_result_text_chars=500, target_title=title, target_subject_type="document",
    )
    assert compact == []


def test_cause_facet_accepts_actual_background_leading_to_decree():
    title = "Chiếu thoái vị của Bảo Đại"
    plan = QueryPlan("search_history", f"{title} bối cảnh nguyên nhân", 4, "context_cause")
    background = {
        "chunk_id": "decree-background",
        "title": title,
        "text": (
            "Thắng lợi của Cách mạng tháng Tám tạo nên tình thế chính trị mới, "
            "dẫn đến việc ban hành Chiếu thoái vị của Bảo Đại."
        ),
        "metadata": {"subject_type": "document"},
    }
    compact = compact_observation(
        [background], plan, task_type="cause", observation_char_budget=2_000,
        max_result_text_chars=500, target_title=title, target_subject_type="document",
    )
    assert [result["chunk_id"] for result in compact] == ["decree-background"]


def test_full_builder_rejects_decree_wording_as_cause_after_one_fallback(tmp_path: Path):
    title = "Chiếu thoái vị của Bảo Đại"
    distant_context = " ".join(["lời chiếu nói về quyết định thoái vị"] * 70)
    decree_wording = {
        "chunk_id": "decree-wording",
        "title": title,
        "text": (
            "Trong non bốn thế kỷ, Liệt Thánh chúng ta đã trải qua biết bao sự gian lao nguy hiểm, "
            "vì nước vì dân mới truyền ngôi lại cho Trẫm được đến ngày nay. "
            "Nay Trẫm nhất định thoái vị để giao vận mạng quốc gia cho một chính phủ mới "
            f"{distant_context} bối cảnh lịch sử được nhắc đến ở cuối đoạn"
        ),
        "history_score": 48,
        "metadata": {"dynasties": ["Nhà Nguyễn"], "years": [1945]},
    }

    class Retriever:
        def __init__(self):
            self.queries: list[str] = []

        def search(self, query: str, *, top_k: int) -> list[dict]:
            self.queries.append(query)
            return [decree_wording]

    retriever = Retriever()
    config = CustomBuildConfig(
        task_counts={"cause": 1}, top_k=6, seed=19, max_corpus_records=1,
        max_candidate_attempts_per_task=10,
    )
    with pytest.raises(ValueError, match=r"0/1 cause trajectories"):
        list(build_custom_trajectories(
            write_corpus(tmp_path, [decree_wording]), retriever, config=config,
        ))
    assert retriever.queries == [
        f"{title} bối cảnh nguyên nhân điều kiện hình thành",
        f"{title} lịch sử",
    ]


def test_full_builder_selects_formation_sentence_and_rejects_late_strategy(tmp_path: Path):
    title = "Không quân Nhân dân Triều Tiên"
    seed = {
        "chunk_id": "korean-air-force-seed",
        "title": title,
        "text": f"Tháng 1 năm 1951, Bộ tư lệnh {title} được thành lập tại Triều Tiên.",
        "metadata": {"subject_type": "organization"},
    }
    formation = {
        **seed,
        "chunk_id": "formation",
        "text": f"Tháng 1 năm 1951, Bộ tư lệnh {title} được thành lập.",
    }
    late_strategy = {
        **seed,
        "chunk_id": "late-strategy",
        "text": (
            "Bằng cách này, Triều Tiên cố gắng để duy trì năng lực quân sự tương đương với Hàn Quốc "
            "bằng cách sử dụng không quân như một lực lượng ngăn chặn, thay vì duy trì công nghệ tương đương."
        ),
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [formation, late_strategy][:top_k]

    config = CustomBuildConfig(task_counts={"cause": 1}, seed=17, max_corpus_records=1)
    row = list(build_custom_trajectories(
        write_corpus(tmp_path, [seed]), Retriever(), config=config,
    ))[0]
    evidence = [
        result
        for message in row["messages"] if message["role"] == "tool"
        for result in json.loads(message["content"])
    ]
    assert [result["chunk_id"] for result in evidence] == ["formation"]
    assert "lực lượng ngăn chặn" not in row["messages"][-1]["content"]


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


def test_significance_filter_remains_moderate_for_historically_relevant_impact_context():
    title = "Không quân Nhân dân Triều Tiên"
    text = (
        "Là một nhánh quân chủng kỹ thuật cao, lực lượng Phòng không Không quân Nhân dân "
        "Triều Tiên chịu tác động nặng nề của sự suy giảm kinh tế."
    )
    plan = QueryPlan("search_history", f"{title} ý nghĩa tác động", 4, "result_significance")
    compact = compact_observation(
        [{"chunk_id": "impact-context", "title": title, "text": text, "metadata": {"subject_type": "organization"}}],
        plan, task_type="significance", observation_char_budget=2_000, max_result_text_chars=500,
        target_title=title, target_subject_type="organization",
    )
    assert [result["chunk_id"] for result in compact] == ["impact-context"]


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


def test_significance_facet_rejects_plain_sports_scoring_rule():
    title = "Giải bóng đá Vô địch U-21 Quốc gia 2015"
    plan = QueryPlan("search_history", f"{title} ý nghĩa tác động", 4, "result_significance")
    result = {
        "chunk_id": "sports-rule",
        "title": title,
        "text": f"Tại {title}, mỗi trận thắng được 3 điểm, hòa 1 điểm và thua không có điểm.",
        "metadata": {"subject_type": "event"},
    }
    assert compact_observation(
        [result], plan, task_type="significance", observation_char_budget=2_000,
        max_result_text_chars=500, target_title=title, target_subject_type="event",
    ) == []


UNIVERSITY_TITLE = "Trường Đại học Kiến trúc Thành phố Hồ Chí Minh"


def university_significance_fixtures() -> tuple[dict, dict, dict, dict]:
    seed = {
        "chunk_id": "university-seed",
        "title": UNIVERSITY_TITLE,
        "text": (
            f"{UNIVERSITY_TITLE} được thành lập năm 1976 trong bối cảnh tái thiết. "
            "Quá trình hình thành của trường gắn với nhu cầu đào tạo kiến trúc sư."
        ),
        "metadata": {"subject_type": "organization", "years": [1976]},
    }
    context = {
        **seed,
        "chunk_id": "university-context",
        "text": f"{UNIVERSITY_TITLE} được thành lập năm 1976 trong bối cảnh tái thiết đất nước.",
    }
    bad_catalog = {
        "chunk_id": "admission-catalog",
        "title": "Đánh giá năng lực",
        "text": (
            "Thành phố Hồ Chí Minh Trường Đại học Giao thông Vận tải Trường Đại học Hoa Sen "
            "Trường Đại học Hùng Vương Thành phố Hồ Chí Minh Trường Đại học Kiên Giang "
            "Trường Đại học Kiến trúc Đà Nẵng "
            f"{UNIVERSITY_TITLE} "
            "Trường Đại học Kinh tế Quốc dân Trường Đại học Kinh tế Thành phố Hồ Chí Minh "
            "Trường Đại học Kinh tế Công nghiệp Long An Trường Đại học Kỹ thuật Công nghệ Cần Thơ "
            "Trường Đại học Khánh Hòa Trường Đại học Lạc Hồng Trường Đại học Lâm nghiệp "
            "Trường Đại học Nam Cần Thơ Trường Đại học Nông Lâm Thành phố Hồ Chí Minh "
            "Danh sách các đơn vị sử dụng kết quả kỳ thi để xét tuyển."
        ),
        "metadata": {"content_facets": ["kết quả"]},
    }
    meaningful = {
        "chunk_id": "university-contribution",
        "title": UNIVERSITY_TITLE,
        "text": (
            "Bên cạnh đào tạo, trường còn là trung tâm nghiên cứu, cố vấn, "
            "thực hiện các dự án cho doanh nghiệp và Chính phủ Việt Nam."
        ),
        "metadata": {"subject_type": "organization"},
    }
    return seed, context, bad_catalog, meaningful


@pytest.mark.parametrize("task_type", ["significance", "summary", "multihop"])
def test_builder_significance_paths_reject_catalog_and_keep_meaningful_evidence(
    tmp_path: Path, task_type: str,
):
    seed, context, bad_catalog, meaningful = university_significance_fixtures()

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            if "bối cảnh" in query or "nguyên nhân" in query:
                return [context]
            return [bad_catalog, meaningful]

    config = CustomBuildConfig(
        task_counts={task_type: 1}, top_k=6, seed=41, max_corpus_records=1,
        max_candidate_attempts_per_task=10,
    )
    row = list(build_custom_trajectories(
        write_corpus(tmp_path, [seed]), Retriever(), config=config,
    ))[0]
    queries = row["provenance"]["retrieval_queries"]
    payloads = [
        json.loads(message["content"])
        for message in row["messages"] if message["role"] == "tool"
    ]
    significance_payload = next(
        payload for query, payload in zip(queries, payloads)
        if query["role"] == "result_significance"
    )
    assert [result["chunk_id"] for result in significance_payload] == ["university-contribution"]
    assert "Trường Đại học Hoa Sen" not in row["messages"][-1]["content"]
    assert "trung tâm nghiên cứu" in row["messages"][-1]["content"]
    assert audit_rows([row], strict_custom=True)["valid"]


@pytest.mark.parametrize("task_type", ["significance", "summary", "multihop"])
def test_catalog_only_significance_fallback_once_then_skips_candidate(
    tmp_path: Path, task_type: str,
):
    seed, context, bad_catalog, _ = university_significance_fixtures()

    class Retriever:
        def __init__(self):
            self.queries: list[str] = []

        def search(self, query: str, *, top_k: int) -> list[dict]:
            self.queries.append(query)
            if "bối cảnh" in query or "nguyên nhân" in query:
                return [context]
            return [bad_catalog]

    retriever = Retriever()
    config = CustomBuildConfig(
        task_counts={task_type: 1}, top_k=6, seed=41, max_corpus_records=1,
        max_candidate_attempts_per_task=10,
    )
    with pytest.raises(ValueError, match=rf"0/1 {task_type}"):
        list(build_custom_trajectories(
            write_corpus(tmp_path, [seed]), retriever, config=config,
        ))
    assert retriever.queries.count(f"{UNIVERSITY_TITLE} lịch sử") == 1
    assert len(retriever.queries) == (2 if task_type == "significance" else 3)


def test_significance_fallback_reapplies_filter_and_skips_only_inventory_candidate(tmp_path: Path):
    title = "Không quân Nhân dân Triều Tiên"
    seed = {
        "chunk_id": "korean-air-force-seed",
        "title": title,
        "text": f"{title} được thành lập vào tháng 1 năm 1951 tại Triều Tiên.",
        "metadata": {"subject_type": "organization"},
    }
    inventory = {
        **seed,
        "chunk_id": "mig-inventory",
        "text": f"{title} cũng có một ít máy bay MiG-29 hiện đại hơn.",
    }

    class Retriever:
        def __init__(self):
            self.queries: list[str] = []

        def search(self, query: str, *, top_k: int) -> list[dict]:
            self.queries.append(query)
            return [inventory]

    retriever = Retriever()
    config = CustomBuildConfig(task_counts={"significance": 1}, seed=17, max_corpus_records=1)
    with pytest.raises(ValueError, match="0/1 significance"):
        list(build_custom_trajectories(
            write_corpus(tmp_path, [seed]), retriever, config=config,
        ))
    assert len(retriever.queries) == 2
    assert retriever.queries[-1] == f"{title} lịch sử"


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


def test_foreign_vietnamese_language_history_subject_is_emitted_at_builder_level(tmp_path: Path):
    foreign = {
        "chunk_id": "foreign-air-force",
        "title": "Không quân Nhân dân Triều Tiên",
        "text": "Tháng 1 năm 1951, Bộ tư lệnh Không quân Nhân dân Triều Tiên được thành lập tại Triều Tiên.",
        "metadata": {"subject_type": "organization", "countries": ["Triều Tiên"]},
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [foreign][:top_k]

    config = CustomBuildConfig(task_counts={"cause": 1}, seed=13, max_corpus_records=1)
    built = list(build_custom_trajectories(
        write_corpus(tmp_path, [foreign]), Retriever(), config=config,
    ))
    assert len(built) == 1
    assert built[0]["provenance"]["primary_title"] == "Không quân Nhân dân Triều Tiên"
    assert built[0]["provenance"]["custom_history_eligible"] is True
    report = audit_rows(built, strict_custom=True)
    assert report["issues"].get("domain_mismatch", 0) == 0
    assert report["valid"]


def test_strict_audit_catches_organization_as_dynasty_and_missing_history_evidence():
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


def test_strict_audit_flags_incompatible_dynasty_institution_compare():
    row = make_trajectory(
        trajectory_id="bad-compare-types",
        source_dataset="custom_history",
        task_type="compare",
        messages=[
            {"role": "user", "content": "So sánh Nhà Mạc và Tam Pháp Ty."},
            {"role": "assistant", "content": "Chưa đủ bằng chứng."},
        ],
        tools=[],
        difficulty="medium",
        provenance={
            "subject_type": "dynasty", "primary_title": "Nhà Mạc",
            "secondary_subject_type": "dynasty",
            "secondary_title": "Tam Pháp Ty (nhà Nguyễn)",
            "source_group": "bad-pair", "requires_final_answer": True,
        },
    )
    report = audit_rows([row], strict_custom=True)
    assert report["issues"]["subject_type_mismatch"] == 1
    assert report["issues"]["compare_type_mismatch"] == 1
    assert not report["valid"]


def test_strict_audit_flags_surname_only_person_observation_even_with_metadata(tmp_path: Path):
    target = person(
        "ngo-xuan-lich", "Ngô Xuân Lịch",
        "Ngô Xuân Lịch sinh năm 1954 và giữ vai trò tướng lĩnh.",
    )

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [target]

    corrupted = list(build_custom_trajectories(
        write_corpus(tmp_path, [target]), Retriever(), config=factual_config(),
    ))[0]
    surname = {
        "chunk_id": "ngo-surname", "title": "Ngô (họ)",
        "text": "Ngô Giáp Đậu, Ngô Đức Kế, Ngô Gia Tự và Ngô Đình Diệm.",
        "metadata": {"people": ["Ngô Xuân Lịch"]},
        "retrieval_role": "factual",
    }
    tool_message = next(message for message in corrupted["messages"] if message["role"] == "tool")
    tool_message["content"] = json.dumps([surname], ensure_ascii=False)
    corrupted["messages"][-1]["content"] = "Danh sách người mang họ Ngô. [ngo-surname]"
    corrupted["provenance"]["observed_evidence_ids"] = ["ngo-surname"]
    corrupted["provenance"]["evidence_ids"] = ["ngo-surname"]

    report = audit_rows([corrupted], strict_custom=True)
    assert report["issues"]["observation_target_mismatch"] == 1
    assert report["issues"]["final_answer_target_mismatch"] == 1
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


def test_nested_state_and_institution_compare_is_valid_but_true_contamination_is_detected(tmp_path: Path):
    state = {
        "chunk_id": "vnch",
        "title": "Việt Nam Cộng hòa",
        "text": "Việt Nam Cộng hòa tồn tại từ năm 1955 đến năm 1975.",
        "metadata": {"subject_type": "state", "states": ["Việt Nam Cộng hòa"]},
    }
    artillery = {
        "chunk_id": "artillery",
        "title": "Binh chủng Pháo binh Việt Nam Cộng hòa",
        "text": (
            "Binh chủng Pháo binh Việt Nam Cộng hòa được thành lập trong thập niên 1950 "
            "và là lực lượng hỏa lực của quân đội."
        ),
        "metadata": {
            "subject_type": "organization",
            "organizations": ["Binh chủng Pháo binh Việt Nam Cộng hòa"],
        },
    }

    class Retriever:
        def search(self, query: str, *, top_k: int) -> list[dict]:
            return [artillery] if artillery["title"] in query else [state]

    config = CustomBuildConfig(task_counts={"compare": 1}, seed=23, max_corpus_records=2)
    row = list(build_custom_trajectories(
        write_corpus(tmp_path, [state, artillery]), Retriever(), config=config,
    ))[0]
    assert {
        row["provenance"]["subject_type"],
        row["provenance"]["secondary_subject_type"],
    } == {"state", "organization"}
    report = audit_rows([row], strict_custom=True)
    assert report["issues"].get("compare_target_contamination", 0) == 0
    assert report["issues"].get("compare_type_mismatch", 0) == 0

    corrupted = copy.deepcopy(row)
    role_index = next(
        index
        for index, query in enumerate(corrupted["provenance"]["retrieval_queries"])
        if query["role"] == "target_b"
    )
    tool_messages = [message for message in corrupted["messages"] if message["role"] == "tool"]
    assigned_title = corrupted["provenance"]["secondary_title"]
    wrong = state if assigned_title == artillery["title"] else artillery
    tool_messages[role_index]["content"] = json.dumps([wrong], ensure_ascii=False)
    corrupted_report = audit_rows([corrupted], strict_custom=True)
    assert corrupted_report["issues"]["compare_target_contamination"] == 1
    assert not corrupted_report["valid"]


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
        person("a", "Nguyễn Bình", "Nguyễn Bình sinh năm 1908, là tướng lĩnh chỉ huy tại Nam Bộ."),
        person("b", "Phaolô Nguyễn Văn Bình", "Phaolô Nguyễn Văn Bình sinh năm 1910 và là tổng giám mục."),
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
