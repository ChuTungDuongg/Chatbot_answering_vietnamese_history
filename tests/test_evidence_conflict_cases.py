from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.schemas import EvidenceModelOutput
from app.agents.schemas import SelectedEvidence
from tests.evidence_v2_fixtures import sanity_rows
from training.evidence_agent.conflicts import (
    conflict_values_incompatible,
    propose_question_relevant_conflict,
)


def test_conflict_fixture_names_both_existing_evidence_ids():
    row = next(item for item in sanity_rows() if item["id"] == "case-e")
    conflict = row["output"]["conflicts"][0]
    assert row["output"]["status"] == "conflicting"
    assert "ev_41" in conflict and "ev_42" in conflict


def test_conflicting_status_requires_a_conflict_description():
    with pytest.raises(ValidationError, match="conflicting output requires"):
        EvidenceModelOutput(
            status="conflicting",
            selected_evidence=[],
            conflicts=[],
            missing_information=[],
            summary="Có mâu thuẫn.",
        )


def _selected(evidence_id: str, claim: str) -> SelectedEvidence:
    return SelectedEvidence(
        evidence_id=evidence_id,
        relevance=1.0,
        claims=[claim],
        compressed_text=claim,
    )


def test_date_conflict_targets_the_requested_answer_slot():
    true_claim = "Hồ Chí Minh đọc Tuyên ngôn Độc lập ngày 2/9/1945."
    proposal = propose_question_relevant_conflict(
        question="Hồ Chí Minh đọc Tuyên ngôn Độc lập vào ngày nào?",
        gold_answer=true_claim,
        selected=[_selected("ev_a", true_claim)],
        evidence_texts=[true_claim],
    )
    assert proposal is not None
    assert proposal[1].conflict_type == "date"
    assert "3/9/1945" in proposal[1].mutated_claim


def test_same_date_paraphrases_are_compatible_not_conflicting():
    assert not conflict_values_incompatible(
        "2 tháng 9 năm 1945", "2/9/1945", "date"
    )


@pytest.mark.parametrize(
    ("question", "gold", "first", "second"),
    [
        (
            "Những di tích tưởng niệm Ngô Quyền tập trung nhiều ở khu vực nào?",
            "Các di tích tập trung nhiều nhất ở Hải Phòng.",
            "Các di tích tập trung nhiều nhất ở Hải Phòng; chiến thắng năm 938.",
            "Các di tích tập trung nhiều nhất ở Hải Phòng; chiến thắng năm 939.",
        ),
        (
            "Bạch Đằng năm 1288 khác Bạch Đằng năm 938 ở nhân vật và đối thủ như thế nào?",
            "Năm 938 gắn với Ngô Quyền đánh quân Nam Hán; năm 1288 gắn với Trần Hưng Đạo đánh quân Nguyên.",
            "Ngô Quyền sinh năm 898 và chỉ huy chiến thắng Bạch Đằng năm 938.",
            "Ngô Quyền sinh năm 899 và chỉ huy chiến thắng Bạch Đằng năm 938.",
        ),
    ],
)
def test_irrelevant_disagreement_does_not_create_conflict(question, gold, first, second):
    proposal = propose_question_relevant_conflict(
        question=question,
        gold_answer=gold,
        selected=[_selected("ev_a", first)],
        evidence_texts=[first, second],
    )
    assert proposal is None


def test_person_conflict_is_generated_for_who_slot():
    first = "Nguyễn Văn An lãnh đạo sự kiện X."
    second = "Trần Văn Bình lãnh đạo sự kiện X."
    proposal = propose_question_relevant_conflict(
        question="Ai lãnh đạo sự kiện X?",
        gold_answer=first,
        selected=[_selected("ev_a", first)],
        evidence_texts=[first, second],
    )
    assert proposal is not None
    assert proposal[1].conflict_type == "person"
    assert proposal[1].original_value == "Nguyễn Văn An"
    assert proposal[1].mutated_value == "Trần Văn Bình"


def test_location_conflict_is_generated_for_where_slot():
    first = "Sự kiện X diễn ra tại Hà Nội."
    second = "Sự kiện X diễn ra tại Huế."
    proposal = propose_question_relevant_conflict(
        question="Sự kiện X diễn ra ở đâu?",
        gold_answer=first,
        selected=[_selected("ev_a", first)],
        evidence_texts=[first, second],
    )
    assert proposal is not None
    assert proposal[1].conflict_type == "location"
    assert proposal[1].original_value == "Hà Nội"
    assert proposal[1].mutated_value == "Huế"
