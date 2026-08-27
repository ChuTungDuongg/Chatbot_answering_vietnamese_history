from __future__ import annotations

from training.evidence_agent.coverage import (
    assess_answer_coverage,
    evidence_relevance,
    specific_missing_information,
)
from training.evidence_agent.prepare_dataset import _base_record, _propose_partial


def source_row(question: str, evidence_text: str, answer: str) -> dict:
    return {
        "id": "semantic-fixture",
        "messages": [
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nTài liệu tham khảo:\n[ev_01] Nguồn kiểm thử\n{evidence_text}",
            },
            {
                "role": "assistant",
                "content": f"Nguồn được dùng: [ev_01]\n\nTrả lời: {answer}",
            },
        ],
    }


def test_one_chunk_can_fully_cover_style_and_courtesy_name():
    question = "Phan Đình Phùng có hiệu và tự là gì?"
    answer = "Theo tài liệu, Phan Đình Phùng hiệu Châu Phong, tự Tôn Cát."
    evidence = "Phan Đình Phùng có hiệu Châu Phong, tự Tôn Cát."
    assessment = assess_answer_coverage(question, answer, [evidence])
    assert assessment.full
    assert assessment.supported_keys == ("style_name", "courtesy_name")


def test_one_chunk_can_fully_cover_leader_and_opponent():
    question = "Khởi nghĩa Hương Khê do ai lãnh đạo và chống lực lượng nào?"
    answer = "Khởi nghĩa Hương Khê do Phan Đình Phùng lãnh đạo, chống lại thực dân Pháp."
    evidence = "Phan Đình Phùng là lãnh đạo khởi nghĩa Hương Khê chống lại thực dân Pháp."
    assert assess_answer_coverage(question, answer, [evidence]).full


def test_true_partial_keeps_date_but_reports_missing_leader():
    question = "Khởi nghĩa ở Huế thắng lợi ngày nào và ai chỉ đạo?"
    answer = "Ngày 23/8, khởi nghĩa ở Huế giành thắng lợi. Việc chỉ đạo có Nguyễn Chí Thanh và Tố Hữu."
    evidence = "Ngày 23/8, khởi nghĩa ở Huế giành thắng lợi."
    assessment = assess_answer_coverage(question, answer, [evidence])
    assert assessment.partial
    assert assessment.supported_keys == ("time",)
    assert assessment.missing_keys == ("person_leader",)
    assert "người chỉ đạo" in specific_missing_information(assessment)[0]


def test_completely_irrelevant_history_evidence_is_not_useful_partial():
    question = "Chỉ thị Nhật - Pháp bắn nhau ban hành khi nào và nhằm mục đích gì?"
    answer = "Chỉ thị được ban hành ngày 12/3/1945 nhằm phát động cao trào kháng Nhật cứu nước."
    evidence = "Ngày 22/12/1944, Võ Nguyên Giáp thành lập Việt Nam Tuyên truyền Giải phóng quân."
    assessment = assess_answer_coverage(question, answer, [evidence])
    assert not assessment.useful
    assert not assessment.full


def test_irrelevant_only_source_target_selects_nothing():
    question = "Chỉ thị Nhật - Pháp bắn nhau ban hành khi nào và nhằm mục đích gì?"
    evidence = "Ngày 22/12/1944, Võ Nguyên Giáp thành lập Việt Nam Tuyên truyền Giải phóng quân."
    row = source_row(question, evidence, "Evidence không đủ để trả lời câu hỏi.")
    row["messages"][1]["content"] = "Nguồn được dùng:\n\nTrả lời: Evidence không đủ để trả lời câu hỏi."
    record = _base_record(row, compression_max_chars=600)
    assert record["output"].status == "insufficient"
    assert record["output"].selected_evidence == []


def test_partial_augmentation_requires_verified_component_loss():
    question = "Khởi nghĩa ở Huế thắng lợi ngày nào và ai chỉ đạo?"
    evidence = (
        "Ngày 23/8, khởi nghĩa ở Huế giành thắng lợi. "
        "Việc chỉ đạo khởi nghĩa có Nguyễn Chí Thanh và Tố Hữu."
    )
    answer = "Ngày 23/8, khởi nghĩa ở Huế giành thắng lợi. Việc chỉ đạo khởi nghĩa có Nguyễn Chí Thanh và Tố Hữu."
    record = _base_record(source_row(question, evidence, answer), compression_max_chars=600)
    proposal, audit = _propose_partial(record)
    assert proposal is not None
    assert audit["audit_outcome"] == "remain_true_partial"
    assert proposal[1].status == "insufficient"
    assert any("người chỉ đạo" in item for item in proposal[1].missing_information)


def test_full_single_claim_is_reclassified_instead_of_forced_partial():
    question = "Phan Đình Phùng có hiệu và tự là gì?"
    evidence = "Phan Đình Phùng có hiệu Châu Phong, tự Tôn Cát và lãnh đạo khởi nghĩa Hương Khê."
    answer = "Phan Đình Phùng hiệu Châu Phong, tự Tôn Cát."
    record = _base_record(source_row(question, evidence, answer), compression_max_chars=600)
    proposal, audit = _propose_partial(record)
    assert proposal is None
    assert audit["audit_outcome"] == "reclassified_to_sufficient"


def test_relevance_is_question_based_not_a_partial_status_constant():
    question = "Khởi nghĩa ở Huế thắng lợi ngày nào và ai chỉ đạo?"
    partial_text = "Ngày 23/8, khởi nghĩa ở Huế giành thắng lợi."
    unrelated_text = "Nhà Trần chống quân Nguyên trên sông Bạch Đằng."
    partial_relevance = evidence_relevance(question, partial_text)
    assert partial_relevance > 0.75
    assert partial_relevance != 0.75
    assert partial_relevance > evidence_relevance(question, unrelated_text)
