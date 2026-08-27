from __future__ import annotations

import json
from typing import Any

from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceModelOutput


def candidate(evidence_id: str, text: str, *, source_type: str = "local") -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "source_type": source_type,
        "title": None,
        "url": None,
        "chunk_id": evidence_id,
        "text": text,
        "retrieval_score": None,
    }


def dataset_row(
    row_id: str,
    behavior: str,
    question: str,
    evidence: list[dict[str, Any]],
    output: dict[str, Any],
    *,
    group_id: str | None = None,
) -> dict[str, Any]:
    request = EvidenceAgentRequest(question=question, max_selected=8, evidence=evidence)
    target = EvidenceModelOutput.model_validate(output)
    request_payload = request.model_dump()
    output_payload = target.model_dump()
    return {
        "id": row_id,
        "source_dataset": "tiny_fixture",
        "original_sample_id": row_id,
        "group_id": group_id or f"group-{row_id}",
        "behavior": behavior,
        "question": question,
        "evidence": [item.model_dump() for item in request.evidence],
        "input": request_payload,
        "output": output_payload,
        "metadata": {"synthetic": behavior in {"duplicate", "conflict", "partial"}},
        "messages": [
            {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
            {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(output_payload, ensure_ascii=False, sort_keys=True)},
        ],
    }


def sanity_rows() -> list[dict[str, Any]]:
    relevant = "Năm 938, Ngô Quyền chỉ huy quân dân đánh bại quân Nam Hán trên sông Bạch Đằng."
    distractor = "Năm 1288, Trần Hưng Đạo chỉ huy quân Đại Việt chống quân Nguyên trên sông Bạch Đằng."
    a = dataset_row(
        "case-a",
        "relevant_distractor",
        "Ai chỉ huy chiến thắng Bạch Đằng năm 938?",
        [candidate("ev_01", relevant), candidate("ev_02", distractor)],
        {
            "status": "sufficient",
            "selected_evidence": [{"evidence_id": "ev_01", "relevance": 1.0, "claims": [relevant], "compressed_text": relevant}],
            "conflicts": [],
            "missing_information": [],
            "summary": "Evidence ev_01 xác định Ngô Quyền chỉ huy chiến thắng Bạch Đằng năm 938.",
        },
    )
    dbp = "Chiến dịch Điện Biên Phủ kết thúc thắng lợi ngày 7/5/1954 và Pháp chịu thất bại quân sự lớn."
    geneva = "Hội nghị Genève năm 1954 bàn về lập lại hòa bình ở Đông Dương."
    b = dataset_row(
        "case-b",
        "clean_relevant",
        "Điện Biên Phủ và Hội nghị Genève năm 1954 có những dữ kiện liên quan nào?",
        [candidate("ev_11", dbp), candidate("ev_12", geneva)],
        {
            "status": "sufficient",
            "selected_evidence": [
                {"evidence_id": "ev_11", "relevance": 1.0, "claims": [dbp], "compressed_text": dbp},
                {"evidence_id": "ev_12", "relevance": 1.0, "claims": [geneva], "compressed_text": geneva},
            ],
            "conflicts": [],
            "missing_information": [],
            "summary": "Hai evidence cung cấp dữ kiện về Điện Biên Phủ và Hội nghị Genève năm 1954.",
        },
    )
    c = dataset_row(
        "case-c",
        "insufficient",
        "Ngô Quyền áp dụng chính sách thuế cụ thể nào?",
        [candidate("ev_21", "Ngô Quyền xưng vương năm 939 và đóng đô tại Cổ Loa.")],
        {
            "status": "insufficient",
            "selected_evidence": [],
            "conflicts": [],
            "missing_information": ["Không có evidence về chính sách thuế cụ thể."],
            "summary": "Evidence chỉ nêu việc xưng vương và đóng đô, không nêu chính sách thuế.",
        },
    )
    proclamation = "Ngày 2/9/1945, Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Ba Đình."
    d = dataset_row(
        "case-d",
        "duplicate",
        "Hồ Chí Minh đọc Tuyên ngôn Độc lập khi nào và ở đâu?",
        [
            candidate("ev_31", proclamation),
            candidate("ev_32", proclamation, source_type="synthetic_duplicate"),
            candidate("ev_33", "Chiến dịch Điện Biên Phủ kết thúc ngày 7/5/1954."),
        ],
        {
            "status": "sufficient",
            "selected_evidence": [{"evidence_id": "ev_31", "relevance": 1.0, "claims": [proclamation], "compressed_text": proclamation}],
            "conflicts": [],
            "missing_information": [],
            "summary": "Evidence ev_31 đủ; ev_32 lặp cùng fact và không cần giữ lại.",
        },
    )
    true_date = "Hồ Chí Minh đọc Tuyên ngôn Độc lập ngày 2/9/1945 tại Ba Đình."
    corrupt_date = "Hồ Chí Minh đọc Tuyên ngôn Độc lập ngày 3/9/1945 tại Ba Đình."
    e = dataset_row(
        "case-e",
        "conflict",
        "Hồ Chí Minh đọc Tuyên ngôn Độc lập ngày nào?",
        [candidate("ev_41", true_date), candidate("ev_42", corrupt_date, source_type="synthetic_conflict")],
        {
            "status": "conflicting",
            "selected_evidence": [
                {"evidence_id": "ev_41", "relevance": 1.0, "claims": [true_date], "compressed_text": true_date},
                {"evidence_id": "ev_42", "relevance": 1.0, "claims": [corrupt_date], "compressed_text": corrupt_date},
            ],
            "conflicts": ["ev_41 nêu ngày 2/9/1945 trong khi ev_42 nêu ngày 3/9/1945."],
            "missing_information": ["Cần nguồn đáng tin cậy để xác minh ngày đang mâu thuẫn."],
            "summary": "ev_41 và ev_42 mâu thuẫn trực tiếp về ngày công bố Tuyên ngôn Độc lập.",
        },
    )
    birth = "Ngô Quyền sinh ngày 17 tháng 4 năm 898."
    partial = dataset_row(
        "case-partial",
        "partial",
        "Ngô Quyền sinh và mất vào thời gian nào?",
        [candidate("ev_51", birth, source_type="synthetic_partial")],
        {
            "status": "insufficient",
            "selected_evidence": [{"evidence_id": "ev_51", "relevance": 0.75, "claims": [birth], "compressed_text": birth}],
            "conflicts": [],
            "missing_information": ["Thiếu evidence về thời điểm Ngô Quyền qua đời."],
            "summary": "ev_51 chỉ cho biết ngày sinh, chưa có ngày mất.",
        },
    )
    return [a, b, c, d, e, partial]
