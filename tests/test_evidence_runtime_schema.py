from __future__ import annotations

import pytest

from app.agents.evidence_agent import (
    EvidenceCriticAgent,
    EvidenceModelContractError,
    question_relevant_excerpt,
)
from app.agents.prompts import EVIDENCE_AGENT_SYSTEM


class FakeRuntime:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_runtime_accepts_canonical_output_and_derives_transport_fields():
    text = "Ngày 2/9/1945, Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Ba Đình."
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_01",
            "relevance": 1.0,
            "claims": [text],
            "compressed_text": text,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "ev_01 đủ để xác định ngày và địa điểm.",
    })
    evidence = [
        {"chunk_id": "ev_01", "text": text, "source_kind": "local", "score": 0.9},
        {"chunk_id": "ev_02", "text": "Chiến dịch Điện Biên Phủ kết thúc năm 1954.", "source_kind": "local"},
    ]

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Hồ Chí Minh đọc Tuyên ngôn Độc lập khi nào và ở đâu?", evidence, final_k=2
    )

    assert critique.selected_ids == ["ev_01"]
    assert critique.rejected_ids == ["ev_02"]
    assert critique.compressed_context == f"[ev_01] {text}"
    assert critique.sufficient is True
    assert contexts[0]["text"] == text
    assert runtime.calls[0]["messages"][0]["content"] == EVIDENCE_AGENT_SYSTEM


def test_runtime_rejects_legacy_list_of_ids_with_clear_diagnostic():
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": ["ev_01"],
        "conflicts": [],
        "missing_information": [],
        "summary": "Legacy output.",
    })
    evidence = [{"chunk_id": "ev_01", "text": "Evidence hợp lệ.", "source_kind": "local"}]

    with pytest.raises(EvidenceModelContractError, match=r"legacy selected_evidence format list\[str\]"):
        EvidenceCriticAgent(model_runtime=runtime).compress("Câu hỏi?", evidence, final_k=1)


def test_runtime_rejects_claim_attributed_to_the_wrong_evidence_id():
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_31",
            "relevance": 1.0,
            "claims": ["Câu chỉ có trong ev_32."],
            "compressed_text": "Câu chỉ có trong ev_32.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Sai attribution.",
    })
    evidence = [
        {"chunk_id": "ev_31", "text": "Câu A thuộc ev_31.", "source_kind": "local"},
        {"chunk_id": "ev_32", "text": "Câu chỉ có trong ev_32.", "source_kind": "local"},
    ]
    with pytest.raises(EvidenceModelContractError, match="same evidence source"):
        EvidenceCriticAgent(model_runtime=runtime).compress("Câu hỏi?", evidence, final_k=2)


def test_runtime_rejects_compressed_text_copied_from_another_source():
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_31",
            "relevance": 1.0,
            "claims": ["Câu A thuộc ev_31."],
            "compressed_text": "Câu B chỉ thuộc ev_32.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Sai compressed attribution.",
    })
    evidence = [
        {"chunk_id": "ev_31", "text": "Câu A thuộc ev_31.", "source_kind": "local"},
        {"chunk_id": "ev_32", "text": "Câu B chỉ thuộc ev_32.", "source_kind": "local"},
    ]
    with pytest.raises(EvidenceModelContractError, match="not derivable"):
        EvidenceCriticAgent(model_runtime=runtime).compress("Câu hỏi?", evidence, final_k=2)


def test_runtime_rejects_conflict_without_two_supplied_ids():
    text = "ev_41 nêu ngày 2/9/1945."
    runtime = FakeRuntime({
        "status": "conflicting",
        "selected_evidence": [{
            "evidence_id": "ev_41",
            "relevance": 1.0,
            "claims": [text],
            "compressed_text": text,
        }],
        "conflicts": ["Hai nguồn mâu thuẫn."],
        "missing_information": ["Cần xác minh ngày."],
        "summary": "Có mâu thuẫn.",
    })
    evidence = [{"chunk_id": "ev_41", "text": text, "source_kind": "local"}]
    with pytest.raises(EvidenceModelContractError, match="at least two"):
        EvidenceCriticAgent(model_runtime=runtime).compress("Ngày nào?", evidence, final_k=1)


def test_question_relevant_excerpt_keeps_a_relevant_late_section():
    text = ("Phần đầu chỉ mô tả diễn biến và chiến thuật. " * 100) + (
        "Ý nghĩa của sự kiện được trình bày ở phần cuối tài liệu."
    )

    excerpt = question_relevant_excerpt(
        text,
        "Sự kiện có ý nghĩa như thế nào?",
        max_chars=1800,
    )

    assert len(excerpt) <= 1800
    assert "Ý nghĩa của sự kiện" in excerpt
    assert excerpt in text
