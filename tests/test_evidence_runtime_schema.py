from __future__ import annotations

import pytest

from app.agents.evidence_agent import (
    EvidenceCriticAgent,
    EvidenceModelContractError,
    question_relevant_excerpt,
)
from app.agents.evidence_validation import grounded_in_source
from app.agents.prompts import EVIDENCE_AGENT_SYSTEM


class FakeRuntime:
    def __init__(self, output):
        self.outputs = list(output) if isinstance(output, list) else [output]
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


def _canonical(text: str, *, evidence_id: str = "ev_01"):
    return {
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": evidence_id,
            "relevance": 1.0,
            "claims": [text],
            "compressed_text": text,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": f"{evidence_id} đủ để trả lời.",
    }


def test_strict_grounding_rejects_known_paraphrase_fixture():
    source = "Chiến thắng Bạch Đằng năm 938 mở ra thời kỳ độc lập tự chủ."
    paraphrase = "Chiến thắng này đã giúp dân tộc giành lại nền tự chủ lâu dài."

    assert grounded_in_source(paraphrase, source) is False


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
    assert len(runtime.calls) == 1


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
    assert len(runtime.calls) == 1


def test_runtime_rejects_invented_id_without_repair():
    runtime = FakeRuntime(_canonical("Câu đúng nhưng ID không tồn tại.", evidence_id="ev_missing"))
    evidence = [{"chunk_id": "ev_01", "text": "Câu đúng nhưng ID không tồn tại.", "source_kind": "local"}]

    with pytest.raises(EvidenceModelContractError) as exc_info:
        EvidenceCriticAgent(model_runtime=runtime).compress("Câu hỏi?", evidence, final_k=1)

    assert exc_info.value.code == "invented_evidence_id"
    assert exc_info.value.repair_attempted is False
    assert len(runtime.calls) == 1


def test_paraphrased_claim_recovers_to_same_source_extractive_span():
    source = "Chiến thắng Bạch Đằng năm 938 mở ra thời kỳ độc lập tự chủ."
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_01",
            "relevance": 0.98,
            "claims": ["Chiến thắng này đã giúp dân tộc giành lại nền tự chủ lâu dài."],
            "compressed_text": "Chiến thắng này đã giúp dân tộc giành lại nền tự chủ lâu dài.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Có thể trả lời ý nghĩa.",
    })
    evidence = [{"chunk_id": "ev_01", "text": source, "source_kind": "local"}]

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?", evidence, final_k=1
    )

    assert critique.selected_evidence[0].claims == [source]
    assert contexts[0]["text"] == source
    assert critique.generation_calls == 1
    assert critique.repair_used is True
    assert critique.repair_path == "deterministic"
    assert len(runtime.calls) == 1


def test_invalid_compressed_text_recovers_from_own_exact_claims():
    source = "Nhà Trần ba lần đánh bại quân Nguyên Mông."
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_01",
            "relevance": 1.0,
            "claims": [source],
            "compressed_text": "Nhà Trần chiến thắng nhờ sức mạnh quân sự vượt trội.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Có bằng chứng.",
    })
    evidence = [{"chunk_id": "ev_01", "text": source, "source_kind": "local"}]

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Nhà Trần kháng chiến chống Nguyên Mông thế nào?", evidence, final_k=1
    )

    assert critique.selected_evidence[0].compressed_text == source
    assert contexts[0]["text"] == source
    assert critique.generation_calls == 1
    assert critique.repair_path == "deterministic"


def test_valid_extractive_output_is_untouched():
    source = "Ngày 2/9/1945, Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Ba Đình."
    runtime = FakeRuntime(_canonical(source))

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Hồ Chí Minh đọc Tuyên ngôn Độc lập ở đâu?", [{"chunk_id": "ev_01", "text": source}], final_k=1
    )

    assert critique.selected_evidence[0].compressed_text == source
    assert contexts[0]["text"] == source
    assert critique.repair_used is False
    assert len(runtime.calls) == 1


def test_unrecoverable_output_raises_controlled_error_after_one_repair():
    source = "Tài liệu chỉ nói về chiến dịch Điện Biên Phủ năm 1954."
    first = {
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_01",
            "relevance": 1.0,
            "claims": ["Chiến thắng này mở ra nền tự chủ lâu dài."],
            "compressed_text": "Chiến thắng này mở ra nền tự chủ lâu dài.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Sai.",
    }
    runtime = FakeRuntime([first, first])

    with pytest.raises(EvidenceModelContractError) as exc_info:
        EvidenceCriticAgent(model_runtime=runtime).compress(
            "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
            [{"chunk_id": "ev_01", "text": source}],
            final_k=1,
        )

    assert exc_info.value.code == "grounding_contract_failed"
    assert exc_info.value.repair_attempted is True
    assert len(runtime.calls) == 2


def test_second_repair_valid_output_is_accepted():
    source = "Chiến thắng Bạch Đằng năm 938 chấm dứt thời kỳ Bắc thuộc."
    first = {
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_01",
            "relevance": 1.0,
            "claims": ["Nội dung này khẳng định một kết luận khác hẳn."],
            "compressed_text": "Nội dung này khẳng định một kết luận khác hẳn.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Cần repair.",
    }
    runtime = FakeRuntime([first, _canonical(source)])

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?",
        [{"chunk_id": "ev_01", "text": source}],
        final_k=1,
    )

    assert critique.selected_evidence[0].claims == [source]
    assert contexts[0]["text"] == source
    assert critique.generation_calls == 2
    assert critique.repair_used is True
    assert critique.repair_path == "model"
    assert len(runtime.calls) == 2


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
