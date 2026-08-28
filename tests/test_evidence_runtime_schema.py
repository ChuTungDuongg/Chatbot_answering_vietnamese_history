from __future__ import annotations

import pytest

from app.agents.evidence_agent import (
    EvidenceCriticAgent,
    EvidenceModelContractError,
    question_relevant_excerpt,
)
from app.agents.evidence_validation import grounded_in_source
from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.telemetry import GenerationMetric, RequestTelemetry, reset_request_telemetry, set_request_telemetry


class FakeRuntime:
    def __init__(self, output):
        self.outputs = list(output) if isinstance(output, list) else [output]
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        telemetry = __import__("app.telemetry", fromlist=["current_request_telemetry"]).current_request_telemetry()
        if telemetry is not None:
            telemetry.add_generation(
                GenerationMetric(
                    adapter=str(kwargs.get("adapter")),
                    input_tokens=1,
                    output_tokens=1,
                    max_new_tokens=int(kwargs.get("max_new_tokens") or 0),
                )
            )
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


def test_runtime_rebuckets_claim_attributed_to_one_unique_wrong_evidence_id():
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

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress("Câu hỏi?", evidence, final_k=2)

    assert critique.selected_ids == ["ev_32"]
    assert critique.selected_evidence[0].claims == ["Câu chỉ có trong ev_32."]
    assert contexts[0]["chunk_id"] == "ev_32"
    assert critique.repair_used is True
    assert critique.repair_path == "deterministic_rebucket"
    assert len(runtime.calls) == 1


def test_runtime_rebuckets_mixed_bach_dang_claims_and_preserves_own_source_claims():
    own_claim_1 = "Chiến thắng Bạch Đằng năm 938 chấm dứt hơn một nghìn năm Bắc thuộc."
    own_claim_2 = "Sau chiến thắng này, Ngô Quyền xưng vương và đóng đô ở Cổ Loa."
    foreign_claim = (
        "Đến khi tiến binh trên Bạch Đằng, quả thấy trên không có tiếng xe ngựa, "
        "trận ấy quả được đại tiệp."
    )
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "hf_wikipedia_trận_bạch_đằng_938_0002_d5f8e1eedf68",
            "relevance": 1.0,
            "claims": [own_claim_1, foreign_claim, own_claim_2],
            "compressed_text": f"{own_claim_1} {foreign_claim} {own_claim_2}",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Bằng chứng đủ để trả lời ý nghĩa chiến thắng Bạch Đằng.",
    })
    evidence = [
        {
            "chunk_id": "hf_wikipedia_trận_bạch_đằng_938_0002_d5f8e1eedf68",
            "text": f"{own_claim_1} {own_claim_2}",
            "source_kind": "history",
        },
        {
            "chunk_id": "hf_wikipedia_trận_bạch_đằng_938_0003_cf59d98b5f8d",
            "text": foreign_claim,
            "source_kind": "history",
        },
    ]

    critique, contexts = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?", evidence, final_k=2
    )

    assert critique.selected_ids == [
        "hf_wikipedia_trận_bạch_đằng_938_0002_d5f8e1eedf68",
        "hf_wikipedia_trận_bạch_đằng_938_0003_cf59d98b5f8d",
    ]
    assert critique.selected_evidence[0].claims == [own_claim_1, own_claim_2]
    assert critique.selected_evidence[0].compressed_text == f"{own_claim_1} {own_claim_2}"
    assert critique.selected_evidence[1].claims == [foreign_claim]
    assert critique.selected_evidence[1].compressed_text == foreign_claim
    assert [context["chunk_id"] for context in contexts] == critique.selected_ids
    assert critique.repair_path == "deterministic_rebucket"
    assert len(runtime.calls) == 1


def test_runtime_rebuckets_multiple_wrong_claims_to_different_sources():
    claim_a = "Nguồn A nêu dữ kiện A."
    claim_b = "Nguồn B nêu dữ kiện B."
    claim_c = "Nguồn C nêu dữ kiện C."
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_a",
            "relevance": 0.8,
            "claims": [claim_a, claim_b, claim_c],
            "compressed_text": f"{claim_a} {claim_b} {claim_c}",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Có đủ dữ kiện.",
    })

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Câu hỏi?",
        [
            {"chunk_id": "ev_a", "text": claim_a},
            {"chunk_id": "ev_b", "text": claim_b},
            {"chunk_id": "ev_c", "text": claim_c},
        ],
        final_k=3,
    )

    assert critique.selected_ids == ["ev_a", "ev_b", "ev_c"]
    assert [item.claims for item in critique.selected_evidence] == [[claim_a], [claim_b], [claim_c]]
    assert critique.repair_path == "deterministic_rebucket"


def test_runtime_rebuckets_into_existing_destination_and_deduplicates_claims():
    claim_a = "Nguồn A nêu dữ kiện A."
    claim_b = "Nguồn B nêu dữ kiện B."
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [
            {
                "evidence_id": "ev_a",
                "relevance": 0.8,
                "claims": [claim_a, claim_b],
                "compressed_text": f"{claim_a} {claim_b}",
            },
            {
                "evidence_id": "ev_b",
                "relevance": 0.9,
                "claims": [claim_b],
                "compressed_text": claim_b,
            },
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Có đủ dữ kiện.",
    })

    critique, _ = EvidenceCriticAgent(model_runtime=runtime).compress(
        "Câu hỏi?",
        [
            {"chunk_id": "ev_a", "text": claim_a},
            {"chunk_id": "ev_b", "text": claim_b},
        ],
        final_k=2,
    )

    assert critique.selected_ids == ["ev_a", "ev_b"]
    assert critique.selected_evidence[0].claims == [claim_a]
    assert critique.selected_evidence[1].claims == [claim_b]
    assert critique.selected_evidence[1].compressed_text == claim_b
    assert critique.repair_path == "deterministic_rebucket"


def test_runtime_keeps_ambiguous_cross_id_claim_as_failure():
    claim = "Câu xuất hiện trong hai nguồn."
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_a",
            "relevance": 1.0,
            "claims": [claim],
            "compressed_text": claim,
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Không an toàn.",
    })

    with pytest.raises(EvidenceModelContractError) as exc_info:
        EvidenceCriticAgent(model_runtime=runtime).compress(
            "Câu hỏi?",
            [
                {"chunk_id": "ev_a", "text": "Nguồn A không chứa claim."},
                {"chunk_id": "ev_b", "text": claim},
                {"chunk_id": "ev_c", "text": claim},
            ],
            final_k=3,
        )

    assert exc_info.value.code == "cross_id_claim"
    assert exc_info.value.repair_attempted is False
    assert len(runtime.calls) == 1


def test_runtime_keeps_no_source_cross_id_claim_as_failure_after_existing_recovery_path():
    runtime = FakeRuntime({
        "status": "sufficient",
        "selected_evidence": [{
            "evidence_id": "ev_a",
            "relevance": 1.0,
            "claims": ["Câu không nằm trong bất kỳ nguồn nào."],
            "compressed_text": "Câu không nằm trong bất kỳ nguồn nào.",
        }],
        "conflicts": [],
        "missing_information": [],
        "summary": "Không an toàn.",
    })

    with pytest.raises(EvidenceModelContractError) as exc_info:
        EvidenceCriticAgent(model_runtime=runtime).compress(
            "Câu hỏi?",
            [{"chunk_id": "ev_a", "text": "Điện Biên Phủ kết thúc năm 1954."}],
            final_k=1,
        )

    assert exc_info.value.code == "grounding_contract_failed"
    assert exc_info.value.repair_attempted is False


def test_runtime_keeps_invalid_conflict_semantics_as_failure_after_rebucket_attempt():
    claim_a = "Nguồn A nêu dữ kiện A."
    claim_b = "Nguồn B nêu dữ kiện B."
    runtime = FakeRuntime({
        "status": "conflicting",
        "selected_evidence": [{
            "evidence_id": "ev_a",
            "relevance": 1.0,
            "claims": [claim_b],
            "compressed_text": claim_b,
        }],
        "conflicts": ["Có mâu thuẫn nhưng không nêu đủ ID."],
        "missing_information": ["Cần đối chiếu."],
        "summary": "Có mâu thuẫn.",
    })

    with pytest.raises(EvidenceModelContractError) as exc_info:
        EvidenceCriticAgent(model_runtime=runtime).compress(
            "Câu hỏi?",
            [{"chunk_id": "ev_a", "text": claim_a}, {"chunk_id": "ev_b", "text": claim_b}],
            final_k=2,
        )

    assert exc_info.value.code == "conflict_requires_two_ids"
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


def test_paraphrased_claim_without_exact_source_match_remains_failure():
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

    with pytest.raises(EvidenceModelContractError) as exc_info:
        EvidenceCriticAgent(model_runtime=runtime).compress(
            "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?", evidence, final_k=1
        )

    assert exc_info.value.code == "grounding_contract_failed"
    assert exc_info.value.repair_attempted is False
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


def test_unrecoverable_output_raises_controlled_error_without_model_repair():
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
    assert exc_info.value.repair_attempted is False
    assert len(runtime.calls) == 1


def test_claim_not_extractive_does_not_use_second_model_repair():
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

    telemetry = RequestTelemetry(request_id="req-evidence")
    token = set_request_telemetry(telemetry)
    try:
        with pytest.raises(EvidenceModelContractError) as exc_info:
            EvidenceCriticAgent(model_runtime=runtime).compress(
                "Chiến thắng Bạch Đằng năm 938 có ý nghĩa gì?",
                [{"chunk_id": "ev_01", "text": source}],
                final_k=1,
                request_id="req-evidence",
            )
    finally:
        reset_request_telemetry(token)

    assert exc_info.value.code == "grounding_contract_failed"
    assert len(runtime.calls) == 1
    assert telemetry.evidence_generation_calls == 1
    assert telemetry.evidence_repair_used is False


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
