from __future__ import annotations

from app.api.routes import _build_debug
from app.chat_modes import ChatMode


def test_central_debug_trace_uses_central_question_analysis():
    trace = _build_debug({
        "question": "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
        "answer": "Một câu trả lời có nguồn. [h1]",
        "inference_mode": ChatMode.CENTRAL,
        "analysis": {
            "question_type": "comparison",
            "analytical": True,
            "comparison_targets": ["Cách mạng Tháng Tám", "chiến thắng Điện Biên Phủ"],
        },
        "source_chunks": [{"chunk_id": "h1", "title": "Nguồn", "source_kind": "history"}],
        "central_debug": {},
        "answer_provenance": {
            "mode": "central",
            "source": "central_qwen3_8b",
            "research_generation_calls": 0,
            "evidence_generation_calls": 0,
            "history_generation_calls": 0,
        },
    })

    assert trace["request"]["question_type"] == "comparison"
    assert trace["request"]["comparison_targets"] == [
        "Cách mạng Tháng Tám",
        "chiến thắng Điện Biên Phủ",
    ]
    assert trace["central"] == {}
