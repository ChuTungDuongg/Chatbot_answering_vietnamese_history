from __future__ import annotations

import json

from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from app.agents.schemas import EvidenceAgentRequest, EvidenceModelOutput
from tests.evidence_v2_fixtures import sanity_rows
from training.history_answerer.evaluate import parse_source_ids
from training.evidence_agent.prepare_dataset import build_dataset_v2


def test_fixture_targets_use_the_canonical_runtime_contract():
    for row in sanity_rows():
        request = EvidenceAgentRequest.model_validate(row["input"])
        target = EvidenceModelOutput.model_validate(row["output"])
        assert row["messages"][0]["content"] == EVIDENCE_AGENT_SYSTEM
        assert json.loads(row["messages"][1]["content"]) == request.model_dump()
        assert json.loads(row["messages"][2]["content"]) == target.model_dump()
        assert "selected_ids" not in row["output"]
        assert "rejected_ids" not in row["output"]
        assert all(not isinstance(item, str) for item in row["output"]["selected_evidence"])


def test_source_parser_keeps_every_id_on_a_multi_source_line():
    text = "Nguồn được dùng: [ev_11] [ev_12]\nTrả lời: Nội dung."
    assert parse_source_ids(text) == ["ev_11", "ev_12"]


def test_builder_drops_legacy_rows_that_cite_absent_context_ids():
    source = {
        "id": "invalid-source",
        "messages": [
            {
                "role": "user",
                "content": "Câu hỏi: Dữ kiện nào?\n\nTài liệu tham khảo:\n[ev_01] Nguồn\nMột dữ kiện lịch sử đủ dài.",
            },
            {
                "role": "assistant",
                "content": "Nguồn được dùng: [ev_missing]\nTrả lời: Một câu trả lời không có source trong input.",
            },
        ],
    }
    assert build_dataset_v2([source]) == []
