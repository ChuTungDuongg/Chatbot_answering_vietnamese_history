from __future__ import annotations

import json

from app.agents.prompts import EVIDENCE_AGENT_SYSTEM
from training.evidence_agent.sft import build_evidence_assistant_only_example, prepare_evidence_split


class CharacterTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<{item['role']}>\n{item['content']}\n" for item in messages)
        return rendered + ("<assistant>\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) for char in text]
        return {"input_ids": ids[:max_length] if max_length else ids}


def _long_evidence_row():
    claim = "Chiến thắng diễn ra năm 938."
    evidence = [
        {
            "evidence_id": "ev-gold",
            "source_type": "local",
            "title": "Bạch Đằng",
            "url": "https://example.test/gold",
            "chunk_id": "chunk-gold",
            "text": ("Ngữ cảnh lịch sử " * 350) + claim + (" diễn giải" * 350),
            "retrieval_score": 0.99,
        },
        {
            "evidence_id": "ev-noise",
            "source_type": "local",
            "title": "Nhiễu",
            "url": None,
            "chunk_id": "chunk-noise",
            "text": "Thông tin không liên quan. " * 500,
            "retrieval_score": 0.1,
        },
    ]
    request = {"question": "Chiến thắng Bạch Đằng diễn ra năm nào?", "max_selected": 8, "evidence": evidence}
    output = {
        "status": "sufficient",
        "selected_evidence": [
            {"evidence_id": "ev-gold", "relevance": 0.99, "claims": [claim], "compressed_text": claim}
        ],
        "conflicts": [],
        "missing_information": [],
        "summary": "Nguồn ev-gold trả lời trực tiếp mốc thời gian được hỏi.",
    }
    return {
        "id": "long-evidence",
        "group_id": "group-long",
        "behavior": "relevant_distractor",
        "input": request,
        "output": output,
        "messages": [
            {"role": "system", "content": EVIDENCE_AGENT_SYSTEM},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False, sort_keys=True)},
        ],
    }


def test_long_evidence_pool_is_capped_as_valid_json_with_gold_claim_and_metadata():
    row = _long_evidence_row()
    feature, stats, messages = build_evidence_assistant_only_example(
        CharacterTokenizer(), row, max_length=1900
    )
    capped = json.loads(messages[1]["content"])
    gold = next(item for item in capped["evidence"] if item["evidence_id"] == "ev-gold")
    assert stats.overlength and stats.structured_truncation
    assert stats.sequence_tokens <= 1900
    assert stats.supervised_tokens == stats.assistant_tokens > 0
    assert capped["question"] == row["input"]["question"]
    assert gold["title"] == "Bạch Đằng"
    assert gold["url"] == "https://example.test/gold"
    assert gold["chunk_id"] == "chunk-gold"
    assert row["output"]["selected_evidence"][0]["claims"][0] in gold["text"]
    assert any(label != -100 for label in feature["labels"])


def test_split_preflight_reports_required_token_diagnostics():
    tokenizer = CharacterTokenizer()
    features, report = prepare_evidence_split(
        tokenizer, [_long_evidence_row()], max_length=1900, split_name="eval"
    )
    assert len(features) == report["rows"] == 1
    assert report["zero_supervised_rows"] == 0
    assert report["overlength_rows"] == 1
    assert report["truncated_rows"] == 1
    assert report["supervised_tokens"]["min"] > 0
    assert report["prompt_tokens"]["max"] > 0
    assert report["assistant_tokens"]["max"] > 0
