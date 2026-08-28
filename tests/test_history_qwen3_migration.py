from __future__ import annotations

import pytest

from app.agents.model_registry import SHARED_BASE_MODEL_ID
from training.history_answerer.config import Phase6Config
from training.history_answerer.loss import IGNORE_INDEX, build_rag_training_example_with_stats
from training.history_answerer.validate_dataset import validate_rows


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<{item['role']}>\n{item['content']}\n" for item in messages)
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return rendered

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": [ord(character) for character in text]}


def test_history_active_default_is_qwen3():
    assert Phase6Config().model_id == SHARED_BASE_MODEL_ID


def test_long_history_prompt_is_left_truncated_and_target_is_complete():
    tokenizer = CharacterTokenizer()
    assistant = "Nguồn được dùng: [c1]\nTrả lời: OK"
    feature, stats = build_rag_training_example_with_stats(
        tokenizer, "x" * 500, assistant, max_length=96
    )
    supervised = [label for label in feature["labels"] if label != IGNORE_INDEX]
    assert stats.prompt_truncated
    assert not stats.assistant_truncated
    assert len(supervised) == stats.assistant_tokens > 0


def test_history_target_too_long_fails_instead_of_all_ignore():
    with pytest.raises(ValueError, match="target alone exceeds"):
        build_rag_training_example_with_stats(
            CharacterTokenizer(), "short", "Nguồn được dùng: [c1]\nTrả lời: " + "x" * 200, max_length=64
        )


def test_history_grounding_validator_rejects_unknown_citation_and_embedding():
    rows = [{
        "id": "g1",
        "type": "grounded_qa",
        "messages": [
            {"role": "user", "content": "Câu hỏi:\nQ\nTài liệu tham khảo:\n[c1] T\nembedding: [1,2,3]"},
            {"role": "assistant", "content": "Nguồn được dùng: [missing]\nTrả lời: A"},
        ],
    }]
    report = validate_rows(rows)
    assert not report["valid"]
    assert report["invalid_source_ids"] == 1
    assert report["embedding_leakage"] == 1

