from __future__ import annotations

import pytest

from training.trajectory_dataset.preprocess import (
    IGNORE_INDEX,
    analyze_truncation,
    build_canonical_sft_example,
)


class EmptyRejectingTokenizer:
    """Small prefix-stable tokenizer that rejects empty chat conversations."""

    def __init__(self) -> None:
        self.rendered_conversations: list[list[dict]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
    ):
        assert tokenize is False
        if not messages:
            raise IndexError("empty conversations are unsupported")
        self.rendered_conversations.append(list(messages))
        rendered = "".join(
            f"<{message['role']}>{message.get('content') or ''}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def row(messages: list[dict]) -> dict:
    return {"messages": messages, "tools": []}


def test_analyze_truncation_supports_user_first_without_rendering_empty_prefix():
    tokenizer = EmptyRejectingTokenizer()
    report = analyze_truncation(
        tokenizer,
        row([
            {"role": "user", "content": "Câu hỏi"},
            {"role": "assistant", "content": "Câu trả lời"},
        ]),
        max_length=10_000,
    )

    assert not report["truncated"]
    assert report["initial_user_span"][0] == 0
    assert len(report["assistant_spans"]) == 1
    assert tokenizer.rendered_conversations
    assert all(messages for messages in tokenizer.rendered_conversations)


def test_fake_tokenizer_proves_empty_chat_template_calls_would_fail():
    tokenizer = EmptyRejectingTokenizer()
    with pytest.raises(IndexError, match="empty conversations"):
        tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=False)


def test_analyze_truncation_still_supports_system_first_conversation():
    tokenizer = EmptyRejectingTokenizer()
    report = analyze_truncation(
        tokenizer,
        row([
            {"role": "system", "content": "Bạn là trợ lý lịch sử."},
            {"role": "user", "content": "Câu hỏi"},
            {"role": "assistant", "content": "Câu trả lời"},
        ]),
        max_length=10_000,
    )

    assert not report["truncated"]
    assert report["initial_user_span"][0] > 0
    assert len(report["assistant_spans"]) == 1


def test_build_canonical_sft_example_remains_compatible_for_normal_sample():
    tokenizer = EmptyRejectingTokenizer()
    feature = build_canonical_sft_example(
        tokenizer,
        row([
            {"role": "user", "content": "Câu hỏi"},
            {"role": "assistant", "content": "Câu trả lời"},
        ]),
        max_length=10_000,
    )

    assert len(feature["input_ids"]) == len(feature["attention_mask"]) == len(feature["labels"])
    assert any(label != IGNORE_INDEX for label in feature["labels"])
    assert all(messages for messages in tokenizer.rendered_conversations)
