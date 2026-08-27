import math

import pytest

from training.common.sft import AssistantOnlyCollator, IGNORE_INDEX, build_assistant_only_example


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<{item['role']}>\n{item['content']}\n" for item in messages)
        return rendered + ("<assistant>\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) for char in text]
        return {"input_ids": ids[:max_length] if max_length else ids}


def test_research_sft_masks_shared_system_and_user_tokens():
    tokenizer = CharacterTokenizer()
    messages = [
        {"role": "system", "content": "shared policy"},
        {"role": "user", "content": '{"question":"q"}'},
        {"role": "assistant", "content": '{"action":"finish"}'},
    ]
    example = build_assistant_only_example(tokenizer, messages, max_length=4096)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    assert example["labels"][: len(prefix)] == [IGNORE_INDEX] * len(prefix)
    assert example["labels"][len(prefix):] == example["input_ids"][len(prefix):]


def _messages(user: str = "prompt", assistant: str = "target"):
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _assistant_ids(tokenizer, messages):
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer(full, add_special_tokens=False)["input_ids"][len(prefix):]


def test_short_prompt_and_assistant_are_retained():
    tokenizer = CharacterTokenizer()
    messages = _messages()
    example = build_assistant_only_example(tokenizer, messages, max_length=256)
    assert [label for label in example["labels"] if label != IGNORE_INDEX] == _assistant_ids(tokenizer, messages)


def test_prefix_exactly_near_max_length_keeps_complete_target():
    tokenizer = CharacterTokenizer()
    messages = _messages(user="p" * 80, assistant="answer")
    assistant_ids = _assistant_ids(tokenizer, messages)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    example = build_assistant_only_example(tokenizer, messages, max_length=len(prefix) + len(assistant_ids) - 1)
    assert len(example["input_ids"]) == len(prefix) + len(assistant_ids) - 1
    assert [label for label in example["labels"] if label != IGNORE_INDEX] == assistant_ids


def test_prefix_longer_than_max_length_trims_prompt_before_assistant():
    tokenizer = CharacterTokenizer()
    messages = _messages(user="context" * 100, assistant="final answer")
    assistant_ids = _assistant_ids(tokenizer, messages)
    example = build_assistant_only_example(tokenizer, messages, max_length=len(assistant_ids) + 8)
    assert example["labels"][:8] == [IGNORE_INDEX] * 8
    assert example["labels"][8:] == assistant_ids
    assert sum(label != IGNORE_INDEX for label in example["labels"]) == len(assistant_ids)


def test_assistant_target_larger_than_window_fails_instead_of_slicing_target():
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="assistant target alone exceeds"):
        build_assistant_only_example(tokenizer, _messages(assistant="x" * 100), max_length=32)


def test_empty_tokenization_cannot_create_zero_supervised_example():
    class EmptyTokenizer(CharacterTokenizer):
        def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
            return {"input_ids": []}

    with pytest.raises(ValueError, match="empty token sequence"):
        build_assistant_only_example(EmptyTokenizer(), _messages(), max_length=128)


def test_collator_padding_masks_labels_with_ignore_index():
    tokenizer = CharacterTokenizer()
    short = build_assistant_only_example(tokenizer, _messages(), max_length=256)
    long = build_assistant_only_example(tokenizer, _messages(user="longer prompt"), max_length=256)
    batch = AssistantOnlyCollator(pad_token_id=0)([short, long])
    padding = batch["attention_mask"][0] == 0
    assert padding.any()
    assert (batch["labels"][0][padding] == IGNORE_INDEX).all()


def test_collator_rejects_an_all_masked_sample_before_loss():
    with pytest.raises(ValueError, match="zero supervised tokens"):
        AssistantOnlyCollator(pad_token_id=0)([
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [IGNORE_INDEX, IGNORE_INDEX]}
        ])


def test_eval_loss_fixture_with_batch_size_one_is_finite():
    import torch
    import torch.nn.functional as F

    tokenizer = CharacterTokenizer()
    feature = build_assistant_only_example(
        tokenizer,
        _messages(user="very long " * 100, assistant="finite target"),
        max_length=64,
    )
    batch = AssistantOnlyCollator(pad_token_id=0)([feature])
    logits = torch.zeros((1, batch["labels"].shape[1], 256), dtype=torch.float32)
    loss = F.cross_entropy(logits.view(-1, 256), batch["labels"].view(-1), ignore_index=IGNORE_INDEX)
    assert math.isfinite(loss.item())
