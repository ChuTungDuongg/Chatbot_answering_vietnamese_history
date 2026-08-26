from __future__ import annotations

from training.history_answerer.loss import IGNORE_INDEX, build_rag_training_example


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = ""
        for message in messages:
            rendered += f"<{message['role']}>\n{message['content']}\n"
        if add_generation_prompt:
            rendered += "<assistant>\n"
        return rendered

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(character) for character in text]
        return {"input_ids": ids[:max_length] if max_length else ids}


def test_phase6_loss_masks_user_and_weights_source_line():
    tokenizer = CharacterTokenizer()
    source = "Nguồn được dùng: [c1]\n"
    example = build_rag_training_example(
        tokenizer,
        "Câu hỏi",
        f"{source}Trả lời: Ngô Quyền.",
        max_length=4096,
    )
    prefix_len = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": "Câu hỏi"}],
        tokenize=False,
        add_generation_prompt=True,
    ))
    assert all(label == IGNORE_INDEX for label in example["labels"][:prefix_len])
    assert example["loss_weights"][prefix_len : prefix_len + len(source)] == [1.6] * len(source)
    assert all(weight == 1.0 for weight in example["loss_weights"][prefix_len + len(source) :])
