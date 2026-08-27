from training.common.sft import IGNORE_INDEX, build_assistant_only_example


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
