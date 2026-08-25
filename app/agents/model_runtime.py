from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


class SharedAgentModelRuntime:
    """One quantized Qwen3 base with two independently trained PEFT adapters."""

    def __init__(
        self,
        *,
        model_id: str,
        research_adapter: str | Path,
        evidence_adapter: str | Path,
        dtype: str = "bfloat16",
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = getattr(torch, dtype)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(
            base,
            str(research_adapter),
            adapter_name="research",
        )
        self.model.load_adapter(str(evidence_adapter), adapter_name="evidence")
        self.model.eval()
        self._lock = threading.RLock()

    def generate_json(
        self,
        *,
        adapter: str,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        repair: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            self.model.set_adapter(adapter)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        try:
            return self._parse_json(output)
        except ValueError:
            if not repair:
                raise
            repair_messages = [
                *messages,
                {"role": "assistant", "content": output},
                {
                    "role": "user",
                    "content": "Output invalid. Return exactly one valid JSON object matching the requested schema.",
                },
            ]
            return self.generate_json(
                adapter=adapter,
                messages=repair_messages,
                max_new_tokens=max_new_tokens,
                repair=False,
            )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        match = JSON_OBJECT_RE.search(text.strip())
        if not match:
            raise ValueError("Model output does not contain a JSON object.")
        value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("Model output must be a JSON object.")
        return value
