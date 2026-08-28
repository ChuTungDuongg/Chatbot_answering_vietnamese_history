from __future__ import annotations

import json
import re
import threading
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from app.agents.model_registry import ROLE_MODELS, RoleName, SHARED_BASE_MODEL_ID, validate_role_adapter


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


class RoleLLMBackend(Protocol):
    tokenizer: Any

    def generate_text(
        self,
        *,
        adapter: RoleName,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
    ) -> str: ...

    def generate_prompt(
        self,
        *,
        adapter: RoleName,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str: ...

    def generate_json(
        self,
        *,
        adapter: RoleName,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        repair: bool = True,
    ) -> dict[str, Any]: ...


class SharedAgentModelRuntime:
    """One quantized Qwen3 base with independently switchable role adapters."""

    def __init__(
        self,
        *,
        model_id: str,
        research_adapter: str | Path,
        evidence_adapter: str | Path,
        history_adapter: str | Path | None = None,
        dtype: str = "bfloat16",
        validate_adapter_metadata: bool = True,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if model_id != SHARED_BASE_MODEL_ID:
            raise ValueError(
                f"active shared runtime requires {SHARED_BASE_MODEL_ID!r}, received {model_id!r}"
            )
        adapter_paths: dict[RoleName, str | Path] = {
            "research": research_adapter,
            "evidence": evidence_adapter,
        }
        if history_adapter is not None:
            adapter_paths["history"] = history_adapter
        if validate_adapter_metadata:
            for role, path in adapter_paths.items():
                validate_role_adapter(role, path)

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
        if history_adapter is not None:
            self.model.load_adapter(str(history_adapter), adapter_name="history")
        self.model.eval()
        self.adapters = frozenset(adapter_paths)
        self._lock = threading.RLock()

    def _max_new_tokens(self, adapter: RoleName, requested: int | None) -> int:
        if adapter not in self.adapters:
            raise ValueError(f"adapter {adapter!r} is not loaded; loaded roles: {sorted(self.adapters)}")
        return int(requested or ROLE_MODELS[adapter].generation["max_new_tokens"])

    def generate_text(
        self,
        *,
        adapter: RoleName,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate_prompt(
            adapter=adapter, prompt=text, max_new_tokens=max_new_tokens
        )

    def generate_prompt(
        self,
        *,
        adapter: RoleName,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        token_budget = self._max_new_tokens(adapter, max_new_tokens)
        with self._lock:
            self.model.set_adapter(adapter)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_json(
        self,
        *,
        adapter: RoleName,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        repair: bool = True,
    ) -> dict[str, Any]:
        output = self.generate_text(
            adapter=adapter, messages=messages, max_new_tokens=max_new_tokens
        )
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


class VLLMOpenAIBackend:
    """OpenAI-compatible role router for a separately managed vLLM server.

    This class performs no server lifecycle management and never loads model
    weights.  The endpoint must expose the registry's three unique LoRA names.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        tokenizer_id: str = SHARED_BASE_MODEL_ID,
        timeout_seconds: float = 120.0,
    ):
        from transformers import AutoTokenizer

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("OpenAI-compatible backend returned a non-object response")
        return value

    def generate_text(
        self,
        *,
        adapter: RoleName,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
    ) -> str:
        spec = ROLE_MODELS[adapter]
        response = self._post(
            "/chat/completions",
            {
                "model": spec.model_name,
                "messages": messages,
                "temperature": spec.generation["temperature"],
                "top_p": spec.generation["top_p"],
                "max_tokens": int(max_new_tokens or spec.generation["max_new_tokens"]),
            },
        )
        return str(response["choices"][0]["message"]["content"])

    def generate_prompt(
        self,
        *,
        adapter: RoleName,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> str:
        spec = ROLE_MODELS[adapter]
        response = self._post(
            "/completions",
            {
                "model": spec.model_name,
                "prompt": prompt,
                "temperature": spec.generation["temperature"],
                "top_p": spec.generation["top_p"],
                "max_tokens": int(max_new_tokens or spec.generation["max_new_tokens"]),
            },
        )
        return str(response["choices"][0]["text"])

    def generate_json(
        self,
        *,
        adapter: RoleName,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        repair: bool = True,
    ) -> dict[str, Any]:
        output = self.generate_text(adapter=adapter, messages=messages, max_new_tokens=max_new_tokens)
        try:
            return SharedAgentModelRuntime._parse_json(output)
        except ValueError:
            if not repair:
                raise
            return self.generate_json(
                adapter=adapter,
                messages=[
                    *messages,
                    {"role": "assistant", "content": output},
                    {"role": "user", "content": "Return exactly one valid JSON object matching the requested schema."},
                ],
                max_new_tokens=max_new_tokens,
                repair=False,
            )
