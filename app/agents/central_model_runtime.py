from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.agents.model_registry import CENTRAL_BASE_MODEL_ID, validate_central_adapter
from app.telemetry import GenerationMetric, current_request_telemetry, log_event


logger = logging.getLogger(__name__)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)


@dataclass(frozen=True)
class CentralToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CentralGeneration:
    content: str = ""
    tool_calls: tuple[CentralToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    generation_ms: float = 0.0


class CentralLLMBackend(Protocol):
    model_id: str
    adapter_loaded: bool

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> CentralGeneration: ...


def parse_central_generation(text: str) -> tuple[str, tuple[CentralToolCall, ...]]:
    calls: list[CentralToolCall] = []
    for index, match in enumerate(TOOL_CALL_RE.finditer(text)):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict):
                continue
            function = item.get("function") if isinstance(item.get("function"), dict) else item
            name = str(function.get("name") or "").strip()
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if name and isinstance(arguments, dict):
                calls.append(CentralToolCall(
                    id=str(item.get("id") or f"central_call_{index + 1}"),
                    name=name,
                    arguments=arguments,
                ))
    content = TOOL_CALL_RE.sub("", text)
    content = THINK_RE.sub("", content)
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"):
        content = content.replace(token, "")
    content = content.strip()
    return content, tuple(calls)


class CentralModelRuntime:
    """Dedicated Qwen3-8B + PEFT runtime; it never loads a role adapter."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str | Path,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        if model_id != CENTRAL_BASE_MODEL_ID:
            raise ValueError(f"Central runtime requires {CENTRAL_BASE_MODEL_ID!r}, received {model_id!r}")
        validate_central_adapter(adapter_path)

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Central runtime was configured for CUDA but CUDA is unavailable.")
        compute_dtype = getattr(torch, dtype)
        self.model_id = model_id
        self.adapter_path = str(adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        placement = {"": 0} if device == "cuda" else None
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=compute_dtype,
            device_map=placement,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
        self.adapter_loaded = True
        self._lock = threading.RLock()
        self.placement = self._placement()
        if device == "cuda" and (self.placement["cpu_offload"] or self.placement["disk_offload"]):
            raise RuntimeError("Central model unexpectedly offloaded layers to CPU/disk.")
        log_event("CENTRAL_MODEL_PLACEMENT", model_id=model_id, **self.placement)

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> CentralGeneration:
        import torch

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        telemetry = current_request_telemetry()
        with self._lock:
            inputs = inputs.to(self.model.get_input_embeddings().weight.device)
            started = time.perf_counter()
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generation_ms = (time.perf_counter() - started) * 1000
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        output_tokens = int(new_tokens.shape[0])
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, tool_calls = parse_central_generation(decoded)
        rate = output_tokens / (generation_ms / 1000) if generation_ms else 0.0
        if telemetry is not None:
            telemetry.add_generation(GenerationMetric(
                adapter="central",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                max_new_tokens=max_new_tokens,
                generation_ms=generation_ms,
                tokens_per_sec=rate,
            ))
        log_event(
            "CENTRAL_GENERATE_END",
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
            tool_call_count=len(tool_calls),
        )
        return CentralGeneration(content, tool_calls, input_tokens, output_tokens, generation_ms)

    def _placement(self) -> dict[str, Any]:
        device_map = getattr(self.model, "hf_device_map", None) or {}
        devices = sorted({str(value) for value in device_map.values()}) if isinstance(device_map, dict) else []
        gpu_name = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return {
            "devices": devices,
            "input_device": str(self.model.get_input_embeddings().weight.device),
            "cpu_offload": "cpu" in devices,
            "disk_offload": "disk" in devices,
            "gpu_name": gpu_name,
        }
