from __future__ import annotations

import json
import logging
import copy
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
    tool_parse_failures: int = 0
    malformed_tool_calls: tuple[str, ...] = ()


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


def parse_central_generation_detailed(text: str) -> tuple[str, tuple[CentralToolCall, ...], int, tuple[str, ...]]:
    calls: list[CentralToolCall] = []
    failures = 0
    malformed: list[str] = []
    for index, match in enumerate(TOOL_CALL_RE.finditer(text)):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            failures += 1
            malformed.append(match.group(1)[:300])
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
            else:
                failures += 1
                malformed.append(json.dumps(item, ensure_ascii=False)[:300])
    content = TOOL_CALL_RE.sub("", text)
    content = THINK_RE.sub("", content)
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"):
        content = content.replace(token, "")
    content = content.strip()
    return content, tuple(calls), failures, tuple(malformed)


def parse_central_generation(text: str) -> tuple[str, tuple[CentralToolCall, ...]]:
    content, calls, _, _ = parse_central_generation_detailed(text)
    return content, calls


class CentralModelRuntime:
    """Dedicated Qwen3-8B + PEFT runtime; it never loads a role adapter."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str | Path,
        dtype: str = "bfloat16",
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ):
        if model_id != CENTRAL_BASE_MODEL_ID:
            raise ValueError(f"Central runtime requires {CENTRAL_BASE_MODEL_ID!r}, received {model_id!r}")
        validate_central_adapter(adapter_path)

        from app.agents.hf_cache import hf_cache_status, resolve_hf_hub_cache_dir

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Central runtime was configured for CUDA but CUDA is unavailable.")
        compute_dtype = getattr(torch, dtype)
        self.model_id = model_id
        self.adapter_path = str(adapter_path)
        self.cache_dir = resolve_hf_hub_cache_dir(cache_dir)
        resolve_started = time.perf_counter()
        before_status = hf_cache_status(model_id, cache_dir=self.cache_dir)
        resolve_ms = (time.perf_counter() - resolve_started) * 1000
        if local_files_only and not before_status["cache_hit"]:
            raise RuntimeError(
                f"Central model cache miss for {model_id!r} under {self.cache_dir}. "
                "Seed the persistent HF cache before enabling CENTRAL_AGENT_LOCAL_FILES_ONLY."
            )
        load_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        placement = {"": 0} if device == "cuda" else None
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=compute_dtype,
            device_map=placement,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
        )
        model_load_ms = (time.perf_counter() - load_started) * 1000
        adapter_started = time.perf_counter()
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        adapter_load_ms = (time.perf_counter() - adapter_started) * 1000
        self.model.eval()
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        for sampling_field in ("temperature", "top_p", "top_k"):
            if hasattr(self.generation_config, sampling_field):
                setattr(self.generation_config, sampling_field, None)
        self.adapter_loaded = True
        self._lock = threading.RLock()
        self.placement = self._placement()
        after_status = hf_cache_status(model_id, cache_dir=self.cache_dir)
        self.cache_info = {
            "central_cache_root": str(self.cache_dir),
            "central_model_snapshot_resolved": after_status.get("snapshot_path"),
            "central_cache_hit": bool(before_status["cache_hit"]),
            "central_cache_miss": not bool(before_status["cache_hit"]),
            "central_model_resolve_ms": resolve_ms,
            "central_model_load_ms": model_load_ms,
            "central_adapter_load_ms": adapter_load_ms,
            "central_local_files_only": local_files_only,
        }
        if device == "cuda" and (self.placement["cpu_offload"] or self.placement["disk_offload"]):
            raise RuntimeError("Central model unexpectedly offloaded layers to CPU/disk.")
        log_event("CENTRAL_MODEL_CACHE", model_id=model_id, **self.cache_info)
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
                generation_config=self.generation_config,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generation_ms = (time.perf_counter() - started) * 1000
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        output_tokens = int(new_tokens.shape[0])
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, tool_calls, parse_failures, malformed = parse_central_generation_detailed(decoded)
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
            tool_parse_failures=parse_failures,
        )
        return CentralGeneration(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
            tool_parse_failures=parse_failures,
            malformed_tool_calls=malformed,
        )

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
