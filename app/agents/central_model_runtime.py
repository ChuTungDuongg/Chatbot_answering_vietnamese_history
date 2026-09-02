from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.agents.hermes_function_call import HermesFunctionCallCodec
from app.agents.model_registry import CENTRAL_BASE_MODEL_ID, validate_central_adapter
from app.telemetry import GenerationMetric, current_request_telemetry, log_event


logger = logging.getLogger(__name__)
FUNCTION_CALL_CODEC = HermesFunctionCallCodec()


def choose_attention_backend(*, device: str, dtype: str, flash_available: bool, cuda_major: int, sdpa_available: bool) -> str:
    if device.startswith("cuda") and dtype in {"bfloat16", "float16"} and flash_available and cuda_major >= 8:
        return "flash_attention_2"
    return "sdpa" if sdpa_available else "eager"


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
    generation_stage: str = "unknown"
    finish_reason: str = "stop"
    generation_stop_reason: str = "stop"
    generation_hit_token_limit: bool = False
    generation_hit_time_limit: bool = False
    tool_parse_failures: int = 0
    malformed_tool_calls: tuple[str, ...] = ()


class CentralLLMBackend(Protocol):
    model_id: str
    adapter_configured: bool
    adapter_loaded: bool
    adapter_path: str | None
    adapter_source: str

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int,
        stage: str,
        deadline: float,
    ) -> CentralGeneration: ...


def parse_central_generation_detailed(text: str) -> tuple[str, tuple[CentralToolCall, ...], int, tuple[str, ...]]:
    decoded = FUNCTION_CALL_CODEC.decode(text)
    calls = tuple(CentralToolCall(call.id, call.name, call.arguments) for call in decoded.tool_calls)
    return decoded.content, calls, decoded.failures, decoded.malformed


def parse_central_generation(text: str) -> tuple[str, tuple[CentralToolCall, ...]]:
    content, calls, _, _ = parse_central_generation_detailed(text)
    return content, calls


class CentralModelRuntime:
    """Dedicated Qwen3-8B base runtime with an optional future PEFT adapter."""

    def __init__(
        self,
        *,
        model_id: str,
        adapter_path: str | Path | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ):
        if model_id != CENTRAL_BASE_MODEL_ID:
            raise ValueError(f"Central runtime requires {CENTRAL_BASE_MODEL_ID!r}, received {model_id!r}")
        normalized_adapter = str(adapter_path).strip() if adapter_path is not None else ""
        if normalized_adapter:
            validate_central_adapter(normalized_adapter)

        from app.agents.hf_cache import hf_cache_status, resolve_hf_hub_cache_dir

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Central runtime was configured for CUDA but CUDA is unavailable.")
        compute_dtype = getattr(torch, dtype)
        self.model_id = model_id
        self.adapter_path = normalized_adapter or None
        self.adapter_configured = self.adapter_path is not None
        self.adapter_source = "peft" if self.adapter_configured else "none"
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
        from transformers.utils import is_flash_attn_2_available
        self.attention_backend = choose_attention_backend(
            device=device, dtype=dtype,
            flash_available=is_flash_attn_2_available() if device == "cuda" else False,
            cuda_major=torch.cuda.get_device_capability(0)[0] if device == "cuda" else 0,
            sdpa_available=hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=compute_dtype,
            device_map=placement,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=self.attention_backend,
            trust_remote_code=True,
            cache_dir=str(self.cache_dir),
            local_files_only=local_files_only,
        )
        model_load_ms = (time.perf_counter() - load_started) * 1000
        adapter_load_ms = 0.0
        if self.adapter_path is not None:
            from peft import PeftModel

            adapter_started = time.perf_counter()
            self.model = PeftModel.from_pretrained(base, self.adapter_path)
            adapter_load_ms = (time.perf_counter() - adapter_started) * 1000
            self.adapter_loaded = True
        else:
            self.model = base
            self.adapter_loaded = False
        self.model.eval()
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.do_sample = False
        for sampling_field in ("temperature", "top_p", "top_k"):
            if hasattr(self.generation_config, sampling_field):
                setattr(self.generation_config, sampling_field, None)
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
        stage: str = "synthesis",
        deadline: float | None = None,
    ) -> CentralGeneration:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class DeadlineStoppingCriteria(StoppingCriteria):
            def __init__(self, absolute_deadline: float | None):
                self.absolute_deadline = absolute_deadline
                self.hit = False

            def __call__(self, input_ids, scores, **kwargs):
                del input_ids, scores, kwargs
                self.hit = self.absolute_deadline is not None and time.monotonic() >= self.absolute_deadline
                return self.hit

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            template_kwargs["tools"] = tools
        rendered = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        telemetry = current_request_telemetry()
        with self._lock, torch.inference_mode():
            inputs = inputs.to(self.model.get_input_embeddings().weight.device)
            started = time.perf_counter()
            stopping = DeadlineStoppingCriteria(deadline)
            remaining = max(0.1, deadline - time.monotonic()) if deadline is not None else None
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "generation_config": self.generation_config,
                "do_sample": False,
                "use_cache": True,
                "output_scores": False,
                "output_hidden_states": False,
                "output_attentions": False,
                "return_dict_in_generate": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "stopping_criteria": StoppingCriteriaList([stopping]),
            }
            if remaining is not None:
                generation_kwargs["max_time"] = remaining
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
            generation_ms = (time.perf_counter() - started) * 1000
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        output_tokens = int(new_tokens.shape[0])
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, tool_calls, parse_failures, malformed = parse_central_generation_detailed(decoded)
        hit_time_limit = stopping.hit or (deadline is not None and time.monotonic() >= deadline)
        hit_token_limit = output_tokens >= max_new_tokens and not hit_time_limit
        stop_reason = "time_limit" if hit_time_limit else "token_limit" if hit_token_limit else "stop"
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
            generation_stage=stage,
            generation_stop_reason=stop_reason,
        )
        return CentralGeneration(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_ms=generation_ms,
            generation_stage=stage,
            finish_reason=stop_reason,
            generation_stop_reason=stop_reason,
            generation_hit_token_limit=hit_token_limit,
            generation_hit_time_limit=hit_time_limit,
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
