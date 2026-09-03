from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from app.agents.common.model_registry import ROLE_MODELS, RoleName, SHARED_BASE_MODEL_ID, validate_role_adapter
from app.telemetry import GenerationMetric, current_request_telemetry, log_event


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)
logger = logging.getLogger(__name__)


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
        research_adapter: str | Path | None = None,
        evidence_adapter: str | Path | None = None,
        history_adapter: str | Path | None = None,
        dtype: str = "bfloat16",
        validate_adapter_metadata: bool = True,
        forbid_cpu_disk_offload: bool = True,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if model_id != SHARED_BASE_MODEL_ID:
            raise ValueError(
                f"active shared runtime requires {SHARED_BASE_MODEL_ID!r}, received {model_id!r}"
            )
        adapter_paths: dict[RoleName, str | Path] = {}
        if research_adapter is not None:
            adapter_paths["research"] = research_adapter
        if evidence_adapter is not None:
            adapter_paths["evidence"] = evidence_adapter
        if history_adapter is not None:
            adapter_paths["history"] = history_adapter
        if not adapter_paths:
            raise ValueError("Shared role runtime requires at least one role adapter.")
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
        first_role, first_path = next(iter(adapter_paths.items()))
        self.model = PeftModel.from_pretrained(base, str(first_path), adapter_name=first_role)
        for role, path in list(adapter_paths.items())[1:]:
            self.model.load_adapter(str(path), adapter_name=role)
        self.model.eval()
        self.adapters = frozenset(adapter_paths)
        self._lock = threading.RLock()
        self._log_model_placement(forbid_cpu_disk_offload=forbid_cpu_disk_offload)

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
        import torch

        token_budget = self._max_new_tokens(adapter, max_new_tokens)
        telemetry = current_request_telemetry()
        request_id = telemetry.request_id if telemetry is not None else "none"
        call_index = telemetry.next_call_index() if telemetry is not None else 0
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_tokens = int(inputs["input_ids"].shape[1])
        cuda_before = self._cuda_memory_snapshot()
        lock_started = time.perf_counter()
        with self._lock:
            lock_wait_ms = (time.perf_counter() - lock_started) * 1000
            switch_started = time.perf_counter()
            self.model.set_adapter(adapter)
            adapter_switch_ms = (time.perf_counter() - switch_started) * 1000
            target_device = self._input_device()
            inputs = inputs.to(target_device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            log_event(
                "LLM_GENERATE_START",
                request_id=request_id,
                call_index=call_index,
                adapter=adapter,
                input_tokens=input_tokens,
                max_new_tokens=token_budget,
            )
            generation_started = time.perf_counter()
            generated = self.model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            output_tokens = int(new_tokens.shape[0])
            decode_started = time.perf_counter()
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            decode_ms = (time.perf_counter() - decode_started) * 1000
            cuda_after = self._cuda_memory_snapshot()
            tokens_per_sec = output_tokens / (generation_ms / 1000) if generation_ms > 0 else 0.0
            peak_allocated_gb = cuda_after.get("peak_allocated_gb")
            metric = GenerationMetric(
                adapter=adapter,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                max_new_tokens=token_budget,
                lock_wait_ms=lock_wait_ms,
                adapter_switch_ms=adapter_switch_ms,
                generation_ms=generation_ms,
                decode_ms=decode_ms,
                tokens_per_sec=tokens_per_sec,
                peak_allocated_gb=peak_allocated_gb,
            )
            if telemetry is not None:
                telemetry.add_generation(metric)
            log_event(
                "LLM_GENERATE_END",
                request_id=request_id,
                call_index=call_index,
                adapter=adapter,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                lock_wait_ms=lock_wait_ms,
                adapter_switch_ms=adapter_switch_ms,
                generation_ms=generation_ms,
                decode_ms=decode_ms,
                tokens_per_sec=tokens_per_sec,
                allocated_gb_before=cuda_before.get("allocated_gb"),
                reserved_gb_before=cuda_before.get("reserved_gb"),
                allocated_gb_after=cuda_after.get("allocated_gb"),
                reserved_gb_after=cuda_after.get("reserved_gb"),
                peak_allocated_gb=peak_allocated_gb,
            )
            return decoded

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
            telemetry = current_request_telemetry()
            request_id = telemetry.request_id if telemetry is not None else "none"
            initial_call_index = telemetry.total_llm_calls if telemetry is not None else None
            log_event(
                "JSON_PARSE_FAILED",
                request_id=request_id,
                adapter=adapter,
                call_index=initial_call_index,
            )
            log_event("JSON_REPAIR_START", request_id=request_id, adapter=adapter)
            repair_messages = [
                *messages,
                {"role": "assistant", "content": output},
                {
                    "role": "user",
                    "content": "Output invalid. Return exactly one valid JSON object matching the requested schema.",
                },
            ]
            repaired = self.generate_json(
                adapter=adapter,
                messages=repair_messages,
                max_new_tokens=max_new_tokens,
                repair=False,
            )
            log_event("JSON_REPAIR_END", request_id=request_id, adapter=adapter)
            if telemetry is not None and adapter == "research":
                telemetry.research_json_repairs += 1
            return repaired

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        match = JSON_OBJECT_RE.search(text.strip())
        if not match:
            raise ValueError("Model output does not contain a JSON object.")
        value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("Model output must be a JSON object.")
        return value

    def _input_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return getattr(self.model, "device", "cpu")

    @staticmethod
    def _cuda_memory_snapshot() -> dict[str, float | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "allocated_gb": None,
                    "reserved_gb": None,
                    "peak_allocated_gb": None,
                }
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        except Exception:
            return {
                "allocated_gb": None,
                "reserved_gb": None,
                "peak_allocated_gb": None,
            }

    def _log_model_placement(self, *, forbid_cpu_disk_offload: bool = True) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        device_counts: dict[str, int] = {}
        cpu_offload = False
        disk_offload = False
        if isinstance(device_map, dict):
            for device in device_map.values():
                key = str(device)
                device_counts[key] = device_counts.get(key, 0) + 1
                cpu_offload = cpu_offload or key == "cpu"
                disk_offload = disk_offload or key == "disk"
        qwen_input_embedding_device = None
        try:
            qwen_input_embedding_device = str(self.model.get_input_embeddings().weight.device)
        except Exception:
            pass
        log_event(
            "MODEL_PLACEMENT",
            hf_device_map_summary=device_counts,
            hf_device_map=device_map if len(device_counts) <= 16 else None,
            cpu_offload=cpu_offload,
            disk_offload=disk_offload,
            qwen_input_embedding_device=qwen_input_embedding_device,
        )
        if cpu_offload or disk_offload:
            logger.warning(
                "QWEN_OFFLOAD_DETECTED",
                extra={"cpu_offload": cpu_offload, "disk_offload": disk_offload},
            )
            if forbid_cpu_disk_offload:
                raise RuntimeError("Shared Qwen3 role runtime unexpectedly offloaded layers to CPU/disk.")


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
