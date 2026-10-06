"""Only application-level backend selection; runtimes consume identical events."""
from __future__ import annotations

from typing import Any

from app.models.base import ModelRuntime


def build_model_runtime(*, settings: Any, model_id: str, **options: Any) -> ModelRuntime:
    if settings.inference_backend == "transformers":
        from app.models.qwen import QwenRuntime

        return QwenRuntime(model_id=model_id, **options)
    if settings.inference_backend == "vllm":
        from app.models.vllm import VLLMRuntime

        return VLLMRuntime(model_id=model_id, **options,
                           enable_prefix_caching=settings.vllm_enable_prefix_caching,
                           gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
                           max_num_seqs=settings.vllm_max_num_seqs,
                           max_model_len=settings.vllm_max_model_len,
                           enforce_eager=settings.vllm_enforce_eager)
    raise ValueError(f"Unsupported inference backend: {settings.inference_backend}")
