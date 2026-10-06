"""Optional pinned vLLM V1 async engine, preserving the Qwen/Hermes prompt."""
from __future__ import annotations

import asyncio
import importlib.metadata
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator
from uuid import uuid4

from app.models.base import ModelDelta, ModelDone, chat_prompt

VLLM_VERSION = "0.23.0"


class VLLMRuntime:
    inference_backend = "vllm"

    def __init__(self, *, model_id: str, revision: str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16", cache_dir: str | None = None,
                 local_files_only: bool = False, do_sample: bool = False,
                 enable_thinking: bool = False, adapter_path: str | Path | None = None,
                 temperature: float = 0.7, top_p: float = 1.0,
                 enable_prefix_caching: bool = False, gpu_memory_utilization: float = 0.75,
                 max_num_seqs: int = 8, max_model_len: int | None = None,
                 enforce_eager: bool = False):
        from app.config import CENTRAL_MODEL_ID, HYBRID_MODEL_ID

        if model_id not in {HYBRID_MODEL_ID, CENTRAL_MODEL_ID}:
            raise ValueError(f"Unsupported baseline model: {model_id}")
        if adapter_path:
            raise NotImplementedError("vLLM SFT/LoRA parity is not verified; use vanilla or INFERENCE_BACKEND=transformers")
        if temperature < 0 or not 0 < top_p <= 1 or do_sample and temperature == 0:
            raise ValueError("Invalid generation sampling settings")
        self.model_id, self.revision = model_id, revision
        self.device, self.dtype, self.cache_dir = device, dtype, cache_dir
        self.local_files_only = local_files_only
        self.do_sample, self.enable_thinking = do_sample, enable_thinking
        self.temperature, self.top_p = temperature, top_p
        self.adapter_path = self.adapter_fingerprint = None
        self.resolved_revision: str | None = None
        self.engine = self.tokenizer = None
        self._load_lock = asyncio.Lock()
        self._effective_config: dict[str, Any] | None = None
        self._engine_options = {"enable_prefix_caching": enable_prefix_caching,
                                "gpu_memory_utilization": gpu_memory_utilization,
                                "max_num_seqs": max_num_seqs, "enforce_eager": enforce_eager}
        if max_model_len is not None:
            self._engine_options["max_model_len"] = max_model_len

    @property
    def is_loaded(self) -> bool:
        return self.engine is not None

    @property
    def model_variant(self) -> str:
        return "vanilla"

    @property
    def adapter_attached(self) -> bool:
        return False

    @property
    def generation_settings(self) -> dict[str, Any]:
        return {"do_sample": self.do_sample, "temperature": self.temperature,
                "top_p": self.top_p, "enable_thinking": self.enable_thinking,
                "dtype": self.dtype, "quantization": None, "adapter": None,
                "model_variant": self.model_variant, "adapter_attached": False,
                "adapter_fingerprint": None}

    @property
    def engine_metadata(self) -> dict[str, Any]:
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {"inference_backend": self.inference_backend,
                "inference_engine_version": version,
                "inference_engine_config": self._effective_config}

    def _initialize(self):
        if self.device != "cuda":
            raise RuntimeError("The supported vLLM runtime requires a Linux CUDA deployment")
        if self.engine_metadata["inference_engine_version"] != VLLM_VERSION:
            raise RuntimeError(f"Install the isolated requirements-vllm.txt bundle (vllm=={VLLM_VERSION})")
        from transformers import AutoConfig, AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        kwargs = {"revision": self.revision, "cache_dir": self.cache_dir,
                  "local_files_only": self.local_files_only, "trust_remote_code": True}
        config = AutoConfig.from_pretrained(self.model_id, **kwargs)
        resolved = getattr(config, "_commit_hash", None) or self.revision
        kwargs["revision"] = resolved
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        engine_model = self.model_id
        if self.local_files_only:
            from huggingface_hub import snapshot_download

            engine_model = snapshot_download(self.model_id, revision=resolved,
                                             cache_dir=self.cache_dir, local_files_only=True)
        args = AsyncEngineArgs(model=engine_model, revision=resolved,
                               tokenizer_revision=resolved, dtype=self.dtype,
                               download_dir=self.cache_dir, trust_remote_code=True,
                               quantization=None, enable_lora=False,
                               generation_config="vllm", enable_log_requests=False,
                               disable_log_stats=False, **self._engine_options)
        # V1 supports constructing outside an event loop and starts the output
        # handler on the first add_request. GPU startup stays off the API loop.
        engine = AsyncLLM.from_engine_args(args)
        effective = engine.vllm_config
        self._effective_config = {
            "enable_prefix_caching": effective.cache_config.enable_prefix_caching,
            "gpu_memory_utilization": effective.cache_config.gpu_memory_utilization,
            "max_num_seqs": effective.scheduler_config.max_num_seqs,
            "max_model_len": effective.model_config.max_model_len,
            "dtype": str(effective.model_config.dtype).removeprefix("torch."),
            "enforce_eager": effective.model_config.enforce_eager,
            "generation_config": effective.model_config.generation_config,
            "quantization": effective.model_config.quantization,
            "tensor_parallel_size": effective.parallel_config.tensor_parallel_size,
        }
        self.tokenizer, self.resolved_revision = tokenizer, resolved
        return engine

    async def aload(self) -> None:
        if self.is_loaded:
            return
        async with self._load_lock:
            if not self.is_loaded:
                self.engine = await asyncio.to_thread(self._initialize)

    async def aclose(self) -> None:
        if self.engine is not None:
            engine, self.engine = self.engine, None
            await asyncio.to_thread(engine.shutdown)

    def _prepare(self, messages, tools):
        prompt = chat_prompt(self.tokenizer, messages, tools=tools,
                             enable_thinking=self.enable_thinking)
        return self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

    async def stream(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                     cancel: threading.Event, tools: list[dict[str, Any]] | None = None
                     ) -> AsyncIterator[ModelDelta | ModelDone]:
        if cancel.is_set():
            return
        await self.aload()
        from vllm import SamplingParams, TokensPrompt
        from vllm.sampling_params import RequestOutputKind

        token_ids = await asyncio.to_thread(self._prepare, messages, tools)
        params = SamplingParams(max_tokens=max_new_tokens,
                                temperature=self.temperature if self.do_sample else 0.0,
                                top_p=self.top_p if self.do_sample else 1.0,
                                stop_token_ids=[self.tokenizer.eos_token_id],
                                skip_special_tokens=True, output_kind=RequestOutputKind.DELTA)
        request_id = uuid4().hex
        stream = self.engine.generate(TokensPrompt(prompt_token_ids=token_ids), params, request_id)
        submitted_ns = time.perf_counter_ns()
        # Core timestamps use time.monotonic(), not wall clock time. Convert
        # their epoch to this server's perf_counter timeline explicitly.
        clock_offset_ns = time.perf_counter_ns() - time.monotonic_ns()
        first_ns = None
        output_tokens = 0
        finished = False
        pending = None
        try:
            while True:
                pending = asyncio.create_task(anext(stream))
                while not pending.done():
                    if cancel.is_set():
                        return
                    await asyncio.wait({pending}, timeout=0.05)
                try:
                    result = pending.result()
                except StopAsyncIteration:
                    raise RuntimeError("vLLM stream ended without completion metadata") from None
                now = time.perf_counter_ns()
                if cancel.is_set():
                    return
                if len(result.outputs) != 1:
                    raise RuntimeError("Expected one vLLM completion")
                output = result.outputs[0]
                output_tokens += len(output.token_ids)
                if output.token_ids and first_ns is None:
                    first_ns = now
                if output.text:
                    yield ModelDelta(output.text, now)
                if result.finished:
                    started_ns, finished_ns = submitted_ns, now
                    observer = "vllm_frontend_receipt_monotonic"
                    stats = result.metrics
                    if (stats is not None and stats.scheduled_ts > 0
                            and stats.first_token_ts >= stats.scheduled_ts
                            and stats.last_token_ts >= stats.first_token_ts):
                        started_ns = round(stats.scheduled_ts * 1e9) + clock_offset_ns
                        first_ns = round(stats.first_token_ts * 1e9) + clock_offset_ns
                        finished_ns = round(stats.last_token_ts * 1e9) + clock_offset_ns
                        observer = "vllm_engine_core_monotonic"
                    reason = output.finish_reason
                    hit = (output_tokens >= max_new_tokens) if reason == "length" else False if reason == "stop" else None
                    finished = True
                    yield ModelDone(model_id=self.model_id, model_revision=self.resolved_revision,
                                    started_ns=started_ns, first_token_ns=first_ns, finished_ns=finished_ns,
                                    input_tokens=len(token_ids), output_tokens=output_tokens,
                                    max_new_tokens=max_new_tokens, finish_reason=reason,
                                    hit_max_new_tokens=hit,
                                    truncated=reason == "length" if reason in {"stop", "length"} else None,
                                    timing_observer=observer)
                    return
        finally:
            if not finished:
                await self.engine.abort(request_id)
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()

    async def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                       tools: list[dict[str, Any]] | None = None,
                       cancel: threading.Event | None = None) -> tuple[str, ModelDone]:
        parts, completed = [], None
        stream = self.stream(messages, max_new_tokens=max_new_tokens,
                             tools=tools, cancel=cancel or threading.Event())
        try:
            async for event in stream:
                if isinstance(event, ModelDelta):
                    parts.append(event.text)
                else:
                    completed = event
        finally:
            await stream.aclose()
        if completed is None:
            raise RuntimeError("Generation was cancelled")
        return "".join(parts), completed
