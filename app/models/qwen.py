"""Lazy, unadapted Qwen runtime with genuine Transformers token streaming."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any, AsyncIterator

from app.models.base import ModelDelta, ModelDone


class QwenRuntime:
    def __init__(self, *, model_id: str, revision: str | None = None,
                 device: str = "cpu", dtype: str = "bfloat16", cache_dir: str | None = None,
                 local_files_only: bool = False, do_sample: bool = False,
                 enable_thinking: bool = False):
        from app.config import CENTRAL_MODEL_ID, HYBRID_MODEL_ID

        if model_id not in {HYBRID_MODEL_ID, CENTRAL_MODEL_ID}:
            raise ValueError(f"Unsupported baseline model: {model_id}")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.do_sample = do_sample
        self.enable_thinking = enable_thinking
        self.model = None
        self.tokenizer = None
        self.resolved_revision: str | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def generation_settings(self) -> dict[str, Any]:
        return {"do_sample": self.do_sample, "enable_thinking": self.enable_thinking,
                "dtype": self.dtype, "quantization": None, "adapter": None}

    def load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            kwargs = {"revision": self.revision, "cache_dir": self.cache_dir,
                      "local_files_only": self.local_files_only, "trust_remote_code": True}
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **kwargs, dtype=getattr(torch, self.dtype),
                device_map={"": 0} if self.device == "cuda" else None,
                low_cpu_mem_usage=True, use_safetensors=True,
            )
            if self.device == "cpu":
                model.to("cpu")
            model.eval()
            self.resolved_revision = getattr(model.config, "_commit_hash", None) or self.revision
            self.tokenizer, self.model = tokenizer, model

    def _prepare(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None):
        self.load()
        template_options = {"tokenize": False, "add_generation_prompt": True,
                            "enable_thinking": self.enable_thinking}
        if tools:
            template_options["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **template_options)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        return inputs, int(inputs["input_ids"].shape[-1])

    async def stream(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                     cancel: threading.Event, tools: list[dict[str, Any]] | None = None
                     ) -> AsyncIterator[ModelDelta | ModelDone]:
        """Run `generate` on a worker and forward model-produced text as it arrives."""
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        inputs, input_tokens = await asyncio.to_thread(self._prepare, messages, tools)
        messages_queue: queue.Queue[tuple[str, Any, int]] = queue.Queue()
        runtime = self

        class TimedStreamer(TextIteratorStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                if text:
                    messages_queue.put(("delta", text, time.perf_counter_ns()))

        streamer = TimedStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        class TokenClock(StoppingCriteria):
            first_ns: int | None = None
            last_ns: int | None = None
            output_tokens = 0

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                now = time.perf_counter_ns()
                if self.first_ns is None:
                    self.first_ns = now
                self.last_ns = now
                self.output_tokens = max(0, int(input_ids.shape[-1]) - input_tokens)
                return cancel.is_set()

        clock = TokenClock()

        def generate_worker() -> None:
            started_ns = time.perf_counter_ns()
            try:
                while not cancel.is_set():
                    if runtime._generate_lock.acquire(timeout=0.1):
                        break
                else:
                    messages_queue.put(("cancelled", None, time.perf_counter_ns()))
                    return
                try:
                    if cancel.is_set():
                        messages_queue.put(("cancelled", None, time.perf_counter_ns()))
                        return
                    started_ns = time.perf_counter_ns()
                    with torch.inference_mode():
                        runtime.model.generate(
                            **inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                            do_sample=runtime.do_sample, use_cache=True,
                            pad_token_id=runtime.tokenizer.pad_token_id,
                            eos_token_id=runtime.tokenizer.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([clock]),
                        )
                finally:
                    runtime._generate_lock.release()
                finished_ns = clock.last_ns or time.perf_counter_ns()
                messages_queue.put(("done", ModelDone(
                    model_id=runtime.model_id, model_revision=runtime.resolved_revision,
                    started_ns=started_ns, first_token_ns=clock.first_ns,
                    finished_ns=finished_ns, input_tokens=input_tokens,
                    output_tokens=clock.output_tokens,
                ), time.perf_counter_ns()))
            except BaseException as exc:
                messages_queue.put(("error", exc, time.perf_counter_ns()))

        worker = threading.Thread(target=generate_worker, name="qwen-generate", daemon=False)
        worker.start()
        try:
            while True:
                try:
                    kind, value, produced_ns = await asyncio.to_thread(messages_queue.get, True, 0.1)
                except queue.Empty:
                    if cancel.is_set() and not worker.is_alive():
                        break
                    continue
                if kind == "delta":
                    yield ModelDelta(str(value), produced_ns)
                elif kind == "done":
                    yield value
                    break
                elif kind == "error":
                    raise value
                elif kind == "cancelled":
                    break
        finally:
            cancel.set()
            await asyncio.to_thread(worker.join, 5.0)

    async def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                       tools: list[dict[str, Any]] | None = None,
                       cancel: threading.Event | None = None) -> tuple[str, ModelDone]:
        parts: list[str] = []
        completed: ModelDone | None = None
        async for event in self.stream(messages, max_new_tokens=max_new_tokens,
                                       cancel=cancel or threading.Event(), tools=tools):
            if isinstance(event, ModelDelta):
                parts.append(event.text)
            else:
                completed = event
        if completed is None:
            raise RuntimeError("Generation was cancelled")
        return "".join(parts), completed
