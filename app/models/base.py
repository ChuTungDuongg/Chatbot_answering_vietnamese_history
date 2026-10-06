"""Generation events shared by both inference modes."""

from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol
import threading


@dataclass(frozen=True)
class ModelDelta:
    text: str
    produced_ns: int


@dataclass(frozen=True)
class ModelDone:
    model_id: str
    model_revision: str | None
    started_ns: int
    first_token_ns: int | None
    finished_ns: int
    input_tokens: int
    output_tokens: int
    max_new_tokens: int | None = None
    finish_reason: str | None = None
    hit_max_new_tokens: bool | None = None
    truncated: bool | None = None
    timing_observer: str | None = None

    @property
    def metrics(self) -> dict[str, float | int | str | bool | None]:
        generation_ms = (self.finished_ns - self.started_ns) / 1e6
        model_ttft_ms = (self.first_token_ns - self.started_ns) / 1e6 if self.first_token_ns else None
        decode_ms = (self.finished_ns - self.first_token_ns) / 1e6 if self.first_token_ns else None
        return {
            "model_ttft_ms": model_ttft_ms,
            "generation_ms": generation_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "max_new_tokens": self.max_new_tokens,
            "finish_reason": self.finish_reason,
            "hit_max_new_tokens": self.hit_max_new_tokens,
            "truncated": self.truncated,
            "model_timing_observer": self.timing_observer,
            "tokens_per_second": self.output_tokens / (generation_ms / 1000) if generation_ms > 0 else None,
            "decode_tokens_per_second": (self.output_tokens - 1) / (decode_ms / 1000)
                if decode_ms is not None and decode_ms > 0 and self.output_tokens > 1 else None,
            "tpot_ms": decode_ms / (self.output_tokens - 1)
                if decode_ms is not None and self.output_tokens > 1 else None,
        }


def chat_prompt(tokenizer: Any, messages: list[dict[str, Any]], *,
                tools: list[dict[str, Any]] | None, enable_thinking: bool) -> str:
    """The same Qwen chat template for both engines, including Hermes tools."""
    options = {"tokenize": False, "add_generation_prompt": True,
               "enable_thinking": enable_thinking}
    if tools:
        options["tools"] = tools
    return tokenizer.apply_chat_template(messages, **options)


class ModelRuntime(Protocol):
    model_id: str
    resolved_revision: str | None
    adapter_path: Any
    adapter_fingerprint: str | None
    inference_backend: str

    @property
    def is_loaded(self) -> bool: ...
    @property
    def model_variant(self) -> str: ...
    @property
    def adapter_attached(self) -> bool: ...
    @property
    def generation_settings(self) -> dict[str, Any]: ...
    @property
    def engine_metadata(self) -> dict[str, Any]: ...
    async def aload(self) -> None: ...
    async def aclose(self) -> None: ...
    def stream(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
               cancel: threading.Event, tools: list[dict[str, Any]] | None = None
               ) -> AsyncIterator[ModelDelta | ModelDone]: ...
    async def generate(self, messages: list[dict[str, Any]], *, max_new_tokens: int,
                       tools: list[dict[str, Any]] | None = None,
                       cancel: threading.Event | None = None) -> tuple[str, ModelDone]: ...
