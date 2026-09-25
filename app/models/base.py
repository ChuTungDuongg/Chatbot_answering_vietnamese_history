"""Generation events shared by both inference modes."""

from dataclasses import dataclass


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

    @property
    def metrics(self) -> dict[str, float | int | None]:
        generation_ms = (self.finished_ns - self.started_ns) / 1e6
        model_ttft_ms = (self.first_token_ns - self.started_ns) / 1e6 if self.first_token_ns else None
        decode_ms = (self.finished_ns - self.first_token_ns) / 1e6 if self.first_token_ns else None
        return {
            "model_ttft_ms": model_ttft_ms,
            "generation_ms": generation_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.output_tokens / (generation_ms / 1000) if generation_ms > 0 else None,
            "decode_tokens_per_second": (self.output_tokens - 1) / (decode_ms / 1000)
                if decode_ms is not None and decode_ms > 0 and self.output_tokens > 1 else None,
            "tpot_ms": decode_ms / (self.output_tokens - 1)
                if decode_ms is not None and self.output_tokens > 1 else None,
        }
