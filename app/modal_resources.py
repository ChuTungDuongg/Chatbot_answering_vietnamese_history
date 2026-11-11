"""Resolve Modal hardware without enabling local LLM inference implicitly."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModalResources:
    gpu: str | None
    device: str
    dtype: str


def resolve_modal_resources(local_inference_enabled: bool, *, local_gpu: str = "L4",
                            retrieval_gpu: str = "") -> ModalResources:
    retrieval_gpu = retrieval_gpu.strip()
    if retrieval_gpu not in {"", "T4", "L4"}:
        raise ValueError("MODAL_RETRIEVAL_GPU_CLASS must be empty, T4 or L4")
    gpu = local_gpu if local_inference_enabled else retrieval_gpu or None
    return ModalResources(gpu, "cuda" if gpu else "cpu",
                          "bfloat16" if local_inference_enabled else "float32")
