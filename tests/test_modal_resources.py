"""Keep retrieval acceleration independent of local language-model loading."""
import pytest


@pytest.mark.parametrize("retrieval_gpu", ["T4", "L4"])
def test_remote_inference_can_accelerate_retrieval(retrieval_gpu):
    from app.modal_resources import resolve_modal_resources
    resources = resolve_modal_resources(False, local_gpu="A100-40GB", retrieval_gpu=retrieval_gpu)
    assert resources.gpu == retrieval_gpu
    assert resources.device == "cuda"
    assert resources.dtype == "float32"  # T4 need not support bfloat16.


def test_remote_inference_still_supports_cpu_only():
    from app.modal_resources import resolve_modal_resources
    resources = resolve_modal_resources(False)
    assert resources.gpu is None
    assert resources.device == "cpu"
    assert resources.dtype == "float32"


def test_local_llm_uses_explicit_local_gpu_over_retrieval_gpu():
    from app.modal_resources import resolve_modal_resources
    resources = resolve_modal_resources(True, local_gpu="A100-40GB", retrieval_gpu="T4")
    assert resources.gpu == "A100-40GB"
    assert resources.device == "cuda"
    assert resources.dtype == "bfloat16"


def test_invalid_retrieval_gpu_cannot_silently_provision_another_accelerator():
    from app.modal_resources import resolve_modal_resources
    with pytest.raises(ValueError, match="MODAL_RETRIEVAL_GPU_CLASS"):
        resolve_modal_resources(False, retrieval_gpu="typo")
