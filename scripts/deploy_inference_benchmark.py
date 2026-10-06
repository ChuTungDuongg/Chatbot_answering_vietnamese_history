"""Deploy one isolated benchmark app without mutating the parent shell or production."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def deployment_environment(args) -> dict[str, str]:
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,60}", args.name) or args.name == "vn-history-rag-api":
        raise ValueError("Use a separate lowercase benchmark app name; production name is refused")
    environment = os.environ.copy()
    environment.update({
        "MODAL_APP_NAME": args.name, "MODAL_GPU_CLASS": args.gpu,
        "MODAL_MAX_INPUTS": "8",
        "MODAL_RUNTIME_IMAGE": args.image, "INFERENCE_BACKEND": args.backend,
        "MODEL_VARIANT": "vanilla", "RUNTIME_LOADING_STRATEGY": "eager",
        "ENABLE_HYBRID_MODE": str(args.mode == "hybrid").lower(),
        "ENABLE_CENTRAL_MODE": str(args.mode == "central").lower(),
        "RETRIEVAL_DENSE_BACKEND": "faiss", "RETRIEVAL_AVAILABLE_BACKENDS": "faiss",
        "VLLM_ENABLE_PREFIX_CACHING": str(args.prefix_caching).lower(),
        "VLLM_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        "VLLM_MAX_NUM_SEQS": str(args.max_num_seqs),
        "VLLM_ENFORCE_EAGER": str(args.enforce_eager).lower(),
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    })
    if args.max_model_len is not None:
        environment["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    else:
        environment.pop("VLLM_MAX_MODEL_LEN", None)
    return environment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["transformers", "vllm"], required=True)
    parser.add_argument("--mode", choices=["hybrid", "central"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--image", choices=["transformers", "vllm"], default="vllm",
                        help="Use the same vLLM-compatible bundle for both comparison backends")
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--enforce-eager", action="store_true", help="Explicit eager/debug variant; not the default baseline")
    parser.add_argument("--serve", action="store_true", help="Ephemeral dev endpoint instead of a separate deployment")
    args = parser.parse_args(argv)
    if args.backend == "vllm" and args.image != "vllm":
        parser.error("vLLM requires --image vllm")
    return subprocess.call([sys.executable, "-m", "modal", "serve" if args.serve else "deploy",
                            "modal_app.py"], env=deployment_environment(args),
                           cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    raise SystemExit(main())
