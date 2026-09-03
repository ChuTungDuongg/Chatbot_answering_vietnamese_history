# Shared Qwen3 multi-adapter serving

The active architecture is one frozen `Qwen/Qwen3-4B-Instruct-2507` base with three independent LoRA adapters:

- `research`: selects policy actions and tools; never writes the final history answer.
- `evidence`: selects, validates, deduplicates, detects conflicts, and compresses evidence using `EvidenceModelOutput`; never orchestrates retrieval or writes final prose.
- `history`: writes the final grounded Vietnamese answer and citations; never retrieves or manages the evidence pool.

The role registry is `app/agents/common/model_registry.py`. Adapter startup validation reads each `adapter_config.json` and rejects any `base_model_name_or_path` other than the shared Qwen3 ID. The old Qwen2.5 History model is explicitly `legacy_only` and may be retained only for same-input benchmark comparisons.

## Transformers/PEFT backend

Set:

```text
LLM_BACKEND=transformers
SHARED_BASE_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507
RESEARCH_AGENT_ADAPTER_PATH=/artifacts/adapters/research
EVIDENCE_AGENT_ADAPTER_PATH=/artifacts/adapters/evidence
HISTORY_AGENT_ADAPTER_PATH=/artifacts/adapters/history
```

The runtime loads one quantized base, attaches all three adapters, and selects the requested adapter explicitly for each sequential call. A missing role or mismatched adapter fails before serving. Hybrid retrieval (E5, FAISS, BM25S, fusion, and BGE reranking) remains independent and unchanged.

## Future vLLM backend

vLLM is optional deployment software and is intentionally not in core training requirements. After pinning and testing a compatible vLLM release, a future server command follows the documented multi-LoRA shape:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules research=/artifacts/adapters/research \
                 evidence=/artifacts/adapters/evidence \
                 history=/artifacts/adapters/history
```

Confirm the exact CLI syntax against the installed vLLM version before deployment. Point the orchestrator at its OpenAI-compatible endpoint:

```text
LLM_BACKEND=vllm
VLLM_BASE_URL=http://vllm:8000/v1
```

Logical requests use model names `research`, `evidence`, and `history`. Generation limits remain role-specific. The current code supports `/chat/completions` for message inputs and `/completions` for the existing pre-rendered History prompt.

QLoRA training format does not guarantee that every quantized serving combination works. Validate Qwen3 architecture support, serving dtype/quantization, vLLM version, LoRA rank, target modules, tokenizer/chat template, and quantization backend together. No vLLM server or GPU benchmark was run as part of this change.

The deployment artifact contains adapters, retrieval indexes, corpus, registry/config, and manifest. Base weights remain in the Hugging Face cache/volume and are not copied three times.
