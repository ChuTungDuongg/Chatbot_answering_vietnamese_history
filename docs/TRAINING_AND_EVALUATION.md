# Frozen SILVER V1: QLoRA, app, and six-system evaluation

The canonical benchmark contains **3,000 automatically annotated SILVER questions** from 60 batches. It is not human-reviewed GOLD. Batch 61 is not part of this experiment. The generated JSONL files below are local and Git-ignored; preserve their hashes when moving them.

## 1. Dataset and fixed split

| Artifact | Local path | Rows |
|---|---|---:|
| Canonical SILVER | `evaluation/datasets/v1_silver/questions_3000.jsonl` | 3000 |
| TRAIN | `evaluation/datasets/v1_silver/splits/v1_seed42/train.jsonl` | 2400 |
| VALIDATION | `evaluation/datasets/v1_silver/splits/v1_seed42/validation.jsonl` | 300 |
| TEST | `evaluation/datasets/v1_silver/splits/v1_seed42/test.jsonl` | 300 |
| TRAIN SFT | `training/datasets/vn_history_rag_sft_v1/train_sft.jsonl` | 3000 variants of 2400 IDs |
| VALIDATION SFT | `training/datasets/vn_history_rag_sft_v1/validation_sft.jsonl` | 300 |

Canonical SHA-256: `97acf491e7409e54ed27f38f1e28a16b9e85107c9c2ed296bffaf48814d539d8`. Split seed: `42`. The split uses greedy source-group stratification; see `split_manifest.json` and `leakage_report.json` for the actual distributions and remaining overlaps. Training examples derive only from TRAIN. TEST is never used for training or style augmentation.

The SFT representation has conversational `messages` with real Corpus V1 passages and an explicit response-mode instruction in the system message. The three modes are `concise`, `standard`, `detailed`, independent of question difficulty. The canonical `gold_answer` is not overwritten. The generated references remain SILVER, even when used for scoring. SFT content is deterministic presentation of SILVER answers and must not be interpreted as newly verified historical facts.

To validate local frozen artifacts:

```bash
python -m evaluation.silver_v1 validate
```

## 2. Mounted Google Drive layout

On **Google Colab**, mount Drive in a Python cell first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then, from the repository root:

```bash
python -m tools.drive_cli init --drive-root /content/drive/MyDrive/VN_History_LLM
```

The root is `/content/drive/MyDrive/VN_History_LLM`. Its layout includes `datasets/canonical`, `datasets/splits/v1_seed42`, `datasets/sft`, `models/qwen3_4b_sft_v1`, `evaluation/predictions`, `evaluation/reports`, and `logs`. A CLI cannot mount Drive by itself. On a local machine with a synchronized Drive filesystem, pass its actual mount path to `--drive-root`.

Run this **where the local ignored dataset files already exist and the Drive root is mounted**:

```bash
python -m tools.drive_cli push-datasets --drive-root /content/drive/MyDrive/VN_History_LLM
python -m tools.drive_cli verify --drive-root /content/drive/MyDrive/VN_History_LLM
```

On Colab, if the repository checkout lacks the ignored files, restore from the mounted Drive:

```bash
python -m tools.drive_cli pull-datasets --drive-root /content/drive/MyDrive/VN_History_LLM
python -m tools.drive_cli verify --drive-root /content/drive/MyDrive/VN_History_LLM
```

Each copy prints source, destination, bytes and SHA-256. `verify` requires both local and Drive copies. An existing file with a different hash stops the command; `--overwrite` is the explicit replacement switch. The CLI also supports `push-adapter`, `pull-adapter`, `push-evaluation`, `pull-evaluation` with `--local-path` and a single-directory `--name`. It does not copy Corpus V1, FAISS/BM25 files, model caches, or secrets. Provision Corpus V1 separately for RAG evaluation. For the commands below, the existing V1 corpus, runtime files and retrieval indexes must be accessible at `/content/drive/MyDrive/VN_History_LLM/corpus_v1/`; this is a one-time transfer of existing artifacts, not a rebuild.

## 3. QLoRA prerequisites and smoke test

Use a CUDA Colab runtime and install a CUDA-compatible PyTorch build, then:

```bash
pip install -r requirements-training.txt
python -m training.train_qwen3 --train-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/train_sft.jsonl --validation-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/validation_sft.jsonl --output-dir /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/smoke_01 --max-train-samples 16 --max-eval-samples 8 --fast-dev-run --dry-run
```

The dry run verifies SFT and split hashes/IDs and prints the effective batch and estimated steps without loading Qwen. Then execute the small GPU smoke:

```bash
python -m training.train_qwen3 --train-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/train_sft.jsonl --validation-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/validation_sft.jsonl --output-dir /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/smoke_01 --max-train-samples 16 --max-eval-samples 8 --fast-dev-run
```

At training time, the frozen SFT `messages` are converted in memory to a conversational `prompt` (system + user/context) and `completion` (assistant answer), for both train and validation. TRL uses the tokenizer's chat template and computes loss on the completion only (`completion_only_loss=True`, `assistant_only_loss=False`). This does not require `{% generation %}` assistant-mask markers in the tokenizer template. The trainer also uses 4-bit NF4 with double quantization, bf16 when supported (otherwise fp16), and LoRA over the attention/MLP projection modules. No special chat tokens are constructed manually, and the frozen SFT files and split artifacts remain unchanged. The smoke runs one training step plus validation. The GPU training path still needs to be rechecked on Colab after this fix.

## 4. Full training and resume

L4 starting preset (adjust after smoke; no single batch size is universally optimal):

```bash
python -m training.train_qwen3 --model-id Qwen/Qwen3-4B-Instruct-2507 --train-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/train_sft.jsonl --validation-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/validation_sft.jsonl --output-dir /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/run_01 --per-device-train-batch-size 2 --per-device-eval-batch-size 2 --gradient-accumulation-steps 8 --epochs 2 --learning-rate 2e-4 --max-seq-length 2048 --gradient-checkpointing --no-packing --logging-steps 5 --eval-steps 50 --save-steps 50
```

A100-oriented starting preset:

```bash
python -m training.train_qwen3 --model-id Qwen/Qwen3-4B-Instruct-2507 --train-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/train_sft.jsonl --validation-file /content/drive/MyDrive/VN_History_LLM/datasets/sft/validation_sft.jsonl --output-dir /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/run_a100_01 --per-device-train-batch-size 4 --per-device-eval-batch-size 4 --gradient-accumulation-steps 4 --epochs 2 --learning-rate 2e-4 --max-seq-length 3072 --gradient-checkpointing --no-packing --logging-steps 5 --eval-steps 50 --save-steps 50
```

Effective batch = per-device train batch × gradient accumulation × number of devices. Larger per-device batches can improve throughput but require more VRAM; sequence length and packing also affect memory. CLI flags expose weight decay, warmup, LoRA rank/alpha/dropout, packing, data-loader workers, precision, logging/evaluation/checkpoint intervals, and sample caps. Console and `training_log.jsonl` report loss, step time, approximate examples/s, GPU peak memory, elapsed and ETA. `checkpoints/checkpoint-*` are resumable. `adapter/` contains the PEFT adapter; a large merged model is not created automatically.

Resume the same command with the **same parameters and output path**, adding:

```bash
--resume-from-checkpoint latest
```

Changing model, dataset hash, or training config causes a resume error. Completed runs cannot be overwritten silently. If the process stopped before its first checkpoint, `latest` restarts the same verified configuration from step zero. The run `manifest.json` records model ID/resolved revision, data hashes, LoRA config, seed, library versions and GPU identity. The optional merge below creates a **large** standalone model only when explicitly run:

```bash
python -m training.merge_adapter --model-id Qwen/Qwen3-4B-Instruct-2507 --adapter-path /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/run_01/adapter --output-dir /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/merged_01 --dtype float16
```

## 5. App response mode and adapter

The app's three-position slider is **Ngắn gọn / Tiêu chuẩn / Chi tiết**, stored in browser localStorage. `response_mode` defaults to `standard`; API rejects other values. It affects the shared prompt instruction and SSE metadata, while retrieval settings and generation sampling remain independent. The existing Hybrid/Central mode selector remains separate.

For a trained Hybrid 4B answer (with local Corpus V1 retrieval metadata prepared):

```bash
APP_MODE=full CORPUS_PATH=/content/drive/MyDrive/VN_History_LLM/corpus_v1/chunks.jsonl RETRIEVAL_ROOT=/content/drive/MyDrive/VN_History_LLM/corpus_v1/retrieval RETRIEVAL_DENSE_BACKEND=faiss MODEL_VARIANT=sft MODEL_ADAPTER_PATH=/content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/run_01/adapter DEVICE=cuda uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set `MODEL_VARIANT=vanilla` and omit `MODEL_ADAPTER_PATH` for vanilla Hybrid 4B. Central Agent remains its existing 8B model. Use platform-specific environment assignment syntax on Windows. Adapter loading requires a matching base model and `adapter_config.json`.

## 6. Fixed six-system quality benchmark

The registry has exactly: `vanilla_no_rag`, `vanilla_faiss`, `vanilla_qdrant`, `sft_no_rag`, `sft_faiss`, `sft_qdrant`. Only model variant and RAG backend vary. Primary quality uses the frozen TEST split, `response_mode=standard`, seed 42, greedy decoding (`temperature=0`), same final-k and retrieval config. No-RAG prompts contain no retrieved chunks and never claim corpus citations. Their retrieval, evidence-grounding and citation metrics are **N/A**, while answer similarity/required-fact phrase and abstention checks are still reported.

Model-free configuration smoke, safe on a CPU host:

```bash
python -m evaluation.six_way --test-file evaluation/datasets/v1_silver/splits/v1_seed42/test.jsonl --systems vanilla_no_rag --max-questions 3 --output-dir reports/six_way_v1_smoke --dry-run
```

Three-question GPU smoke after adapter and Corpus V1 artifacts are available:

```bash
python -m evaluation.six_way --test-file /content/drive/MyDrive/VN_History_LLM/datasets/splits/v1_seed42/test.jsonl --adapter-path /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/run_01/adapter --corpus-path /content/drive/MyDrive/VN_History_LLM/corpus_v1/chunks.jsonl --retrieval-root /content/drive/MyDrive/VN_History_LLM/corpus_v1/retrieval --systems vanilla_faiss,sft_faiss --max-questions 3 --response-mode standard --output-dir /content/drive/MyDrive/VN_History_LLM/evaluation/reports/six_way_smoke
```

Full primary benchmark after Qdrant is finalized:

```bash
python -m evaluation.six_way --test-file /content/drive/MyDrive/VN_History_LLM/datasets/splits/v1_seed42/test.jsonl --adapter-path /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/run_01/adapter --corpus-path /content/drive/MyDrive/VN_History_LLM/corpus_v1/chunks.jsonl --retrieval-root /content/drive/MyDrive/VN_History_LLM/corpus_v1/retrieval --systems all --response-mode standard --output-dir /content/drive/MyDrive/VN_History_LLM/evaluation/reports/six_way_v1
```

Append `--resume` to the **same command** after interruption. Each prediction is appended and flushed before advancing. Identity checks cover TEST hash, adapter fingerprint, model/decoding settings and backend manifests; a changed run must use a new output directory. `--overwrite` explicitly replaces an existing run. `--start-after`, `--max-questions`, and batch/concurrency flags exist; the primary quality runner currently enforces batch sizes and concurrency of 1. Use `benchmarks.latency.runner` separately for throughput, TTFB, TTFT, TPOT and repeated latency experiments.

If you explicitly created a merged model, replace `--adapter-path ...` with `--sft-model-path /content/drive/MyDrive/VN_History_LLM/models/qwen3_4b_sft_v1/merged_01`. Vanilla systems continue to load the unmodified base ID. The merged model files are fingerprinted in the run identity.

Outputs: `run_manifest.json`, `retrieval_cache_{faiss,qdrant}.jsonl` with identity manifests, each system's `predictions.jsonl`, `progress.json`, `progress.log`, `metrics.json`, and aggregate `summary.json`/`summary.csv`. Metrics include Exact Match, token F1, ROUGE-L, lexical required-fact phrase coverage, abstention behavior, retrieval HitRate/Recall/Precision/MRR/nDCG at 1/3/5/10, grounding and citations where applicable. Breakdown is by difficulty, answerability, and category; small groups should not be overinterpreted. The runner never declares a “best” model automatically.

## 7. Qdrant and limitations

The local directory `artifacts/corpus_v1/retrieval/qdrant.partial/` is incomplete. Qdrant runs require finalized `artifacts/corpus_v1/retrieval/qdrant/manifest.json`, a matching Corpus V1 identity, remote collection/vector/HNSW configuration and exact point count. There is **no FAISS fallback**. After finalization, set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` in the environment (or Colab Secrets); never put the key in commands, logs, or Git. FAISS and no-RAG systems can be tested independently now.

If Colab reports missing corpus/index files, mount or separately transfer the user's existing Corpus V1 artifacts; `drive_cli` intentionally transfers only the small benchmark/model/evaluation artifacts. If a full six-way run fails at startup, inspect the Qdrant manifest and collection readiness before retrying. If training runs out of VRAM, reduce per-device batch or sequence length, keep gradient checkpointing enabled, and adjust accumulation to preserve a comparable effective batch. Record any changed settings in a new run, never silently resume with different configuration.

This SILVER benchmark can inherit question-generation, retriever and model biases. FAISS/Qdrant agreement does not establish historical truth. Compare findings against the separate human-reviewed GOLD audit set before drawing strong quality conclusions.

Training API references: [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization), and the [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
