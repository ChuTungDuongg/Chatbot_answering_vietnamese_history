# 🧠 Training Pipelines

[🏠 README gốc](../README.md) · [🧠 Central Agent](central_agent/README.md) · [🧭 History Answerer](history_answerer/README.md) · [🔎 Research Agent](research_agent/README.md) · [🧹 Evidence Agent](evidence_agent/README.md) · [🛠️ Data scripts](scripts/README.md)

Thư mục `training/` thay thế hoàn toàn workflow notebook Phase 1-10. Package viết thường có chủ đích để `python -m training...` chạy giống nhau trên Windows, Linux, Colab và Modal.

## 🗂️ Cấu trúc

```text
training/
├── common/
│   ├── cli.py                 # shared CLI flags + validation
│   ├── qlora.py               # NF4 4-bit và LoRA config
│   ├── trainer.py             # TrainingArguments + JSONL/GPU logging
│   ├── sft.py                 # generic assistant-only CE for policy agents
│   ├── datasets.py            # load/split chat rows
│   └── jsonl.py               # typed JSONL I/O
├── central_agent/           # final Qwen3-8B canonical-trajectory QLoRA CLI
│   ├── cli.py                 # orchestration: dry-run, preflight, train, resume
│   ├── config.py              # CLI/JSON config resolution + validation
│   ├── data.py                # canonical validation, split stats, SHA256, token audit
│   ├── runtime.py             # GPU/disk/manifest/checkpoint safeguards
│   ├── engine.py              # QLoRA model, Trainer, eval, adapter exports
│   └── configs/               # explicit L4/A100 starting profiles
├── history_answerer/          # fresh Qwen3 grounded RAG-SFT; Qwen2.5 Phase-1 legacy only
├── research_agent/            # Qwen3 tool policy
├── evidence_agent/            # Qwen3 critic/compressor
├── scripts/                   # corpus/index/merge/export/benchmark
└── Dataset/                   # 520 context rows used by the migrated workflow
```

## ⚙️ Cài dependency

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

Không cài Jupyter. Chạy mọi lệnh tại repository root.

## ✅ Kiểm tra trước GPU

```bash
python -m training.history_answerer.train --help
python -m training.research_agent.train --help
python -m training.research_agent.validate_dataset --help
python -m training.research_agent.preflight --help
python -m training.evidence_agent.train --help

python -m training.history_answerer.prepare_dataset
python -m training.history_answerer.validate_dataset
python -m training.history_answerer.preflight
python -m training.history_answerer.train --max-samples 10 --dry-run
```

Dry-run validate dataset/split và chạy tokenizer path thật, nhưng không tải model weights.

## 🧠 Final central Qwen3-8B agent

Trainer cuối cùng cho canonical trajectories được tách thành package `central_agent/` để dễ maintain và debug. Lệnh tương thích cũ vẫn là entry point chính:

```bash
python -m training.train_qwen3_8b_agent --help
```

Lệnh module tương đương:

```bash
python -m training.central_agent.cli --help
```

Ba chế độ tách biệt rõ ràng:

- `--dry-run`: kiểm tra config, path và schema; không load tokenizer/model.
- `--preflight-only`: load tokenizer và audit canonical loss/truncation; không load model weights.
- Không có hai flag trên: GPU preflight rồi train QLoRA.

Ví dụ validate dataset trên Google Drive mà không dùng GPU:

```bash
python -m training.train_qwen3_8b_agent \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --dry-run
```

Tokenizer preflight:

```bash
python -m training.train_qwen3_8b_agent \
  --dataset-root /content/drive/MyDrive/vn-history/trajectory_dataset_FINAL/final \
  --drive-root /content/drive/MyDrive/vn-history \
  --run-name qwen3-8b-agent-v1 \
  --max-seq-length 4096 \
  --preflight-only
```

Trainer hỗ trợ JSON config; precedence là defaults `<` `--config` `<` CLI flags. Các config L4/A100 trong `central_agent/configs/` chỉ là starting profiles tường minh, không tự áp dụng theo GPU. Xem toàn bộ lệnh train/resume, cấu trúc output và Colab recipe tại [Central Agent README](central_agent/README.md).

## 🎓 Thứ tự train đề xuất

```text
Một vanilla Qwen3-4B shared base
  → fresh Evidence adapter
  → corrective Research history-policy adapter
  → fresh History grounded adapter

History trajectories → Research Agent QLoRA
History evidence rows → Evidence Agent QLoRA

Build corpus/index
  → export bundle ba adapter (không bundle ba base copies)
  → Modal Volume
```

Ba pipeline dùng QLoRA 4-bit NF4, double quantization và PEFT LoRA. Research Agent dùng generic assistant-only CE; loss có trọng số theo format lịch sử chỉ thuộc History Answerer.

## 📊 Training logs và checkpoint

Mỗi output directory chứa `training_log.jsonl`. Callback ghi:

```json
{"step":100,"epoch":0.4,"loss":1.2,"learning_rate":0.00008,"gpu_allocated_gb":8.1,"gpu_reserved_gb":9.0}
```

Trainer tạo `checkpoint-<step>/` gồm adapter/model state, tokenizer khi được save, optimizer, scheduler, RNG, trainer state và training args. Resume:

```bash
python -m training.evidence_agent.train \
  --dataset datasets/evidence_agent/train.jsonl \
  --output-dir outputs/evidence_agent \
  --resume-from-checkpoint outputs/evidence_agent/checkpoint-500
```

Không đổi dataset, seed hoặc model ID giữa hai lần resume.

## 🎛️ Shared CLI

| Nhóm | Flags |
|---|---|
| Model/data | `--model-id`, `--dataset`, `--output-dir`, `--max-samples`, `--seed` |
| Optimization | `--epochs`, `--batch-size`, `--eval-batch-size`, `--gradient-accumulation-steps`, `--learning-rate`, `--weight-decay`, `--warmup-ratio` |
| Sequence/LoRA | `--max-length`, `--lora-r`, `--lora-alpha`, `--lora-dropout` |
| Logging/save | `--logging-steps`, `--eval-steps`, `--save-steps`, `--resume-from-checkpoint` |
| Precision | `--bf16`, `--no-bf16`, `--fp16`, `--no-fp16`, `--gradient-checkpointing` |
| Research precision | `--bnb-compute-dtype {auto,float16,bfloat16,float32}` |
| Optional | `--report-to wandb`, `--dry-run` |

Alias `--lr` và `--grad-accum-steps` được giữ để tương thích lệnh cũ, nhưng tài liệu dùng tên đầy đủ.

## 🧮 Effective batch size

```text
effective_batch_size = batch_size × gradient_accumulation_steps × GPU count
```

Ví dụ một GPU, batch 1 và accumulation 16 cho effective batch 16. Khi OOM, ưu tiên giảm max length và batch trước; không khẳng định cấu hình nào chắc chắn fit nếu chưa đo trên đúng GPU/model revision.

## ☁️ Colab recipe

```bash
git clone <YOUR_REPOSITORY_URL>
cd Chatbot_answering_vietnamese_history
pip install -r requirements-training.txt
```

T4:

```bash
python -m training.research_agent.train \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 2048 \
  --no-bf16 --fp16 \
  --output-dir /content/drive/MyDrive/vn-history/research_agent
```

L4/A100 dùng `--bf16 --no-fp16` sau khi preflight xác nhận BF16 support. `--bnb-compute-dtype auto` giữ Trainer và BitsAndBytes cùng dtype. Luôn lưu output lên Drive nếu cần chống mất checkpoint khi Colab disconnect.

## 🧳 Migration Phase 1-10

| Phase cũ | Đích Python | Behavior giữ lại |
|---|---|---|
| 1 | `history_answerer/train_instruction_sft.py` | Legacy optional; không còn là prerequisite của Phase 6 |
| 2 | `scripts/build_corpus.py` | Build corpus JSONL từ chunk packs |
| 3 | `scripts/build_corpus.py` + JSONL utils | Chunk/export và dedup theo `chunk_id` |
| 4 | cùng scripts Phase 3 | Extra-topic packs |
| 5 | `common/jsonl.py`, dataset preparation | Merge/normalize JSONL |
| 6 | `history_answerer/prepare_dataset.py`, `validate_dataset.py`, `preflight.py`, `train.py`, `loss.py`, `evaluate.py` | Vanilla Qwen3 → fresh History LoRA, source weight 1.6 |
| 7 | evaluate/benchmark CLIs | Inference sanity và metrics |
| 8 | `scripts/enrich_corpus.py` | Metadata/year enrichment |
| 9 | `scripts/build_index.py`, `app/rag/retrieval.py` | FAISS + BM25S + hybrid runtime |
| 10 | `scripts/export_artifacts.py` | Ba role adapters 4B + optional future Central V2 adapter 8B + retrieval/corpus |

Không còn `.ipynb` hoặc notebook archive trong source project; mọi workflow bắt buộc đều là Python CLI.

## 🔍 Verification

```bash
python -m compileall training
python -m pytest -q tests/test_training_cli.py tests/test_evidence_schema.py
```

Các test CLI chỉ gọi `--help`; chúng không tải Qwen hoặc Hugging Face dataset.
