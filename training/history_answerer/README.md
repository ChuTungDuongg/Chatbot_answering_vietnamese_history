# 🏯 History Answerer Training

[⬅️ Training overview](../README.md) · [🏠 Project README](../../README.md)

History Answerer là LLM thứ ba. Nó nhận **text evidence đã được critic chọn**, không nhận embedding vector, rồi sinh answer tiếng Việt và source IDs.

## 🧩 Mapping và policy hiện tại

- Old Phase 1 → `train_instruction_sft.py`, chỉ để tái hiện legacy.
- Old Phase 6 → `train.py` + `loss.py` + `evaluate.py`, hiện train thẳng từ vanilla Qwen2.5.

## 1️⃣ Grounded RAG-SFT chính

```bash
python -m training.history_answerer.train \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-messages Dataset/merged_jsonl/all_messages.jsonl \
  --dataset-chunks training/Dataset/merged_jsonl/all_chunk_id.jsonl \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 5 \
  --output-dir outputs/history_answerer/phase6
```

`train.py` load vanilla `Qwen/Qwen2.5-3B-Instruct` ở 4-bit, chuẩn bị k-bit training rồi gắn fresh LoRA. Nó không đọc, merge hoặc resume từ Phase 1 adapter.

## 2️⃣ Instruction SFT Phase 1 legacy, tùy chọn

```bash
python -m training.history_answerer.train_instruction_sft \
  --dataset Dataset/merged_jsonl/all_messages.jsonl \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --output-dir outputs/history_answerer/phase1_legacy
```

User tokens bị mask. Nếu target có `<analysis>` và `<final>`, analysis weight là 0.5 và final weight là 1.0. Adapter này không đi vào flow Phase 6 mới.

## ⚖️ Weighted loss

`loss.py` tạo `labels=-100` cho user/padding tokens. Assistant tokens:

| Segment | Weight |
|---|---:|
| `Nguồn được dùng:` | 1.6 |
| `Trả lời:` và answer body | 1.0 |

`WeightedCETrainer` áp dụng token-level cross entropy rồi chuẩn hóa theo tổng weight thực tế.

## 🔗 Merge model cuối

```bash
python -m training.scripts.merge_model \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/history_answerer/phase6/adapter \
  --output-dir outputs/history_answerer/merged
```

Merge chỉ để deploy: Phase 6 adapter được merge vào đúng vanilla base đã dùng khi train. `merge_phase1.py` còn lại như legacy compatibility CLI, không phải bước bắt buộc.

## 📏 Evaluation

Prediction JSONL mỗi dòng cần `answer`, `assistant` hoặc `prediction`. Gold có thể là chat `messages`.

```bash
python -m training.history_answerer.evaluate \
  --gold artifacts/training/history_answerer/messages_normalized.jsonl \
  --predictions predictions/history_answerer.jsonl \
  --output reports/history_answerer.json
```

Metrics: source exact/P/R/F1, format OK, answer non-empty, source ID tồn tại, insufficient empty-rate, ROUGE-L, generation composite; `eval_loss`/`test_loss` được tổng hợp nếu prediction/evaluation rows chứa hai trường này.

## ♻️ Resume

```bash
python -m training.history_answerer.train \
  --output-dir outputs/history_answerer/phase6 \
  --resume-from-checkpoint outputs/history_answerer/phase6/checkpoint-500
```

Giữ nguyên `--model-id`, output, dataset và split seed. Resume checkpoint thuộc chính Phase 6, không dùng Phase 1 checkpoint.
