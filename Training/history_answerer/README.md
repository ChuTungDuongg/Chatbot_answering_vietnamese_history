# 🏯 History Answerer Training

[⬅️ Training overview](../README.md) · [🏠 Project README](../../README.md)

History Answerer là LLM thứ ba. Nó nhận **text evidence đã được critic chọn**, không nhận embedding vector, rồi sinh answer tiếng Việt và source IDs.

## 🧩 Mapping cũ

- Old Phase 1 → `train_instruction_sft.py`.
- Old Phase 6 → `merge_phase1.py` + `train.py` + `loss.py` + `evaluate.py`.

## 1️⃣ Instruction SFT

```bash
python -m training.history_answerer.train_instruction_sft \
  --dataset Dataset/merged_jsonl/all_messages.jsonl \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --output-dir outputs/history_answerer/phase1
```

User tokens bị mask. Nếu target có `<analysis>` và `<final>`, analysis weight là 0.5 và final weight là 1.0. Hidden long-form reasoning không được đưa vào runtime response.

## 2️⃣ Grounded RAG-SFT

```bash
python -m training.history_answerer.train \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --phase1-adapter outputs/history_answerer/phase1 \
  --dataset-messages Dataset/merged_jsonl/all_messages.jsonl \
  --dataset-chunks training/Dataset/merged_jsonl/all_chunk_id.jsonl \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 5 \
  --learning-rate 1.5e-4 \
  --max-length 4096 \
  --output-dir outputs/history_answerer/phase6
```

`merge_phase1.py` load base model, gắn Phase 1 adapter, gọi `merge_and_unload()`, save intermediate base. `train.py` reload base đó ở 4-bit, gọi `prepare_model_for_kbit_training()`, tạo **LoRA mới** và mới bắt đầu Phase 6.

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
  --base-model outputs/history_answerer/phase6/phase1_merged_base \
  --adapter outputs/history_answerer/phase6/adapter \
  --output-dir outputs/history_answerer/merged
```

Không dùng Qwen2.5 base nguyên thủy ở bước này; base phải là intermediate đã merge Phase 1.

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
  --phase1-adapter outputs/history_answerer/phase1 \
  --output-dir outputs/history_answerer/phase6 \
  --resume-from-checkpoint outputs/history_answerer/phase6/checkpoint-500
```

Giữ nguyên `--merged-base-dir`/output và dataset để không merge sai nền model.
