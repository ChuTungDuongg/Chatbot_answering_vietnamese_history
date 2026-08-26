# 🧹 Evidence Critic / Compressor Training

[⬅️ Training overview](../README.md) · [🧠 Runtime agents](../../app/agents/README.md)

Evidence Agent dùng Qwen3 adapter riêng. Nó chỉ được lọc/nén evidence đã cung cấp, không được thêm kiến thức mới hoặc tạo evidence ID mới.

## 🧾 Chuẩn bị dataset

```bash
python -m training.evidence_agent.prepare_dataset \
  --input Dataset/merged_jsonl/all_messages.jsonl \
  --output datasets/evidence_agent/train.jsonl
```

Một row có dạng:

```json
{
  "question": "...",
  "evidence": [{"evidence_id":"c1","source_type":"local","text":"..."}],
  "output": {
    "status": "sufficient",
    "selected_evidence": [{"evidence_id":"c1","relevance":1.0,"claims":[],"compressed_text":"..."}],
    "conflicts": [],
    "missing_information": [],
    "summary": "..."
  }
}
```

Để tăng chất lượng, có thể bổ sung HotpotQA supporting facts, retrieved negatives, duplicate/partially relevant chunks, insufficient rows và conflict rows có provenance rõ ràng.

## 🚂 Train

```bash
python -m training.evidence_agent.train \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --dataset datasets/evidence_agent/train.jsonl \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --max-length 4096 \
  --epochs 2 \
  --output-dir outputs/evidence_agent
```

## 📏 Evaluate

```bash
python -m training.evidence_agent.evaluate \
  --gold datasets/evidence_agent/gold.jsonl \
  --predictions predictions/evidence_agent.jsonl
```

Metric hiện có là selected evidence ID F1. Runtime Pydantic validation kiểm tra schema, selected/rejected không chồng nhau và mọi selected ID phải có trong candidate evidence.

## 📦 Output

Upload `outputs/evidence_agent/` vào `/evidence_agent/adapter` trên Modal Volume. Research và Evidence adapters phải được train trên cùng base model ID nếu dùng shared runtime.
