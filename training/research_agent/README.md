# 🔎 Research / Tool Agent Training

[⬅️ Training overview](../README.md) · [🧰 Runtime tools](../../app/tools/README.md)

Research Agent dùng `Qwen/Qwen3-4B-Instruct-2507` mặc định. Nhiệm vụ là chọn action/tool và quyết định dừng; nó không viết final history answer.

## 🧾 Dữ liệu

Converter chấp nhận JSONL đã tải riêng từ xLAM, AgentInstruct hoặc Hotpot-style data:

```bash
python -m training.research_agent.prepare_dataset \
  --input datasets/raw/agent_rows.jsonl \
  --output datasets/research_agent/external_normalized.jsonl
```

Không commit toàn bộ dataset Hugging Face. History-specific trajectories:

```bash
python -m training.research_agent.build_history_trajectories \
  --input Dataset/merged_jsonl/all_messages.jsonl \
  --output datasets/research_agent/history_trajectories.jsonl
```

Trajectory classes gồm `local_only`, `local_then_web`, `multi_hop`, `conflicting_sources`, `insufficient`, `no_tool_needed`. State chỉ lưu action, observation summary và evidence IDs; không lưu hidden long-form chain-of-thought.

## 🚂 Train

```bash
python -m training.research_agent.train \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --max-length 4096 \
  --epochs 2 \
  --output-dir outputs/research_agent
```

QLoRA mặc định: NF4 4-bit, double quantization, LoRA r=32/alpha=64/dropout=0.05 trên attention và MLP projections.

## 📏 Evaluate

```bash
python -m training.research_agent.evaluate \
  --gold datasets/research_agent/gold.jsonl \
  --predictions predictions/research_agent.jsonl
```

Metric hiện có là exact match của tool sequence. Production acceptance nên bổ sung success rate theo từng trajectory class, budget violations và evidence recall trên bộ eval riêng.

## 📦 Output

Adapter deploy là `outputs/research_agent/`. Upload nó vào `/research_agent/adapter` trên Modal Volume. Base Qwen3 được cache riêng, không copy vào artifact Volume.
