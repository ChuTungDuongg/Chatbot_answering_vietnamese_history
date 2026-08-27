# Research / Tool Agent training

[Training overview](../README.md) · [Runtime tools](../../app/tools/README.md)

Research Agent dùng `Qwen/Qwen3-4B-Instruct-2507` để chọn tool/action và quyết định khi nào evidence đã đủ. Nó **không viết câu trả lời lịch sử cuối cùng**; việc đó thuộc History Answerer.

## Dataset contract

Mọi row production có `group_id`, `trajectory_id`, `step`, system/user/assistant messages, shared policy state và validated JSON decision. State chứa tool definition đầy đủ (`name`, `description`, `input_schema`) lấy từ cùng các class runtime. Initial state luôn có `observations=[]`, `evidence_ids=[]`.

Nguồn dữ liệu được tách stage rõ ràng:

- xLAM → `generic_tool_use`: function calling tổng quát; converter đọc đúng `query/tools/answers` JSON strings và giữ toàn bộ parallel calls.
- AgentInstruct → `multi_step_agent`: chỉ subset có environment mapping chắc chắn. Hiện hỗ trợ split `os` với action `bash`; split khác bị skip và report, không suy diễn tool call.
- VN History Phase 6 → `history_policy`: câu hỏi được tách khỏi `Tài liệu tham khảo`, recorded IDs chỉ xuất hiện sau observation thật, trajectory được unroll theo từng state → action.
- no-tool seed → greeting, capability, usage và thanks; false-premise history question không được xem là no-tool.

Khuyến nghị train hai stage thay vì trộn ngầm:

```text
Stage A (tuỳ chọn): xLAM + selected AgentInstruct generic/multi-step SFT
Stage B: grounded VN-history policy + real no-tool examples
```

Nếu tự build mixture, ghi ratio trong manifest và giữ `stage`/`source_dataset`; code không áp một ratio bí mật.

## Convert data

xLAM là gated dataset. Accept terms trên Hugging Face và chạy `hf auth login` (hoặc `huggingface-cli login`) trước:

```bash
python -m training.research_agent.prepare_dataset \
  --source xlam \
  --output datasets/research_agent/xlam.jsonl

python -m training.research_agent.prepare_dataset \
  --source agentinstruct \
  --split os \
  --output datasets/research_agent/agentinstruct_os.jsonl
```

Có thể dùng fixture/local export bằng `--input file.jsonl`; unit tests không download dataset.

Build grounded history trajectories:

```bash
python -m training.research_agent.build_history_trajectories \
  --input Dataset/merged_jsonl/all_messages.jsonl \
  --output datasets/research_agent/history_trajectories.jsonl
```

Builder không gọi web. Vì vậy nó không tạo fake web success/conflict; recorded Phase-6 references được dùng như kết quả local retrieval đã quan sát. `false_premise` vẫn đi qua `search_history` để kiểm chứng.

## Pre-training validation

```bash
python -m training.research_agent.validate_dataset \
  --dataset datasets/research_agent/history_trajectories.jsonl

python -m training.research_agent.preflight \
  --dataset datasets/research_agent/history_trajectories.jsonl

python -m training.research_agent.train \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --batch-size 1 \
  --max-samples 10 \
  --dry-run
```

Validator trả exit code 2 khi JSONL/schema/action/tool/argument/evidence causality/ID/trajectory ordering sai. Dry-run deterministic-shuffle theo group trước khi `--max-samples`, split toàn trajectory/question group cùng partition và ghi `split_manifest.json`. Nó không tải model.

## Precision trên Colab

`--bnb-compute-dtype auto` dùng cùng dtype với Trainer. Khi không truyền precision flag, hardware hỗ trợ BF16 sẽ chọn BF16; nếu không sẽ chọn FP16. Explicit mismatch bị từ chối.

T4 (FP16):

```bash
python -m training.research_agent.train \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --batch-size 1 --gradient-accumulation-steps 16 --max-length 2048 \
  --no-bf16 --fp16 --bnb-compute-dtype auto \
  --output-dir /content/drive/MyDrive/vn-history/research_agent
```

L4/A100 (BF16, sau khi preflight báo support):

```bash
python -m training.research_agent.train \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --batch-size 1 --gradient-accumulation-steps 16 --max-length 4096 \
  --bf16 --no-fp16 --bnb-compute-dtype auto \
  --output-dir /content/drive/MyDrive/vn-history/research_agent
```

QLoRA dùng NF4 4-bit, double quantization và LoRA attention/MLP. Research Agent dùng assistant-only CE utility trong `training/common/sft.py`, không phụ thuộc History Answerer loss.

## Evaluation

```bash
python -m training.research_agent.evaluate \
  --gold datasets/research_agent/gold.jsonl \
  --predictions predictions/research_agent.jsonl
```

Report gồm parse rate, tool selection, argument validity, unknown tool, finish/no-tool accuracy, unnecessary web search, budget violation, exact sequence và trajectory success theo class. `trajectory_success_rate` hiện là offline proxy; environment success thật cần integration evaluation riêng.
