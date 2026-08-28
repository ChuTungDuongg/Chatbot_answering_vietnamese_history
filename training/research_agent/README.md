# Research / Tool Agent training

[Training overview](../README.md) · [Runtime tools](../../app/tools/README.md)

Research Agent dùng `Qwen/Qwen3-4B-Instruct-2507` để chọn tool/action và quyết định khi nào evidence đã đủ. Nó **không viết câu trả lời lịch sử cuối cùng**; việc đó thuộc History Answerer.

## Dataset contract

Mọi row production có `group_id`, `trajectory_id`, `step`, system/user/assistant messages, shared policy state và validated JSON decision. State chứa tool definition đầy đủ (`name`, `description`, `input_schema`) lấy từ cùng các class runtime. Initial state luôn có `observations=[]`, `evidence_ids=[]`.

Nguồn dữ liệu được tách stage rõ ràng:

- xLAM → `generic_tool_use`: function calling tổng quát; converter đọc đúng `query/tools/answers` JSON strings và giữ toàn bộ parallel calls.
- AgentInstruct → `multi_step_agent`: chỉ subset có environment mapping chắc chắn. Hiện hỗ trợ split `os` với action `bash`; split khác bị skip và report, không suy diễn tool call.
- VN History Phase 6 → `history_policy`: câu hỏi được tách khỏi `Tài liệu tham khảo`, recorded IDs chỉ xuất hiện sau observation thật, trajectory được unroll theo từng state → action.
- no-tool V2.3 → 10 semantic families (greeting, thanks, farewell, capability, usage help, control, acknowledgement, reformat/repeat, UI help, near-empty) được group để paraphrase không cross split.
- policy boundary V2.3 → greeting/thanks/help prefix đi kèm câu hỏi lịch sử vẫn bắt buộc `search_history`; false-premise history question cũng không được xem là no-tool.

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

Builder không gọi web. Nó thêm 80 no-tool rows và 12 conversational-prefix hard negatives, nhưng giữ recorded Phase-6 state machine `search → inspect → finish`. Các source row có gold ID vắng khỏi input bị loại thay vì remap.

## Pre-training validation

```bash
python -m training.research_agent.validate_dataset \
  --dataset datasets/research_agent/history_trajectories.jsonl

python -m training.research_agent.preflight \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --tokenizer-id Qwen/Qwen3-4B-Instruct-2507

python -m training.research_agent.train \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --batch-size 1 \
  --max-samples 10 \
  --dry-run
```

Validator còn báo no-tool percentage/category, boundary hard negatives, per-split no-tool, semantic-family overlap và group leakage. Preflight/dry-run tải tokenizer nhưng không tải Qwen weights; cả hai fail nếu target rỗng hoặc zero-supervised.

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

Vì policy lịch sử hiện tại đã giữ tốt factual search, false-premise và `search → inspect → finish`, lựa chọn mặc định cho đợt sửa no-tool là corrective run từ Research adapter hiện có, trộn toàn bộ history trajectories với 80 no-tool + 12 boundary rows, LR khởi điểm `2e-5`, 1–2 epochs. Dùng `--init-adapter` để nạp PEFT weights với optimizer/scheduler mới; không dùng cùng `--resume-from-checkpoint`. Theo dõi riêng search recall và no-tool F1 để chống quên hành vi retrieval tốt.

## Evaluation

```bash
python -m training.research_agent.evaluate \
  --gold datasets/research_agent/gold.jsonl \
  --predictions predictions/research_agent.jsonl
```

Report gồm action accuracy, no-tool precision/recall/F1, history-search recall, conversational-prefix accuracy, meta no-tool, false-premise initial search, search→inspect, inspect→finish, schema validity và exact trajectory proxy.

### Canonical sanity replay

Không dựng synthetic T2 state bằng tay. Tạo sanity gold trực tiếp từ canonical rows để system message, serialized `ResearchPolicyState`, tool schemas, `search_history` observation, `evidence_ids` và gold action giữ nguyên distribution train/runtime:

```bash
python -m training.research_agent.sanity \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --output-gold artifacts/reports/research_sanity_gold.jsonl
```

Default suite giữ năm category riêng: 5 `no_tool`, 5 factual-history search, 5 conversational-prefix history search, 20 real `search→inspect` step-2 rows và 20 real `inspect→finish` rows. Step-2 sampler ưu tiên group khác nhau và từ chối synthetic/non-canonical payload.

Trong notebook, dùng nguyên input messages thay vì tự rút gọn state:

```python
from training.common.jsonl import read_jsonl
from training.research_agent.sanity import (
    build_sanity_suite,
    evaluate_sanity_predictions,
    inference_messages,
)

gold = build_sanity_suite(
    read_jsonl("datasets/research_agent/history_trajectories.jsonl")
)
predictions = []
for row in gold:
    decision = generate_research_json(inference_messages(row))
    predictions.append({"prediction": decision})

report = evaluate_sanity_predictions(predictions, gold)
```

Hoặc ghi predictions theo đúng thứ tự suite rồi chạy:

```bash
python -m training.research_agent.sanity \
  --dataset datasets/research_agent/history_trajectories.jsonl \
  --predictions predictions/research_sanity.jsonl
```

Step-2 report tách riêng:

- `step2_action_tool_transition_accuracy`: model có chuyển đúng sang một `inspect_evidence` action hay không;
- `step2_evidence_id_exact_match`: exact set match, không phụ thuộc thứ tự ID;
- `step2_evidence_id_precision` và `step2_evidence_id_recall`: micro ID metrics;
- `step2_evidence_id_scored_rows`: chỉ các rows đã chọn đúng transition mới được chấm ID.

Vì vậy search lại ở T2 chỉ là transition failure. Chọn đúng `inspect_evidence` nhưng sai/thừa/thiếu IDs vẫn pass transition và chỉ làm giảm ID metrics.
