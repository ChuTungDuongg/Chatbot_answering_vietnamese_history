# Evidence Critic / Compressor v2

[Training overview](../README.md) · [Runtime agents](../../app/agents/README.md)

Evidence Agent chỉ lọc, kiểm tra, khử trùng lặp và nén evidence được cung cấp. Training và runtime dùng cùng `EVIDENCE_AGENT_SYSTEM` và cùng canonical model-output schema.

## Canonical contract

Model chỉ sinh các field semantic:

```json
{
  "status": "sufficient",
  "selected_evidence": [
    {
      "evidence_id": "ev_01",
      "relevance": 1.0,
      "claims": ["Fact trích trực tiếp từ evidence nguồn."],
      "compressed_text": "Fact trích trực tiếp từ evidence nguồn."
    }
  ],
  "conflicts": [],
  "missing_information": [],
  "summary": "Kết luận cụ thể dựa trên evidence."
}
```

Runtime tự derive `selected_ids`, `rejected_ids`, `compressed_context` và `sufficient`. Các field này không phải training target. Production reject output legacy dạng `selected_evidence: list[str]`; debug fallback chỉ hoạt động khi được bật rõ ràng.

## Build và validate dataset v2

```bash
python -m training.evidence_agent.prepare_dataset \
  --input Dataset/merged_jsonl/all_messages.jsonl \
  --output datasets/evidence_agent/train.jsonl

python -m training.evidence_agent.validate_dataset \
  --dataset datasets/evidence_agent/train.jsonl
```

Builder giữ clean sufficient examples và tạo các behavior có tỷ lệ cấu hình được: duplicate, controlled numeric conflict, partial evidence, relevant+distractor và insufficient. Conflict giả lập chỉ tồn tại trong training row với metadata `synthetic_conflict`; nó không được ghi vào history corpus. Claims/compression là extractive để validator có thể kiểm tra grounding.

Partial augmentation không còn được gán chỉ bằng regex/quota. Builder trích các answer component có độ tin cậy cao từ cấu trúc câu hỏi và gold answer, thử loại supporting chunk/claim, rồi chỉ giữ target partial khi input còn hỗ trợ ít nhất một component nhưng thực sự mất component khác. Nếu evidence còn cover đủ, row trở lại sufficient; nếu không chứng minh được partial một cách bảo thủ, augmentation bị bỏ. `missing_information` được sinh theo slot cụ thể như thời gian, người chỉ đạo, lực lượng đối phương, hiệu hoặc tự.

`relevance` được tính từ độ phủ token nội dung giữa question và từng evidence item, độc lập với status của toàn evidence pool. Metadata `semantic_coverage` và `coverage_audit` không nằm trong messages dùng để train; chúng phục vụ audit/validator.

Source row legacy cite evidence ID không có trong chính input context sẽ bị loại thay vì tự remap hoặc bịa evidence. CLI báo rõ `dropped_source_rows`.

Validator kiểm tra JSON/schema, ID/group uniqueness, selected ID tồn tại, grounded claims, non-empty compression, generic summary/missing-information, answer-component coverage, relevance shortcut, semantics duplicate/conflict/partial, class distribution và group overlap của split mô phỏng. Critical error trả exit code khác 0; các coverage case chỉ có heuristic confidence được báo warning riêng thay vì tự động relabel.

## Corrective fine-tune từ adapter hiện có

```bash
python -m training.evidence_agent.train \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --dataset datasets/evidence_agent/train.jsonl \
  --init-adapter /path/to/evidence-agent-full \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 4096 \
  --bf16 \
  --no-fp16 \
  --output-dir outputs/evidence-agent-v2
```

`--init-adapter` load learned PEFT weights ở chế độ trainable nhưng tạo optimizer/scheduler mới cho run v2. `--resume-from-checkpoint` phục hồi trainer/optimizer/scheduler của cùng một run. Hai mode không được dùng cùng nhau.

Split luôn group theo `group_id`, vì vậy các row cùng `original_sample_id` không thể rơi vào train/eval/test khác nhau. Training dùng assistant-only SFT trên đầy đủ system/user/assistant messages.

## Evaluate

```bash
python -m training.evidence_agent.evaluate \
  --gold datasets/evidence_agent/gold.jsonl \
  --predictions predictions/evidence_agent.jsonl
```

Evaluator báo JSON parse rate, runtime schema validity, status accuracy, selected-evidence precision/recall/F1, invented-ID rate, duplicate reduction, conflict detection, missing-information proxy, non-empty compression và compression ratio. Semantic grounding được ghi rõ là chưa có metric tự động đáng tin cậy, không giả lập một điểm số.

## Output

Upload output adapter mới vào vị trí Evidence adapter của runtime. Research và Evidence adapters phải dùng cùng base model ID nếu chạy trong shared runtime.
