# Evidence Critic / Compressor v2.2

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

python -m training.evidence_agent.preflight \
  --dataset datasets/evidence_agent/train.jsonl \
  --tokenizer-id Qwen/Qwen3-4B-Instruct-2507 \
  --max-length 4096
```

Builder giữ clean sufficient examples và tạo các behavior có tỷ lệ cấu hình được: duplicate, question-relevant conflict, irrelevant-disagreement hard negative, partial evidence, relevant+distractor và insufficient. Augmentation chỉ tồn tại trong training row với provenance ở top-level metadata; nó không được ghi vào history corpus. Claims/compression là extractive để validator có thể kiểm tra grounding.

V2.2 chỉ tạo conflict khi mutation đổi một giá trị thuộc answer slot mà question thực sự yêu cầu. Date/year, person, location, count và role/title mutation đều được rút deterministically từ question + gold answer + supplied evidence; nếu không chứng minh được same-slot incompatibility thì augmentation conflict bị bỏ. Các disagreement ngoài answer slot được dùng làm hard negative `irrelevant_disagreement` với target non-conflicting.

Synthetic provenance chỉ nằm trong top-level `metadata`. Model-visible `EvidenceAgentRequest` dùng opaque stable IDs và production-like `source_type`; các marker `__conflict`, `__dup_*`, `synthetic_conflict` và `synthetic_duplicate_*` bị validator cấm. Duplicate target luôn ưu tiên source gốc và không gán claim của bản paraphrase sang evidence ID khác.

Partial augmentation không còn được gán chỉ bằng regex/quota. Builder trích các answer component có độ tin cậy cao từ cấu trúc câu hỏi và gold answer, thử loại supporting chunk/claim, rồi chỉ giữ target partial khi input còn hỗ trợ ít nhất một component nhưng thực sự mất component khác. Nếu evidence còn cover đủ, row trở lại sufficient; nếu không chứng minh được partial một cách bảo thủ, augmentation bị bỏ. `missing_information` được sinh theo slot cụ thể như thời gian, người chỉ đạo, lực lượng đối phương, hiệu hoặc tự.

`relevance` được tính từ độ phủ token nội dung giữa question và từng evidence item, độc lập với status của toàn evidence pool. Metadata `semantic_coverage` và `coverage_audit` không nằm trong messages dùng để train; chúng phục vụ audit/validator.

Source row legacy cite evidence ID không có trong chính input context sẽ bị loại thay vì tự remap hoặc bịa evidence. CLI báo rõ `dropped_source_rows`.

Validator kiểm tra JSON/schema, ID/group uniqueness, selected ID tồn tại, grounded claims, non-empty compression, generic summary/missing-information, answer-component coverage, relevance shortcut, semantics duplicate/conflict/partial, class distribution và group overlap của split mô phỏng. Critical error trả exit code khác 0; các coverage case chỉ có heuristic confidence được báo warning riêng thay vì tự động relabel.

Grounding được kiểm tra theo từng evidence ID bằng normalized extractive containment. `compressed_text` phải là đoạn của chính source hoặc được ghép từ các claim đã grounded của chính source đó. Runtime áp dụng cùng invariant và reject output cross-ID attribution; conflict output còn phải nhắc ít nhất hai supplied evidence IDs.

Tokenization preflight chỉ tải tokenizer, không tải Qwen weights. Nó chạy đúng đường assistant-only tokenization của Trainer cho cả train/eval/test và fail nếu input/target rỗng, assistant target bị mất, hoặc sample có 0 supervised token. Báo cáo gồm độ dài sequence, số row overlength/capped, và min/mean/max của prompt, assistant, supervised token. Với Evidence JSON quá dài, pipeline giữ nguyên question, evidence ID, title/source metadata và mọi grounded target claim; nó cắt evidence text theo cấu trúc trước khi render JSON. Assistant target được giữ nguyên toàn bộ, prompt không bị token-slice thành JSON lỗi.

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

Split luôn group theo `group_id`, vì vậy các row cùng `original_sample_id` không thể rơi vào train/eval/test khác nhau. Trong ràng buộc đó, allocator tối đa hóa độ phủ `behavior` và độ gần phân phối toàn cục cho eval/test. Training dùng assistant-only SFT trên đầy đủ system/user/assistant messages; `--dry-run` cũng chạy tokenization preflight nhưng dừng trước khi tải model weights.

## Evaluate

```bash
python -m training.evidence_agent.evaluate \
  --gold datasets/evidence_agent/gold.jsonl \
  --predictions predictions/evidence_agent.jsonl
```

Evaluator báo JSON parse rate, runtime schema validity, status accuracy, selected-evidence precision/recall/F1, invented-ID rate, duplicate reduction, conflict detection, missing-information proxy, non-empty compression và compression ratio. Semantic grounding được ghi rõ là chưa có metric tự động đáng tin cậy, không giả lập một điểm số.

## Output

Upload output adapter mới vào vị trí Evidence adapter của runtime. Research và Evidence adapters phải dùng cùng base model ID nếu chạy trong shared runtime.
