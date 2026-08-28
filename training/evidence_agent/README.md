# Evidence Critic / Compressor v2.3

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

## Build và validate dataset v2.3

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

Builder audit toàn bộ gold sufficient theo answer slot, tính coverage của từng item và hợp coverage của tập chọn, rồi rút về một minimal sufficient subset. Gold `sufficient` có selected subset thiếu slot, redundant hoặc irrelevant bị loại/sửa trước khi train. Các true multi-source fixture giữ full row và sinh cả hai leave-one-out partial; partial luôn giữ evidence hữu ích và nêu slot thiếu cụ thể.

V2.2 chỉ tạo conflict khi mutation đổi một giá trị thuộc answer slot mà question thực sự yêu cầu. Date/year, person, location, count và role/title mutation đều được rút deterministically từ question + gold answer + supplied evidence; nếu không chứng minh được same-slot incompatibility thì augmentation conflict bị bỏ. Các disagreement ngoài answer slot được dùng làm hard negative `irrelevant_disagreement` với target non-conflicting.

Synthetic provenance chỉ nằm trong top-level `metadata`. Model-visible `EvidenceAgentRequest` dùng opaque stable IDs và production-like `source_type`; các marker `__conflict`, `__dup_*`, `synthetic_conflict` và `synthetic_duplicate_*` bị validator cấm. Duplicate target luôn ưu tiên source gốc và không gán claim của bản paraphrase sang evidence ID khác.

Partial augmentation không còn được gán chỉ bằng regex/quota. Builder hỗ trợ time/topic, leader/opponent, one-source multi-slot và explicit/partial relation fixtures; một câu hỏi nhiều slot không bị ép phải chọn nhiều source. Nếu không chứng minh được `supported_slots >= 1` và `missing_slots >= 1` một cách bảo thủ, candidate bị bỏ.

`relevance` được tính từ độ phủ token nội dung giữa question và từng evidence item, độc lập với status của toàn evidence pool. Metadata `semantic_coverage` và `coverage_audit` không nằm trong messages dùng để train; chúng phục vụ audit/validator.

Source row legacy cite evidence ID không có trong chính input context sẽ bị loại thay vì tự remap hoặc bịa evidence. CLI báo rõ `excluded_source_rows`.

Validator kiểm tra JSON/schema, ID/group uniqueness, grounding, minimal sufficient coverage, useful partial retention, slot-specific missing information, duplicate/conflict semantics và group overlap. `sufficient_selected_subset_incomplete`, `insufficient_partial_selected_empty` và `insufficient_no_support_selected_nonempty` đều là hard failures.

Grounding được kiểm tra theo từng evidence ID bằng normalized extractive containment. `compressed_text` phải là đoạn của chính source hoặc được ghép từ các claim đã grounded của chính source đó. Runtime áp dụng cùng invariant và reject output cross-ID attribution; conflict output còn phải nhắc ít nhất hai supplied evidence IDs.

Tokenization preflight chỉ tải tokenizer, không tải Qwen weights. Nó chạy đúng đường assistant-only tokenization của Trainer cho cả train/eval/test và fail nếu input/target rỗng, assistant target bị mất, hoặc sample có 0 supervised token. Báo cáo gồm độ dài sequence, số row overlength/capped, và min/mean/max của prompt, assistant, supervised token. Với Evidence JSON quá dài, pipeline giữ nguyên question, evidence ID, title/source metadata và mọi grounded target claim; nó cắt evidence text theo cấu trúc trước khi render JSON. Assistant target được giữ nguyên toàn bộ, prompt không bị token-slice thành JSON lỗi.

## Fresh V2.3 QLoRA

```bash
python -m training.evidence_agent.train \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --dataset datasets/evidence_agent/train.jsonl \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 4096 \
  --bf16 \
  --no-fp16 \
  --output-dir outputs/evidence-agent-v23
```

V2.3 thay đổi semantics của sufficient selection và partial retention, vì vậy khuyến nghị fresh QLoRA từ vanilla Qwen3. `--init-adapter` vẫn tồn tại cho thí nghiệm corrective có chủ đích nhưng không phải lựa chọn mặc định của lần retrain này.

Split luôn group theo `group_id`, vì vậy các row cùng `original_sample_id` không thể rơi vào train/eval/test khác nhau. Trong ràng buộc đó, allocator tối đa hóa độ phủ `behavior` và độ gần phân phối toàn cục cho eval/test. Training dùng assistant-only SFT trên đầy đủ system/user/assistant messages; `--dry-run` cũng chạy tokenization preflight nhưng dừng trước khi tải model weights.

## Evaluate

```bash
python -m training.evidence_agent.evaluate \
  --gold datasets/evidence_agent/gold.jsonl \
  --predictions predictions/evidence_agent.jsonl
```

Evaluator báo status/ID metrics cùng answer-slot coverage, selected-subset sufficiency, partial retention, missing-slot accuracy, true-multi recall, minimal-subset rate, redundant selection và conflict/grounding metrics. Heuristic semantic metrics vẫn cần post-training human sanity.

## Output

Upload output vào `adapters/evidence`. Cả ba active adapters phải khai báo cùng shared Qwen3 base ID.
