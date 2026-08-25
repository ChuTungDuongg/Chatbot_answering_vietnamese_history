# 📚 RAG-SFT Message Dataset

[Về README gốc](../README.md)

`Dataset/` chứa các sample RAG-grounded SFT được sinh theo topic pack. Đây là dữ liệu huấn
luyện/trung gian cho Phase 6, không phải corpus 58.603 chunks mà FastAPI dùng khi retrieval.

Các output lớn/generated khớp rule trong `.gitignore`; README này ghi lại layout và snapshot
local để developer không nhầm vai trò dữ liệu.

## Snapshot hiện tại

| Thành phần | Số lượng |
|---|---:|
| Topic packs | 30 |
| Generated files trong `Samples/` | 120 |
| Records trong `merged_jsonl/all_messages.jsonl` | 1.000 |
| `noisy_context` | 650 |
| `grounded_qa` | 200 |
| `insufficient_context` | 100 |
| `false_premise` | 50 |

## Cấu trúc

```text
Dataset/
├── README.md
├── Samples/
│   ├── Pack1/
│   ├── Pack2/
│   └── ... Pack30/
└── merged_jsonl/
    └── all_messages.jsonl
```

Mỗi pack thường có bốn loại output:

| Pattern | Nội dung | Trường chính |
|---|---|---|
| `*_structured*.jsonl` | Sample trước khi chuyển sang chat format | `id`, `type`, `question`, `context`, `gold_answer`, `gold_evidence` |
| `*_messages*.jsonl` | Hugging Face chat messages | `id`, `type`, `messages` |
| `*_preview*.csv` / `*_full_with_context.csv` | Bản bảng để audit | Question, evidence, answer, context |
| `*_stats.json` | Thống kê generation của pack | Source, count, ratio, outputs |

Một dòng trong merged messages:

```json
{
  "id": "sample_0001",
  "type": "noisy_context",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

JSONL thật lưu mỗi object trên đúng một dòng; ví dụ được format lại chỉ để dễ đọc.

## Ý nghĩa bốn loại sample

| `type` | Hành vi cần học |
|---|---|
| `noisy_context` | Chọn đúng evidence khi context chứa nhiễu |
| `grounded_qa` | Trả lời dựa trên context và nêu đúng source IDs |
| `insufficient_context` | Từ chối khi tài liệu không đủ bằng chứng |
| `false_premise` | Không tiếp nhận tiền đề lịch sử sai |

## Lưu ý về `id`

`all_messages.jsonl` hiện có 1.000 records nhưng chỉ 40 giá trị `id` khác nhau. ID được đánh
lại trong từng pack, nên `id` không phải khóa duy nhất toàn cục.

Không deduplicate merged dataset chỉ theo `id`. Khi cần định danh ổn định:

- ở dữ liệu pack, dùng `pack/file path + id`;
- ở merged file hiện tại, dùng row index hoặc tạo `sample_uid` mới trong bước regenerate;
- nếu thêm provenance vào schema, regenerate cả structured và messages outputs cùng lúc.

Đây là đặc điểm dữ liệu hiện tại, không phải 960 records bị trùng nội dung theo kết luận tự động.
Cần so sánh question/context/answer nếu muốn phát hiện duplicate thực sự.

## Quan hệ với pipeline

```text
training/Dataset/Chunk_id
  -> Dataset/Samples/<Pack>
  -> Dataset/merged_jsonl/all_messages.jsonl
  -> Phase 6 RAG-SFT
  -> Phase 8-10 retrieval artifacts và merged model
```

- [`../training/Dataset`](../training/README.md) giữ 520 context records có `chunk_id`.
- `Samples/` giữ sample sinh từ các pack.
- `merged_jsonl/all_messages.jsonl` là đầu vào chat-format cho training.
- [`../artifacts/`](../artifacts/README.md) là output deployment, không đọc dataset này khi API
  phục vụ request.

## Kiểm tra nhanh

PowerShell, chạy từ repository root:

```powershell
$rows = Get-Content Dataset/merged_jsonl/all_messages.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
$rows.Count
$rows | Group-Object type | Select-Object Name, Count
($rows.id | Sort-Object -Unique).Count
```

Checklist trước khi train:

- JSONL là UTF-8 và mỗi dòng parse thành một object;
- role theo đúng thứ tự user/assistant;
- tỷ lệ bốn loại sample đúng với run mong muốn;
- `gold_evidence` tồn tại trong context;
- answer giữ output/source contract;
- kiểm tra duplicate bằng content hoặc composite key, không dùng riêng `id`;
- kiểm tra leakage giữa train/validation/test theo question và evidence.

## Quy tắc cập nhật

- Không chỉnh riêng merged messages mà bỏ qua structured source tương ứng.
- Regenerate merged file sau khi thêm/xóa pack.
- Ghi lại generator config, seed và source pack cho lần sinh mới.
- Không commit token, dữ liệu nhạy cảm hoặc tài liệu không có quyền sử dụng.
