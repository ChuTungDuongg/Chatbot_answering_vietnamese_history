# RAG-SFT message dataset

[Về README gốc](../README.md)

Thư mục `Dataset/` chứa output tạo dữ liệu RAG-SFT theo chủ đề. Đây là dữ liệu trung gian phục vụ kiểm tra và huấn luyện Phase 6, không phải corpus 58.603 chunks mà FastAPI dùng ở production.

Phần lớn file trong thư mục này khớp các pattern generated dataset của `.gitignore`; chỉ README được dùng làm bản đồ để developer hiểu dữ liệu local.

## Cấu trúc

```text
Dataset/
├── README.md
├── merged_jsonl/
│   └── all_messages.jsonl
└── Samples/
    ├── Pack1/
    ├── Pack2/
    └── ... Pack30/
```

Snapshot hiện tại có 30 packs, 121 files và `merged_jsonl/all_messages.jsonl` gồm 1.000 records.

## Các định dạng trong mỗi pack

| Pattern | Nội dung | Trường chính |
|---|---|---|
| `*_structured*.jsonl` | Sample có cấu trúc trước khi chuyển sang chat format | `id`, `type`, `question`, `context`, `gold_answer`, `gold_evidence` |
| `*_messages*.jsonl` | Sample theo Hugging Face chat messages | `id`, `type`, `messages` |
| `*_preview*.csv` hoặc `*_full_with_context.csv` | Bản dễ audit bằng bảng | question, evidence, answer và context tùy file |
| `*_stats.json` | Thống kê generation theo pack | source file, sample count, ratio, counts và outputs |

`all_messages.jsonl` là file hợp nhất dùng cho pipeline huấn luyện. Mỗi dòng là một JSON object độc lập:

```json
{"id":"sample_0001","type":"noisy_context","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Nhãn sample

Dataset hiện dùng bốn nhóm hành vi:

| `type` | Mục tiêu |
|---|---|
| `noisy_context` | Chọn đúng evidence khi context có nhiễu |
| `grounded_qa` | Trả lời trực tiếp và trích đúng source IDs |
| `insufficient_context` | Từ chối khi tài liệu không đủ bằng chứng |
| `false_premise` | Không chấp nhận tiền đề lịch sử sai |

## Quan hệ với các thư mục khác

- [`../Training/Dataset`](../Training/Dataset) giữ các chunk packs đầu vào có `chunk_id`.
- `Dataset/Samples` giữ output sample được sinh từ các packs.
- `Dataset/merged_jsonl/all_messages.jsonl` hợp nhất messages để dùng ở Phase 6.
- [`../artifacts`](../artifacts) là deployment bundle sau Phase 8-10, không dùng trực tiếp dữ liệu trong thư mục này khi API chạy.

## Kiểm tra nhanh

Chạy từ root bằng PowerShell:

```powershell
(Get-Content Dataset/merged_jsonl/all_messages.jsonl).Count
Get-Content Dataset/merged_jsonl/all_messages.jsonl | ForEach-Object { $_ | ConvertFrom-Json > $null }
```

Trước khi train, cần kiểm tra thêm unique sample ID, tỷ lệ bốn nhóm, `gold_evidence` tồn tại trong context, message roles hợp lệ và answer tuân thủ format source.

## Quy tắc khi cập nhật

- Giữ JSONL ở UTF-8 và đúng một JSON object trên mỗi dòng.
- Không chỉnh riêng `messages` mà không cập nhật structured source tương ứng.
- Regenerate merged file sau khi thêm/xóa pack; không nối thủ công nếu có nguy cơ trùng `id`.
- Không đưa dữ liệu nhạy cảm hoặc tài liệu không có quyền sử dụng vào sample/context.
