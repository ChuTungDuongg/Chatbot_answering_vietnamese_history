# 🧠 Agent Runtime

[⬅️ Backend](../README.md) · [🧰 Tool contracts](../tools/README.md)

## 🎭 Ba vai trò

| Thành phần | Model/adapter | Trách nhiệm |
|---|---|---|
| `ResearchAgent` | Qwen3 + Research adapter | PLAN/ACTION/OBSERVATION/FINISH, chọn typed tool. |
| `EvidenceCriticAgent` | Qwen3 + Evidence adapter | Lọc, dedup, conflict check, compress, validate IDs. |
| `HistoryAnswererAgent` | merged Qwen2.5 History model | Viết grounded Vietnamese answer và citations. |

`SharedAgentModelRuntime` nạp Qwen3 base một lần ở NF4 4-bit, load hai adapter tên `research` và `evidence`, rồi chuyển adapter dưới lock trước generation. Điều này tránh duplicate base weights và ngăn hai request đổi adapter đồng thời.

## 🔁 Orchestration

```text
Research run (max 6 steps)
  → prefetch evidence từ PDF/ảnh của đúng conversation nếu có
  → Evidence critique
  → nếu insufficient và model controller đang bật: thêm tối đa 1 research round
  → Evidence critique lại
  → History answer + guards
  → cleanup SessionEvidenceStore
```

Research policy chỉ trả JSON action hoặc finish. Evidence policy chỉ trả structured JSON. Parser có tối đa một repair cho JSON không hợp lệ; Pydantic và candidate-ID validation chạy trước khi evidence tới History model. Attachment được prefetch bằng request scope nội bộ và cũng xuất hiện dưới dạng tool `search_uploaded_documents`; model không được nhìn thấy hoặc tự chọn owner/conversation ID.

## 🛟 Fallback

Khi `AGENT_CONTROLLER=deterministic`, Research chạy local search rồi web fallback nếu local rỗng; Evidence chọn các chunk không rỗng đầu tiên. Chế độ này hữu ích cho smoke test, không đại diện chất lượng của hai agent đã train.

## 🧪 Unit testing

Tests dùng fake generator/tool và deterministic path. Không khởi tạo `SharedAgentModelRuntime`, do đó không tải Qwen3.
