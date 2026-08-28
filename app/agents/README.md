# 🧠 Agent Runtime

[⬅️ Backend](../README.md) · [🧰 Tool contracts](../tools/README.md)

## 🎭 Ba vai trò

| Thành phần | Model/adapter | Trách nhiệm |
|---|---|---|
| `ResearchAgent` | Qwen3 + Research adapter | PLAN/ACTION/OBSERVATION/FINISH, chọn typed tool. |
| `EvidenceCriticAgent` | Qwen3 + Evidence adapter | Lọc, dedup, conflict check, compress, validate IDs. |
| `HistoryAnswererAgent` | Qwen3 + History adapter | Viết grounded Vietnamese answer và citations. |

`SharedAgentModelRuntime` nạp Qwen3 base một lần ở NF4 4-bit, load ba adapter tên `research`, `evidence`, `history`, rồi chuyển đúng adapter dưới lock trước generation. Metadata base mismatch hoặc role chưa load bị từ chối. `VLLMOpenAIBackend` giữ cùng interface và ba model name nhưng không tự quản lý server.

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
