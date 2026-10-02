# Loại bỏ response modes khỏi live application

App giữ đúng hai inference modes: **Hybrid RAG** và **Central Agent**. Composer không còn slider độ chi tiết. Live API/OpenAPI/SSE không còn field `response_mode`; frontend không gửi field này và không đọc/ghi localStorage `vn-history-response-mode`.

Hybrid và final generation của Central gọi `build_messages(question, contexts, history)`, sử dụng đúng `SYSTEM_PROMPT`. Chunk IDs, title, text, history, attachment evidence và citation format `[chunk_id]` vẫn được truyền như trước. Central planning, tool observations, action rounds, registry và fallback `search_history` được giữ nguyên. Không thay đổi model IDs, generation settings hoặc retrieval FAISS/BM25/Qdrant.

## Files modified

| File | Thay đổi |
|---|---|
| `frontend/src/App.jsx` | Bỏ hook/state/props response detail. |
| `frontend/src/components/ChatInput.jsx` | Bỏ slider; giữ textarea, Hybrid/Central selector, upload, stop/send và paste/drop. |
| `frontend/src/hooks/useChatStream.js` | Bỏ parameter và option `responseMode`. |
| `frontend/src/services/api.js` | Bỏ parameter và JSON `response_mode`. |
| `frontend/src/styles/composer.css` | Chỉ bỏ CSS của slider. |
| `app/schemas.py` | Bỏ response detail khỏi ChatRequest/ChatResponse/OpenAPI. |
| `app/api/routes.py` | Bỏ field khỏi status/done/error metadata, runtime.prepare và /chat response. |
| `app/rag/hybrid_runtime.py` | Bỏ parameter; gọi grounded prompt mặc định. |
| `app/central/runtime.py` | Bỏ parameter; final generation gọi grounded prompt mặc định. |
| `app/rag/prompting.py` | Chỉ cập nhật docstrings về live/offline compatibility. |
| `app/rag/response_modes.py` | Chỉ cập nhật docstring về frozen research compatibility. |
| `modal_app.py` | Smoke request không còn gửi response detail. |
| `frontend/tests/ui/App.test.jsx` | Regression: không có slider, không dùng legacy storage key; composer vẫn hiện đủ controls. |
| `frontend/tests/ui/useChatStream.test.jsx` | Kiểm tra Hybrid/Central và request fields còn dùng; không gửi responseMode. |
| `tests/test_baseline_runtime.py` | Regression cho API/OpenAPI/SSE, fixed prompt, chunk evidence, history, attachments, Central tools/fallback. |
| `tests/test_training_six_way_drive.py` | Bỏ assertions về live API detail field; giữ tests cho explicit offline modes. |
| `docs/TRAINING_AND_EVALUATION.md` | Phân biệt live fixed prompt và historical response modes. |
| `MODAL_QUICKSTART.md` | Bỏ obsolete response field khỏi terminal example. |
| `RESTORE_MODAL_QDRANT_REPORT.md` | Bỏ obsolete response field khỏi terminal example. |

Added: `frontend/tests/ui/api.test.jsx` kiểm tra JSON streaming thật của service cho cả Hybrid/Central; report này ghi nhận kết quả và phạm vi.

## Files deleted

- `frontend/src/components/ResponseDetailSlider.jsx`
- `frontend/src/config/responseModes.js`
- `frontend/src/hooks/useResponseMode.js`
- `frontend/tests/ui/responseDetailSlider.test.jsx`

## Historical compatibility và kiểm tra cuối bằng rg

Search đúng các token `response_mode`, `responseMode`, `ResponseMode`, `ResponseDetailSlider`, `useResponseMode`, `MODE_INSTRUCTIONS`, `mode_instruction`, `vn-history-response-mode`:

- **Live:** không còn occurrence trong `frontend/src`, schemas, API routes, Hybrid/Central runtimes hoặc Modal entrypoint. Dùng `\bresponse_mode\b` để không nhầm với FastAPI `response_model`.
- **Compatibility:** `app/rag/response_modes.py` và optional explicit arguments trong `app/rag/prompting.py` vẫn phục vụ frozen SFT V1/Citation V2/six-way. Default live helper không append instruction. AST của hai module, khi bỏ docstrings, giống hệt HEAD trước thay đổi.
- **Historical:** giữ các occurrences trong `evaluation/silver_v1.py`, `evaluation/silver_citation_v2.py`, `evaluation/six_way.py`, `Training/train_qwen3.py`, dataset/benchmark files và research docs.
- **Tests:** negative assertions kiểm tra field/storage key không được dùng; offline tests vẫn kiểm tra cả ba modes.

Không có thay đổi trong 275 files được snapshot thuộc evaluation, Training, artifacts, portable_backup và frozen V1/V2 tests: 262 SHA-256 khớp; 13 files lớn giữ nguyên size/mtime. Không regenerate dataset, sửa SQLite nghiên cứu, rebuild index, re-embed, train hoặc sửa adapter/ZIP.

## Validation

| Kiểm tra | Kết quả |
|---|---|
| `frontend: npm test` | PASS: 66 Node tests + 53 Vitest UI tests = 119. |
| `frontend: npm run build` | PASS. |
| Targeted pytest (bốn files yêu cầu + training/six-way compatibility) | PASS: 60 tests. |
| Toàn bộ `pytest -q` | PASS: 243 tests. |
| `git diff --check` | PASS. |
| Protected files / compatibility prompt logic | PASS. |

Lệnh đã chạy trên Windows PowerShell, từ repo root:

```powershell
Push-Location frontend
npm test
npm run build
Pop-Location
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONCASEOK = '1'
.\.conda\python.exe -m pytest -q tests/test_baseline_runtime.py tests/test_v1_runtime_evaluation.py tests/test_silver_v1_freeze_sft.py tests/test_silver_citation_v2.py tests/test_training_six_way_drive.py
.\.conda\python.exe -m pytest -q
```

`PYTHONCASEOK=1` cần cho offline imports trên máy Windows này: thư mục repository là `Training`, các historical imports dùng `training`. Đây là điều chỉnh môi trường kiểm thử; không sửa code frozen. Pytest có một deprecation warning từ Starlette/httpx; không có test failure.

## LLM smoke trên Modal

Run [ap-HV3BKlwSIAIhG52VyYFRN7](https://modal.com/apps/tungduong156gli/main/ap-HV3BKlwSIAIhG52VyYFRN7) đã hoàn tất trên A100 với V1 `/artifacts/corpus_v1/chunks.jsonl`, FAISS/BM25/reranker và hai base models hiện tại. Readiness xác nhận 624,288 chunks/vectors/BM25 documents. Hybrid và Central đều trả `status=done`, mỗi lượt 3 sources, JSON không có `response_mode`. Smoke dùng SQLite tạm; không đổi conversations của người dùng. Evidence lưu ở `reports/response_modes/runtime_smoke.json` (gitignored).

Hybrid nhận diện được 3/3 cited sources. **WARNING:** ở câu hỏi về ý nghĩa chiến thắng Bạch Đằng năm 938, Central sinh `[chunk_49a08d9c04e0be77e9bf6ff9]` trong khi source ID là `chk_49a08d9c04e0be77e9bf6ff9`; vì vậy 0/3 sources được đánh dấu cited. Không đổi parser hoặc sửa câu trả lời bằng heuristic để che vấn đề này. Fixed grounded prompt và citation parser được giữ nguyên; việc model tuân thủ citation chưa được bảo đảm cho mọi câu hỏi.

Kiểm tra bổ sung bằng câu hỏi “Ngô Quyền giành chiến thắng trên sông Bạch Đằng vào năm nào?” trong run [ap-6BxP5WxGWiftphrwngbDgF](https://modal.com/apps/tungduong156gli/main/ap-6BxP5WxGWiftphrwngbDgF): **PASS**. Cả hai modes trả lời năm 938, `status=done`, mỗi lượt 3 sources và không có response detail field. Hybrid có 2 cited sources; Central có 1 cited source hợp lệ (`[chk_00c764d281ba9961f5d90673]`). Evidence: `reports/response_modes/runtime_smoke_direct.json` (gitignored). Hai runs xác nhận real retrieval → chunk evidence → LLM → response hoạt động; không phải benchmark chất lượng toàn bộ.

Các thay đổi đang ở working tree; **không commit hoặc push**. Bản mới được kiểm tra bằng `modal run`; production deployment hiện có chưa được redeploy trong task này. Để áp dụng backend mới sau khi review, dùng môi trường/lệnh `modal deploy modal_app.py` trong `MODAL_QUICKSTART.md`; frontend mới đã build thành công.
