# Backend application

[Về README gốc](../README.md)

Thư mục `app/` chứa toàn bộ FastAPI runtime: cấu hình, schema API, conversation memory, xử lý tài liệu, Hybrid RAG, generation và guardrails. Không đặt notebook huấn luyện hoặc artifact model vào đây.

## Điểm bắt đầu

`main.py` tạo FastAPI app và quản lý lifecycle theo thứ tự:

```text
Settings
  -> ConversationStore.initialize()
  -> RAGService.load()
  -> HybridRetriever + attachment services (retrieval-only/full)
  -> RAGGenerator (full)
  -> gắn dependencies vào app.state
  -> mount conversation và RAG routers
```

Chạy toàn bộ dự án từ root bằng `npm run dev`. Khi cần kiểm tra API local độc lập:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mode mặc định khi chạy trực tiếp là `api-only`. Đặt `APP_MODE=retrieval-only` hoặc `APP_MODE=full` và chuẩn bị artifact tương ứng nếu cần retrieval/generation.

## Bản đồ module

| File | Trách nhiệm | Bắt đầu ở đây khi |
|---|---|---|
| [`main.py`](main.py) | Lifespan, CORS, health/readiness và router wiring | Thêm dependency cấp ứng dụng hoặc endpoint hệ thống |
| [`config.py`](config.py) | Đọc `.env`, runtime mode và đường dẫn artifact/SQLite | Thêm cấu hình hoặc thay layout artifact |
| [`schemas.py`](schemas.py) | Pydantic request/response models | Thay API contract |
| [`api/routes.py`](api/routes.py) | Retrieve, chat và validated SSE | Thay request flow hoặc SSE events |
| [`api/conversations.py`](api/conversations.py) | CRUD conversation và upload/delete attachment | Thay ownership hoặc document endpoints |
| [`chat/store.py`](chat/store.py) | SQLite conversations, messages, attachments và temporary chunks | Thay persistence/memory |
| [`chat/attachments.py`](chat/attachments.py) | PDF extraction, OCR, chunking, embedding và temporary retrieval | Thay document ingestion |
| [`services/rag_service.py`](services/rag_service.py) | Validate/load corpus, FAISS, BM25S, embedder, reranker và model | Thay startup hoặc artifact loading |
| [`rag/retrieval.py`](rag/retrieval.py) | Query analysis, E5/FAISS, BM25S, RRF, rerank và diversity | Thay retrieval quality |
| [`rag/prompting.py`](rag/prompting.py) | History formatting, prompt budget, output parsing và repair prompt | Thay prompt/output format |
| [`rag/generation.py`](rag/generation.py) | Contextual retrieval, global/temp merge, generation và repair orchestration | Thay end-to-end answer behavior |
| [`rag/guards.py`](rag/guards.py) | Source/year/format/completeness validation | Thay hallucination safeguards |

## Runtime modes

| `APP_MODE` | SQLite/CRUD | Global retrieval | Upload + temp corpus | Generation |
|---|---:|---:|---:|---:|
| `api-only` | Có | Không | Không | Không |
| `retrieval-only` | Có | Có | Có | Không |
| `full` | Có | Có | Có | Có |

`RAGService` chỉ load những thành phần cần cho mode hiện tại. `/ready` là nguồn chính xác để biết runtime nào đã sẵn sàng.

## Luồng dữ liệu chính

### Conversation và memory

```text
X-Client-ID + conversation_id
  -> kiểm tra ownership
  -> đọc messages gần nhất từ SQLite
  -> history phục vụ contextual retrieval và prompt
  -> lưu user/assistant messages cùng cited sources
```

History chỉ là ngữ cảnh hội thoại, không được xem là evidence. Các khẳng định lịch sử vẫn phải dựa trên global hoặc temporary chunks đã retrieval.

### Attachment

```text
UploadFile
  -> kiểm tra MIME/dung lượng
  -> PyMuPDF đọc text PDF
  -> OCR trang scan hoặc ảnh bằng Tesseract vie+eng
  -> chunk 220 từ, overlap 40 từ
  -> multilingual E5 embedding
  -> lưu text/chunk/embedding vào SQLite theo conversation
```

Giới hạn hiện tại: 20 MB/file, 100 trang/PDF và 400 chunks/file. File gốc không được lưu.

### Validated chat

```text
question + history
  -> global retrieval
  -> temporary retrieval trong conversation
  -> rerank/merge
  -> generation
  -> source/year/quality guards
  -> repair tối đa một lần
  -> persist answer
  -> SSE answer đã được duyệt
```

## Invariants cần giữ

- Mọi conversation endpoint phải kiểm tra `X-Client-ID` và ownership.
- Không dùng `X-Client-ID` như authentication khi public API.
- Source ID phải thuộc đúng context đã đưa vào prompt, kể cả temporary source.
- Không stream token thô trước khi guards/repair hoàn tất.
- Không đưa file bytes vào SQLite; chỉ lưu metadata, extracted text, chunks và embeddings.
- Không thay schema SQLite mà bỏ qua migration/compatibility cho database hiện có.
- SQLite deployment hiện dùng một Modal container; scale ngang cần external database.

## Kiểm tra sau khi sửa

```powershell
python -m compileall app
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Với thay đổi retrieval/model, chạy thêm smoke tests ở root: `modal_runtime_sanity.py` hoặc `full_modal_runtime_sanity.py`.
