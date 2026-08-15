# Backend FastAPI

[Về README gốc](../README.md)

Thư mục `app/` chứa toàn bộ runtime của ứng dụng: API, conversation memory, temporary
document corpus, Hybrid RAG, generation và guardrails. Notebook huấn luyện nằm trong
[`Training/`](../Training/README.md); model, corpus và index triển khai tuân theo contract tại
[`artifacts/`](../artifacts/README.md).

## Khởi động

Luồng development chuẩn chạy từ repository root:

```powershell
npm run dev
```

Lệnh trên khởi động đồng thời Vite và `modal serve modal_app.py`. Chỉ dùng lệnh dưới đây
khi cần chạy FastAPI local độc lập:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mode local mặc định là `api-only`. `retrieval-only` và `full` cần deployment bundle hợp lệ.

## Bản đồ module

| File | Trách nhiệm | Sửa khi |
|---|---|---|
| [`main.py`](main.py) | Lifespan, CORS, health/readiness, dependency wiring | Thay startup hoặc endpoint hệ thống |
| [`config.py`](config.py) | Settings, runtime mode, artifact path, SQLite path | Thêm biến môi trường hoặc đổi layout |
| [`schemas.py`](schemas.py) | Pydantic request/response models | Thay API contract |
| [`api/conversations.py`](api/conversations.py) | CRUD conversation, upload/delete attachment | Thay ownership hoặc document API |
| [`api/routes.py`](api/routes.py) | Retrieve, chat và validated SSE | Thay request flow hoặc SSE events |
| [`chat/store.py`](chat/store.py) | SQLite conversations, messages, attachments, chunks | Thay persistence hoặc memory |
| [`chat/attachments.py`](chat/attachments.py) | PDF/image extraction, OCR, chunking, embedding | Thay temporary corpus ingestion |
| [`services/rag_service.py`](services/rag_service.py) | Validate và load corpus/index/model | Thay startup hoặc artifact loader |
| [`rag/retrieval.py`](rag/retrieval.py) | E5, FAISS, BM25S, RRF, rerank, diversity | Thay retrieval quality |
| [`rag/prompting.py`](rag/prompting.py) | History, prompt budget, output parser, repair prompt | Thay cấu trúc prompt/answer |
| [`rag/generation.py`](rag/generation.py) | Contextual retrieval, temp/global merge, generation, repair | Thay answer pipeline |
| [`rag/guards.py`](rag/guards.py) | Source/year/format/completeness validation | Thay hallucination safeguards |

## Runtime modes

| `APP_MODE` | Conversation CRUD | Global retrieval | Upload/temp corpus | LLM generation |
|---|---:|---:|---:|---:|
| `api-only` | Có | Không | Không | Không |
| `retrieval-only` | Có | Có | Có | Không |
| `full` | Có | Có | Có | Có |

`GET /ready` phản ánh các thành phần thực sự đã load; `GET /health` chỉ xác nhận tiến trình
API còn sống.

## Lifecycle

```text
Settings
  -> ConversationStore.initialize()
  -> RAGService.load()
  -> HybridRetriever + TemporaryDocumentService
  -> RAGGenerator (chỉ trong full mode)
  -> gắn dependencies vào app.state
  -> mount system, conversation và RAG routers
```

## Conversation memory

```text
X-Client-ID + conversation_id
  -> kiểm tra conversation thuộc client
  -> đọc messages gần nhất từ SQLite
  -> dùng history để contextualize retrieval và tạo prompt
  -> lưu user message, assistant answer và cited sources
```

SQLite là persistence layer, không phải model memory. Backend chủ động đọc lịch sử và truyền
vào pipeline ở mỗi lượt chat. Toàn bộ message được lưu, nhưng prompt chỉ lấy phần gần nhất theo
giới hạn số message và token/character budget.

Các bảng chính:

| Bảng | Nội dung |
|---|---|
| `conversations` | Cửa sổ chat và anonymous owner |
| `messages` | User/assistant messages cùng source JSON |
| `attachments` | Metadata và trạng thái file |
| `temporary_chunks` | Text, page, source ID và embedding của file |

Xóa attachment sẽ xóa temporary chunks liên quan; xóa conversation sẽ cascade messages,
attachments và temporary corpus. File bytes gốc không được lưu trong SQLite.

## Upload và temporary corpus

```text
PDF / PNG / JPEG / WebP
  -> validate định dạng và dung lượng
  -> PDF: PyMuPDF trích text từng trang
  -> trang scan/ít text hoặc ảnh: Tesseract OCR vie+eng
  -> chunk 220 từ, overlap 40 từ
  -> multilingual E5 embedding
  -> lưu vào SQLite, cô lập theo conversation
```

Giới hạn hiện tại:

- 20 MB mỗi file;
- tối đa 100 trang mỗi PDF;
- tối đa 400 chunks mỗi file;
- trang PDF có dưới 80 ký tự extracted text được render ở tỉ lệ 2x để OCR.

Upload endpoint chờ extraction/OCR/chunking/embedding hoàn tất rồi mới trả response. Trạng thái
`processing` không có nghĩa ingestion đang chạy qua background queue. OCR chỉ chuyển hình
thành text; hệ thống chưa hiểu layout phức tạp, bản đồ, biểu đồ hoặc chữ viết tay như VLM.

## Chat và validated SSE

```text
question + history
  -> global Hybrid RAG
  -> temporary retrieval trong conversation
  -> rerank và merge contexts
  -> generation
  -> source/year/quality guards
  -> structured section expansion hoặc evidence-only repair
  -> persist answer
  -> phát SSE
```

SSE không stream token thô từ model. Backend hoàn tất generation, validation và repair trước,
sau đó chia answer cuối thành các `answer_delta`. Thứ tự event:

1. `status: processing`
2. `ping` khoảng mỗi 8 giây nếu tác vụ còn chạy
3. `status: validated`
4. một hoặc nhiều `answer_delta`
5. `sources`
6. `debug` khi request bật debug
7. `done`, hoặc `error` nếu thất bại

## Cấu hình generation

Runtime đọc `config/inference_config.json` trong deployment bundle khi khởi động. Contract và
cách cập nhật file được mô tả tại [`artifacts/README.md`](../artifacts/README.md). Hai phần quan
trọng khi muốn answer dài và có bố cục:

- `generation.max_new_tokens`: ngân sách token model được phép sinh;
- `generation.repair_min_new_tokens`: minimum generation cho repair câu thường;
- `generation.repair_min_multi_part_new_tokens`: minimum generation cho repair câu nhiều ý;
- `generation.enable_structured_expansion`: sinh riêng ba section phụ khi model trả lời ngắn;
- `generation.section_max_new_tokens`: trần token cho mỗi section pass;
- `prompt.default_system`: yêu cầu nội dung và các heading như câu trả lời, bằng chứng,
  alternatives và kết luận.

Frontend đã render Markdown/GFM, nên không cần thay component chỉ để hiển thị heading. Với câu
trả lời factual, quality critic kiểm tra đủ bốn heading đúng thứ tự, section không rỗng và độ dài
tối thiểu 140 từ (180 từ cho câu nhiều ý). Câu ngắn/thiếu section sẽ dùng structured expansion;
vi phạm source/year/quality khác dùng evidence-only repair. Câu từ chối do OOD/thiếu evidence
được miễn cấu trúc này. Có thể override bằng các key
`guards.require_structured_answer`, `guards.min_answer_words` và
`guards.min_multi_part_answer_words` trong inference config.

Sau khi thay config trên Modal Volume, phải restart `modal serve`/`modal deploy` hoặc container
đang chạy vì `RAGService` chỉ load config một lần trong startup. Xem lệnh cập nhật tại
[`artifacts/README.md`](../artifacts/README.md).

## Invariants

- Mọi conversation/attachment/chat request phải kiểm tra `X-Client-ID` và ownership.
- `X-Client-ID` chỉ là anonymous demo identity, không phải authentication.
- History là ngữ cảnh hội thoại, không được dùng như evidence lịch sử.
- Source ID phải thuộc context thật sự đã đưa vào prompt, gồm cả global và temporary sources.
- Không phát answer ra client trước khi guards/repair hoàn tất.
- Không lưu file bytes vào SQLite.
- Không đổi SQLite schema mà bỏ qua compatibility/migration.
- Deployment SQLite hiện giữ `max_containers=1`; scale ngang cần external database.

## Kiểm tra sau khi sửa

```powershell
python -m compileall app
npm run frontend:lint
npm run frontend:build
modal run modal_runtime_sanity.py
modal run full_modal_runtime_sanity.py
```

Hai Modal runtime sanity scripts kiểm tra retrieval/full generation cơ bản. Chúng chưa phải test
suite cho conversation persistence, upload/OCR hoặc toàn bộ SSE contract.
