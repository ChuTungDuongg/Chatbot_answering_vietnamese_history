# Sử Việt Eval Studio

Công cụ nội bộ để con người biên soạn và đánh giá câu trả lời. Studio không gọi mô hình, không tự tạo câu hỏi, không tự chấm điểm và không kiểm chứng tính đúng của sử liệu.

## Cấu hình và chạy local

1. Dùng môi trường Python của dự án và cài dependencies theo `requirements.txt`. Chuẩn bị owner authentication theo [DEPLOYMENT.md](DEPLOYMENT.md): `AUTH_ENABLED=true`, `AUTH_PASSWORD_HASH` là Argon2id hash và `AUTH_SESSION_SECRET` là secret ngẫu nhiên ít nhất 32 byte. Không đưa secret vào Git. Cookie đăng nhập dùng cờ Secure; truy cập qua localhost hoặc HTTPS.
2. Đặt `EVAL_STUDIO_ENABLED=true`, `APP_MODE=api-only` (nếu chỉ cần Studio), và `CHAT_DATABASE_PATH` trỏ tới SQLite cần lưu. Nếu đã có cơ sở dữ liệu chat, Studio chỉ thêm bảng riêng; sao lưu tệp trước các thao tác quản trị dữ liệu.
3. Build cùng origin và chạy FastAPI. Ví dụ PowerShell từ root repo:

   ```powershell
   $env:VITE_API_BASE_URL = '/'
   npm --prefix frontend ci
   npm --prefix frontend run build
   $env:FRONTEND_DIST_DIR = (Resolve-Path frontend/dist).Path
   $env:EVAL_STUDIO_ENABLED = 'true'
   .conda/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

   Dùng Python interpreter của bạn nếu không có `.conda`. Dùng `/` thay cho chuỗi rỗng để PowerShell không xóa biến môi trường rồi Vite đọc lại URL remote từ `frontend/.env`. API, export, snapshot và backup của Studio luôn dùng cùng origin với owner cookie, độc lập với cấu hình API chatbot. Các biến auth vẫn phải được cung cấp từ môi trường hoặc `.env` trước khi chạy. Mở `http://localhost:8000/eval-studio`, đăng nhập bằng owner password. Không đặt `AUTH_ENABLED=false`: API Studio từ chối truy cập nếu không có owner session, kể cả khi flag được bật.

## Modal

Triển khai theo [DEPLOYMENT.md](DEPLOYMENT.md), giữ secret auth và URL ASGI hiện tại. Trước khi deploy, đặt `EVAL_STUDIO_ENABLED=true` trong môi trường chạy `modal deploy modal_app.py`; flag mặc định là `false`. Mở `https://<URL-fastapi_app>/eval-studio` và đăng nhập. Tắt bằng cách đặt flag về `false` và deploy lại; dữ liệu vẫn nằm trong volume.

Các bảng `evaluation_datasets`, `evaluation_questions`, `evaluation_responses`, `human_reviews`, `dataset_versions` nằm trong `CHAT_DATABASE_PATH` hiện có (`/data/chat.sqlite3` trên Modal). Volume `vn-history-chat-data` được commit sau mỗi thay đổi Studio; deploy giữ một container/writer như cấu hình hiện tại. Không dùng nhiều container ghi cùng SQLite trên Volume. Backup và khôi phục toàn bộ SQLite/Volume theo quy trình vận hành của bạn. Chưa chạy thử deploy hay khôi phục Modal trong thay đổi này. Studio dùng URL ASGI cùng origin; gateway frontend-only cũ không chuyển owner cookie và không hỗ trợ luồng này.

## Biên soạn và kiểm duyệt

Lần truy cập API đầu tiên khởi tạo idempotent dataset `vietnamese-history-eval-v1` với 40 slot `HIST-EVAL-001`–`040`, phân bổ 6/12/8/8/4/2; 20 slot lẻ được chọn smoke và mọi nhóm có ít nhất một slot smoke. Nội dung lịch sử để trống. Có thể tạo thêm, sao chép hoặc xóa slot; khởi tạo không chạy lại và không xóa dữ liệu hiện có.

Trong **Ngân hàng câu hỏi**, chọn slot, điền các mục trong editor, thêm từng fact, luận điểm, nguồn và phương án khác. Mục ranking có tiêu chí riêng và không yêu cầu một thứ tự duy nhất. Nội dung tự lưu sau khoảng 700 ms; có thể nhấn **Lưu** hoặc Ctrl/⌘+S. Chuyển câu đợi bản lưu đang chạy. Thông báo lỗi/xung đột revision cần được xử lý trước khi rời editor; nếu một tab khác đã ghi, sao chép nội dung chưa lưu rồi tải lại để tránh ghi đè. Khi một câu đã approved cần sửa thành trạng thái chưa duyệt trước nếu sửa sẽ làm thiếu trường bắt buộc.

Trong **Kiểm duyệt & phê duyệt**, chọn `needs_review` hoặc `approved`. Approval yêu cầu câu hỏi, đáp án, ít nhất một fact, một luận điểm và một nguồn; ranking cần tiêu chí, câu gây tranh luận cần ghi chú bất định. Kiểm tra chất lượng chỉ phát hiện thiếu trường, quota/smoke và chuỗi câu hỏi gần trùng; nó không xác minh sự kiện. **Dữ liệu & Backup** hiển thị các cảnh báo, bao gồm metadata output và tên reviewer còn thiếu.

## Human review

Trong **Human review**, chọn câu hỏi, dán output và evidence, ghi provider/model/chat mode/reasoning/retrieval nếu biết, rồi lưu. Hoặc nhập JSON/JSONL gồm các object với `question_id`, `answer`, và các trường metadata đó; nếu thiếu `question_id`, câu đang chọn được dùng. JSON object cũng có thể chứa `responses`. Nhập output qua preview và một transaction; dòng lỗi làm cả batch không được ghi. Với JSONL, question_id thiếu dùng câu đang chọn.

Chọn output đã lưu, tự chọn điểm 0–5 cho cả năm tiêu chí, quyết định và error tags; sau đó lưu review. Điểm thang 100 được backend tính từ trọng số 30/20/20/20/10. Có thể lưu nhiều output và nhiều review cho mỗi câu. Review đã lưu hiện chỉ đọc; để sửa kết luận, lưu một review mới. Nhấn **Xuất review** để nhận JSON riêng gồm responses và reviews.

Gửi lại review giống hoàn toàn review mới nhất của response trả về cùng record (kể cả hai request đồng thời). Một kết luận khác tạo record mới; có thể quay lại điểm cũ bằng một review mới sau khi đã thay đổi kết luận. Chuyển output đặt lại rubric trống để không vô tình dùng điểm của output trước. Run v1.1 dùng reference từ snapshot khi chấm, kể cả working question đã được sửa.

## Import, export và phiên bản

Ở **Dữ liệu & Backup**, chọn JSON (manifest và cases), JSONL (một case mỗi dòng), hoặc CSV (metadata và question). Có thể lọc smoke, approved-only và 40 ID gốc; không chọn bộ lọc ID gốc sẽ bao gồm các câu thêm. Xuất JSON cùng bộ lọc để lưu manifest cho bản JSONL tương ứng. Manifest chứa thời gian tạo/export, phân bổ, số case, schema version và SHA-256 của JSON canonical (`sort_keys=True`, UTF-8, separators `,` và `:`). Không có nội dung lịch sử được tạo sẵn.

Import JSON dạng `{ "cases": [...] }`, mảng cases hoặc JSONL qua nút nhập; trường hợp có ID tồn tại cần xác nhận ghi đè. Mọi case được Pydantic validate và import trong một transaction; ID không có trong tệp được giữ nguyên. Import ghi đè tăng revision, giữ response và review đã liên kết. Nên export backup trước khi xác nhận.

Tạo snapshot bằng tên phiên bản chưa dùng, ví dụ `1.0.0`; snapshot lưu manifest/cases bất biến, xuất lại từ danh sách phiên bản hoặc `GET /api/v1/eval/snapshots/1.0.0`. Dataset đang soạn có version `draft`; snapshot không tự cập nhật khi câu hỏi thay đổi. Để tái lập benchmark, lưu snapshot cùng output và cấu hình model. Với run v1.1, question phải vẫn tồn tại trong working dataset do khóa ngoại của bảng response v1.0; snapshot không bị sửa nếu question bị xóa, nhưng cần khôi phục question trước khi nhập run.

## Kiểm thử

```powershell
.conda/python.exe -m pytest -q
npm --prefix frontend run lint
npm --prefix frontend run build
npm --prefix frontend test
# Cửa sổ thứ nhất, sau khi build same-origin:
.conda/python.exe tests/eval_browser_server.py
# Cửa sổ thứ hai:
cd frontend
npx playwright test --config eval.playwright.config.js
```

Browser fixture dùng password thử nghiệm, dữ liệu thử riêng và chỉ lắng nghe loopback. Nó inject executor offline vào app trong bộ nhớ; không có HTTP option hoặc environment flag để chọn fake provider trên production. Playwright dùng Chrome trên local, Chromium được cài trong CI. Không chạy fixture thử trên mạng công khai.

## Kiến trúc và migration v1.1

Khi truy cập một API v1.1 lần đầu, migration idempotent tạo thêm `evaluation_experiments`, `evaluation_runs`, `evaluation_evidence`, `evaluation_blind_sessions`, `evaluation_blind_assignments`, `evaluation_blind_reviews`, `evaluation_import_batches` và `evaluation_id_counters`. Các bảng v1.0 không bị drop/recreate; run completed có một response tương ứng trong bảng v1.0 để Human Review hiện có tiếp tục hoạt động. Review luôn gắn response, không sửa question. Experiment gắn snapshot ID, run chỉ nhận question có trong snapshot đó. Các endpoint dùng cùng owner session và feature flag v1.0.

DDL và seed/migration dùng transaction rõ ràng: lỗi tạo một bảng rollback toàn migration. Có indexes theo response/question/run/snapshot cho lookup và khóa ngoại. ID tự cấp nằm trong `HIST-EVAL-001`–`HIST-EVAL-999999999`; numeric ID import ngoài khoảng này được giữ như external ID và không làm tràn SQLite counter. Việc khởi động sửa counter không hợp lệ từ bản cũ mà không đổi question IDs.

Luồng nhập v1.1 không tự chạy benchmark hoặc gọi LLM. Một experiment chứa metadata do người dùng cung cấp; giá trị chưa biết để trống/null. Trong **Experiments**, tạo experiment từ snapshot rồi thêm output thủ công, hoặc dùng import batch. Không dán API key, mật khẩu, token, private chain-of-thought hay secret vào metadata, output hoặc evidence. Input v1.1 và backup từ chối một số mẫu credential phổ biến; kiểm tra này không thể phát hiện mọi bí mật. Cấu hình experiment bị khóa khi đã có run hoặc benchmark v1.2; run bị khóa khi có review, blind assignment hoặc measurement tự động. Xóa chỉ khả dụng khi không có dữ liệu phụ thuộc.

Record run mẫu dùng trong JSONL (thay ID bằng giá trị thật):

```json
{"experiment_id":"<experiment-id>","question_id":"HIST-EVAL-001","external_id":"run-001","answer":"Output do model trả về","latency_ms":1200,"input_tokens":100,"output_tokens":80,"tool_calls":2,"cost":null,"evidence":[{"chunk_id":"chunk-01","title":"Nguồn","excerpt":"Trích đoạn được capture","selected":true,"cited":true}]}
```

Có thể dùng `completion_status: "failed"` cùng `error_reason`; không tự điền số đo không có thật. `trace_captured` là khai báo nguồn dữ liệu đã capture, không phải xác nhận tính đúng. Evidence là bản sao người dùng cung cấp cho đúng run; Inspector không truy vấn lại Qdrant và hiện `Not captured` khi thiếu. Chỉ các trường telemetry/evidence được schema cho phép được lưu; `trace` có thể chứa `retrieval_backend`, `final_context_chunk_ids`, `cited_chunk_ids` và `tools` với name/status/duration_ms/error_code. Không nhập raw debug trace hay private reasoning. Trong Human Review, dùng Evidence Inspector cho tham chiếu từ snapshot, thêm nhãn nguyên nhân và ánh xạ must-cover point tới chunk ID hoặc `missing`.

## Comparison và blind A/B

Chọn hai hoặc nhiều experiments ở **Model Comparison**. Chỉ experiment cùng snapshot được ghép cặp; các khác biệt/thiếu sót metadata hiển thị để người dùng tự xét khả năng so sánh. Với mỗi experiment/question, thống kê dùng run completed mới nhất có review và review mới nhất của response đó. Các run chưa có review không trở thành điểm 0. Tổng số, số đã chấm, thiếu review, thất bại, điểm 5 tiêu chí, phân phối, W/T/L, danh mục và telemetry đều lấy từ dữ liệu lưu thật. Các bộ lọc model/mode/backend lọc danh sách experiment; category/difficulty/smoke/diagnosis lọc kết quả. Chọn điểm trong bảng để xem question snapshot, output, review và evidence.

"Mới nhất" theo thứ tự insert SQLite (`rowid`), không theo timestamp do client cung cấp. Mỗi question đóng góp tối đa một điểm cho mỗi experiment. Chọn run/review trước rồi mới lọc diagnosis; lọc không hồi sinh run cũ. Telemetry là trung bình trên **mọi lần chạy** trong bộ lọc có giá trị được capture, gồm cả lần chưa chấm/thất bại; failed runs là số lần chạy, không phải số question. Ví dụ 40 outputs/20 reviews chỉ có 20 câu được tính điểm. Comparison và evidence đọc trong một SQLite read transaction để không trộn trạng thái khi có writer.

Trong **Blind A/B**, chọn snapshot và hai experiments cùng snapshot. Server lấy run completed mới nhất theo question, tạo assignment bằng random an toàn và lưu cố định; view blind chỉ nhận question, reference và hai answer đã ẩn các tên model/provider biết được, không nhận mapping, source metadata hoặc trace. Nhập đủ hai rubric và preference rồi lưu trước khi reveal. Assignment/review không thể ghi đè, kể cả hai lượt gửi đồng thời. Có thể tạo phiên mới để đánh giá lại. Blind không có vai trò reviewer riêng: người có owner access vẫn có thể mở workspace/API experiment khác và tự tìm mapping; để bảo đảm cách ly reviewer theo quyền, cần triển khai tài khoản/quyền riêng. Answer do model tự viết có thể chứa dấu hiệu nhận diện không thể tự động phát hiện hết; kiểm tra dữ liệu trước khi dùng cho blind thực sự.

## Atomic import

Question, legacy response và run hỗ trợ JSON/JSONL/CSV qua preview trong giao diện hoặc `POST /api/v1/eval/batch/preview` và `/batch/apply`. Giới hạn 2 MB và 1.000 records; HTTP request tối đa 2,1 MB kể cả JSON wrapper. Preview hiển thị mẫu, lỗi theo dòng và ID xung đột; apply cần token của preview và lựa chọn từ chối/bỏ qua/ghi đè đối với question. Revision và toàn bộ cấu hình experiment được kiểm tra lại trong transaction ngay trước khi ghi; đổi snapshot/model sau preview bắt buộc preview lại. Mọi lỗi rollback toàn batch, kể cả lỗi insert ở dòng cuối. File retry giống hệt cùng chính sách trả kết quả cũ **chỉ khi records/evidence đã ghi vẫn nguyên vẹn**; nếu bị sửa/xóa thì trả 409, cần tạo batch mới với nội dung/ID thao tác mới. Receipt từ bản cũ thiếu fingerprint cũng trả 409 để tránh báo thành công sai. Run hỗ trợ external_id (unique trong từng experiment) hoặc content hash ổn định. CSV metadata question nhận UTF-8 BOM, chỉ cập nhật các cột được cung cấp và giữ ground truth còn lại, nhưng yêu cầu `version` khi ghi đè. Không có bulk overwrite review hoặc snapshot.

## Backup và khôi phục

Nút **Tải backup** hoặc CLI tạo archive gồm manifest (schema, thời điểm, số record, SHA-256) và SQLite chỉ chứa các bảng evaluation. SQLite backup API tạo snapshot nhất quán ngay cả khi có writer; auth/session/chat không nằm trong archive. File người dùng tự nhập vẫn có thể chứa thông tin nhạy cảm không khớp mẫu phát hiện: rà soát và lưu backup như dữ liệu nội bộ. Backup từ chối khi phát hiện mẫu credential phổ biến.

```powershell
# Backup evaluation; đích phải chưa tồn tại.
.conda/python.exe -m app.eval_studio.backup backup data/chat.sqlite3 data/eval-backup.zip
# Restore vào database evaluation mới; không ghi đè bất kỳ file nào.
.conda/python.exe -m app.eval_studio.backup restore data/eval-backup.zip data/restored-eval.sqlite3
# Giữ chat/auth từ database gốc bằng cách tạo bản sao MỚI, chỉ thay evaluation trên bản sao:
.conda/python.exe -m app.eval_studio.backup restore-copy data/eval-backup.zip data/recovered-copy.sqlite3 --base data/chat.sqlite3 --replace-evaluation
```

`restore-copy` từ chối nếu đích tồn tại; nếu database gốc đã có evaluation data, chỉ cờ `--replace-evaluation` cho phép thay chúng **trên bản sao**, nguồn không đổi. CLI kiểm tra checksum, schema version, cấu trúc bảng, foreign keys và integrity. Không cho app dùng bản sao tới khi đã kiểm tra dữ liệu. Để chuyển đổi thực tế: dừng app và writer, sao lưu toàn bộ database/volume hiện tại, tạo bản sao, kiểm tra số record và đăng nhập trong môi trường cô lập, sau đó chủ ứng dụng thay đường dẫn database theo quy trình vận hành và giữ bản cũ để rollback. Không thay file đang có kết nối hoạt động.

Restore kiểm tra cả CHECK/UNIQUE và unique indexes, không chỉ tên/cột/foreign keys. Lookup indexes được giữ; archive cũ thiếu nonunique indexes vẫn được nhận rồi migration bổ sung chúng. SHA-256 phát hiện hỏng nội dung, không phải chữ ký xác thực tác giả archive. Đường dẫn có ký tự như `#` được URL-encode khi mở read-only SQLite. Bộ lọc secrets kiểm tra cả JSON đã decode để không bỏ sót escape/newline; vẫn cần owner tự rà soát thông tin nhạy cảm.

## API và phạm vi triển khai

Mọi route dưới `/api/v1/eval` yêu cầu feature flag và owner session; POST/PATCH/DELETE dùng Origin/CSRF protection hiện có. Endpoint synchronous chạy trong threadpool FastAPI; SQLite/file I/O không chạy trực tiếp trên async event loop. Không có endpoint restore live database.

| Nhóm | Endpoint |
| --- | --- |
| Question CRUD/filter, validation | `GET/POST /questions`, `GET/PATCH/DELETE /questions/{id}`, `GET /validation` |
| Responses/reviews | `GET/POST /responses`, `GET/POST /reviews` |
| Export/legacy import | `GET /export`, `GET /export/reviews`, `POST /import` |
| Snapshot bất biến | `GET/POST /snapshots`, `GET /snapshots/{version}` |
| Experiment | `GET/POST /experiments`, `PATCH/DELETE /experiments/{id}` |
| Run/evidence | `GET/POST /runs`, `PATCH/DELETE /runs/{id}`, `GET /runs/{id}/evidence` |
| Analytics | `POST /compare` |
| Blind | `GET/POST /blind`, `GET /blind/{session}`, `POST /blind/{session}/{assignment}/review`, `GET /blind/{session}/{assignment}/reveal` |
| Atomic import và backup | `POST /batch/preview`, `POST /batch/apply`, `GET /backup` |

Trước khi bật trên Modal: backup volume hiện tại; giữ `max_containers=1`/một writer; kiểm tra `/data/chat.sqlite3` nằm trên `vn-history-chat-data`; cấu hình auth secrets và same-origin ASGI URL; bật flag chủ động; kiểm tra đăng nhập/CRUD/export/backup trong môi trường staging; restart container rồi xác minh IDs, reviews và snapshot checksums. Nếu lỗi, tắt flag và rollback bằng database copy đã được kiểm tra khi app đã dừng. Chưa chạy bước xác minh Modal thực tế trong QA local.

Modal: dừng app và bảo đảm không còn container ghi Volume; tải backup về môi trường cô lập, tạo và kiểm tra bản sao từ database volume, giữ bản full-volume cũ, sau đó thực hiện khôi phục theo quy trình Volume của bạn và commit trước khi khởi động lại. Chưa kiểm tra Volume/restart/khôi phục trên Modal thật. Không chạy nhiều container ghi cùng SQLite.

## Giới hạn

Không tự xác minh sử liệu hoặc chạy LLM judge. V1.2 capture tự động chỉ cho benchmark do owner chủ động khởi chạy; output chatbot cũ vẫn nhập thủ công. Không có cách ly quyền reviewer độc lập cho blind A/B; không bảo đảm phát hiện mọi dấu hiệu định danh trong chính output. Review đã lưu chỉ đọc; tạo review mới hoặc phiên blind mới để đánh giá lại. JSONL/CSV export question không tự chứa manifest; xuất JSON cùng bộ lọc. Chưa kiểm tra deploy hoặc restore Modal thực tế.

## V1.2 — Automated Benchmark Runner

V1.2 thêm workspace **Benchmarks** bên cạnh các workspace cũ. Các bảng mới chỉ được tạo idempotent, không drop/recreate bảng cũ: `evaluation_benchmarks`, `evaluation_benchmark_items`, `evaluation_run_measurements`, `evaluation_judge_profiles`, `evaluation_judge_scores`. Job gắn experiment/snapshot; từng item gắn question; mỗi attempt tạo run riêng. Run completed có response v1.0 để tiếp tục Human Review và Blind A/B. Run, evidence, measurement và trạng thái item được ghi trong cùng transaction. Backup schema `1.2` chứa cả năm bảng mới; restore vẫn nhận archive `1.1` và giữ nguyên IDs/relationships.

Backend chia thành các module: `execution.py` bọc runtime hiện có cho riêng request benchmark; `benchmark.py` quản lý job/lease/attempt; `behavior.py` tính deterministic observations; `analytics.py` ghép cặp measurements; `judge.py` lưu profile/result và xuất structured contract. `benchmark_api.py` dùng owner session, feature flag và CSRF/Origin protection giống API cũ. SQLite/file I/O chạy trong threadpool; provider streaming vẫn chạy qua execution boundary và quota của chatbot. Không ghi ephemeral benchmark conversations vào lịch sử chat.

### Workflow từ snapshot đến so sánh

1. Biên soạn các câu cần chạy, sau đó tạo immutable snapshot. Snapshot 40 slots trống không chạy được: mọi câu được chọn phải có question dài 2–1.000 ký tự. Có thể chọn vài ID đã biên soạn hoặc smoke trước khi hoàn thiện cả 40 câu.
2. Tạo experiment trong **Experiments**, chọn cùng snapshot cho các cấu hình cần so sánh. Live execution yêu cầu provider, model ID thật (ví dụ `gpt-6-luna` hoặc `gpt-6.1-terra` theo allowlist runtime), Hybrid/Central, FAISS/Qdrant và `final_k`. OpenAI cần reasoning effort; Central cần planning reasoning effort (`inherit`, `low`, `medium`, `high`, `xhigh`). Local model ID phải khớp model đang được runtime phục vụ. Metadata chưa biết không được suy đoán; live runner sẽ từ chối cấu hình thiếu trường cần thiết.
3. Mở **Benchmarks**, chọn experiment, question IDs hoặc smoke, timeout, requests/minute và retry. Mặc định không retry. Có thể đặt prompt overrides theo tên tài nguyên hiện trên UI và tool description overrides theo tool được phép. Chọn **Xem cấu hình trước khi tạo**, rồi **Tạo benchmark chưa chạy**. Hai thao tác này không gọi model.
4. Kiểm tra cấu hình đã đóng băng và các hash. Đánh dấu xác nhận riêng cho lần **Start/Resume**, rồi nhấn nút chạy. Chỉ thao tác này mới có thể gọi API có tính phí. Runtime phải đang chạy full mode với provider/retrieval hợp lệ; chế độ API-only không thể chạy model thật.
5. Theo dõi số câu pending/running/completed/failed/cancelled. Mở run để xem answer, retrieved evidence, final context, citations, tool diagnostics và metrics. Các trường chưa capture hiển thị `Not captured`/`Chưa ghi`. Không truy vấn lại retrieval backend để dựng evidence cũ.
6. Chấm run completed trong **Human review** theo rubric 30/20/20/20/10, hoặc tạo Blind A/B từ các experiments. Điểm human vẫn riêng với metrics tự động.
7. Chọn **So sánh tự động** trong Benchmarks, chọn experiments và một benchmark cụ thể cho mỗi experiment hoặc dùng bản mới nhất. Chỉ cùng snapshot được ghép cặp. Xem từng metric, latency/usage/cost có thật, W/T/L theo question và drill-down. **Model Comparison** v1.1 vẫn giữ cách tổng hợp human review hiện có.

Một experiment chỉ có một configuration identity sau khi tạo benchmark. Để so sánh Prompt v1/v2 hoặc mô tả tool khác nhau, tạo hai experiments. Job lưu prompt đã resolve, tool descriptors, hash cấu hình, git SHA, runtime/retrieval settings, index/corpus fingerprints có sẵn. Start/resume và mỗi câu đều kiểm tra live baseline; thay đổi code/prompt/index/model làm từ chối drift, cần tạo experiment/job mới. Không sửa prompt hoặc ranking production. Tool variants chỉ đổi descriptions; argument schema và tool policy vẫn được runtime kiểm soát. MCP chỉ chạy khi được cấu hình và cho phép rõ ràng; metadata MCP nếu dùng phải là JSON `ToolSteering` hợp lệ.

### Timeout, retry, cancellation và recovery

Mỗi job chạy tuần tự; requests/minute giới hạn thời điểm bắt đầu câu/attempt và quota/concurrency của provider vẫn áp dụng. Retry chỉ cho lỗi tạm thời được phân loại, tối đa 3 lần, exponential backoff; owner chọn trước vì một request timeout vẫn có thể đã bị tính phí. Mỗi attempt đã ghi giữ nguyên error/measurement; không ghi đè attempt cũ. Hoàn tất job có thể còn failed questions: tạo job mới cho một replicate khác. Cancel/resume giữ các câu completed và chỉ thử lại phần chưa hoàn tất. Cancel đúng lúc provider trả xong vẫn giữ answer đã nhận.

SQLite transaction cấp lease 30 giây cho một worker, heartbeat gia hạn trong lúc chạy. Worker hết lease không được ghi hoặc retry attempt của worker mới. Khởi động app không tự tiếp tục paid work. Sau restart/container eviction, mở job, chờ lease hết hạn nếu cần, kiểm tra cảnh báo interrupted attempt và chủ động Resume. Không thể bảo đảm exactly-once billing nếu process chết sau khi upstream nhận request nhưng trước khi SQLite xác nhận kết quả. Timeout/process death có thể làm mất phần usage/trace chưa hoàn tất; các giá trị này giữ null và không được ước tính. App shutdown yêu cầu dừng worker trước khi đóng provider.

Trên Modal, job/items/results dùng cùng `/data/chat.sqlite3` trên Volume hiện có và commit sau các chuyển trạng thái/kết quả. Giữ một container/writer; runner nằm trong process ASGI, không phải durable external queue. Container có thể dừng giữa job; dùng recovery ở trên. Chưa xác minh live Modal persistence/eviction trong v1.2. Không bật runner trên nhiều container cùng Volume SQLite.

### Expected behavior và giới hạn deterministic evaluation

Question Editor có JSON editor **Expected agent behavior** tùy chọn. Metadata này nằm trong snapshot; chỉ question text được gửi vào model benchmark, không gửi reference answer, facts hoặc expected behavior. Ví dụ:

```json
{
  "required_tools": ["search_history"],
  "any_tool_groups": [["get_events", "search_history"]],
  "forbidden_tools": ["search_web"],
  "max_tool_calls": 6,
  "max_repeated_calls": 1,
  "expected_evidence_ids": ["known-chunk-id"],
  "evidence_terms": ["thuật ngữ cần có trong evidence"],
  "expected_exit_reasons": ["evidence_sufficient"],
  "require_citations": true,
  "output_format": "text"
}
```

Đây là expectations, không phải tool allowlist runtime. Có thể dùng `allowed_tools`, `required_json_keys` khi `output_format="json"`, `answerable`, `in_domain`; schema từ chối trường lạ. Không ép một chuỗi tool-call duy nhất. Argument validity dựa schema thực; repetition dựa tên tool + fingerprint arguments, không lưu raw arguments. Tool success và argument-validity rates có denominators quan sát được. Không có trace thì không suy ra từ aggregate count. Tool lựa chọn hợp lý về ngữ nghĩa, lập luận, grounded claims, answerability/OOD vẫn cần human/judge; rule checks không xác nhận chúng.

Tái sử dụng metrics trong `evaluation/metrics/` cho lexical answer overlap, citation ID validity và evidence phrase coverage. Nhãn UI ghi rõ đây là heuristic. Không có composite heuristic "accuracy score"; missing observations và chưa được chấm luôn null. Lexical LCS được bỏ qua nếu answer/reference quá 8.192 ký tự hoặc 512 tokens; evidence checks cần dữ liệu đầy đủ được bỏ qua nếu quá 500 chunks/65.536 ký tự, không tính điểm trên prefix rồi giả như toàn bộ dữ liệu. Artifact gốc vẫn được lưu. Metrics ghi `schema_version=1.2`, `metric_version=behavior-v1` để truy vết.

So sánh tự động dùng đúng một benchmark được chọn cho mỗi experiment và đúng final completed attempt của mỗi question; review human mới nhất của response tương ứng. Các metric được paired riêng chỉ khi cả hai phía có giá trị trên cùng question/snapshot; missing score không là 0. Telemetry trung bình ở bảng chính thuộc final completed attempts; `attempt_totals` gồm cả retry/failed attempts và chỉ có tổng nếu mọi attempt có giá trị. W/T/L lấy phía A làm mốc; delta là B − A và hướng tốt/xấu tùy metric. Điểm human không được thay bằng heuristic hoặc judge score.

### Optional judge contract (offline)

Không có automatic LLM-as-a-Judge execution trong milestone này. API hỗ trợ profile bất biến theo name/version, provider/model/version, configurable five-criterion rubric có tổng weight 100; xuất input/schema contract và import structured scores 0–5. Judge result là ý kiến đánh giá được nhập, không phải sự thật lịch sử đã xác minh. Không chấp nhận trường reasoning/replay tùy ý; notes chỉ dành cho nhận xét ngắn. Owner tự quản lý việc chạy judge bên ngoài và không nhập credentials/private reasoning.

Kết quả judge append-only, external ID retry idempotent, liên kết run/profile cụ thể. Calibration đối chiếu latest judge với latest human review của cùng response, báo số cặp và mean absolute difference trên thang 100. Muốn dùng judge trong automated comparison API phải chọn rõ `judge_profile_id`; mặc định không trộn các judge model/rubric khác nhau. UI MVP hiển thị chỗ cho judge score nhưng chưa có form tạo/chạy judge; dùng API cho contract/import/calibration. Việc chạy judge tự động và UI calibration là phần mở rộng sau.

### API v1.2

Tất cả đường dẫn dưới `/api/v1/eval`, cùng auth/flag cũ; body mutation tối đa 2,1 MB.

| Nhóm | Endpoint |
| --- | --- |
| Khả năng runtime | `GET /benchmark/capabilities` |
| Job | `GET/POST /benchmarks`, `GET /benchmarks/{id}` |
| Chủ động chạy/hủy | `POST /benchmarks/{id}/start` với `{"confirm_live":true}`, `POST /benchmarks/{id}/cancel` |
| Diagnostics/metrics | `GET /runs/{id}/measurements` (null cho legacy run) |
| So sánh tự động | `POST /benchmark/compare` với `experiment_ids`, optional `benchmark_ids`, `judge_profile_id` |
| Judge profile/result | `GET/POST /judge/profiles`, `GET/POST /judge/scores` |
| Structured contract | `GET /judge/profiles/{id}/contract/{run_id}` |
| Calibration | `GET /judge/profiles/{id}/calibration` |

Live benchmarks không tự ước tính cost từ bảng giá: hiện provider không trả cost nên để null; chỉ lưu actual token usage khi đủ các phase được capture. Không capture raw model messages, tool arguments, encrypted replay hoặc private chain-of-thought. Bộ lọc secrets fail closed với mẫu đã biết nhưng không thể bảo đảm nhận diện mọi dữ liệu nhạy cảm trong answer/evidence do người dùng cung cấp.
