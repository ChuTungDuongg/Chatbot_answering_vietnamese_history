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

Browser fixture dùng password thử nghiệm, dữ liệu thử riêng và chỉ lắng nghe loopback. Playwright cần Chrome cài sẵn. Không chạy fixture thử trên mạng công khai.

## Kiến trúc và migration v1.1

Khi truy cập một API v1.1 lần đầu, migration idempotent tạo thêm `evaluation_experiments`, `evaluation_runs`, `evaluation_evidence`, `evaluation_blind_sessions`, `evaluation_blind_assignments`, `evaluation_blind_reviews`, `evaluation_import_batches` và `evaluation_id_counters`. Các bảng v1.0 không bị drop/recreate; run completed có một response tương ứng trong bảng v1.0 để Human Review hiện có tiếp tục hoạt động. Review luôn gắn response, không sửa question. Experiment gắn snapshot ID, run chỉ nhận question có trong snapshot đó. Các endpoint dùng cùng owner session và feature flag v1.0.

DDL và seed/migration dùng transaction rõ ràng: lỗi tạo một bảng rollback toàn migration. Có indexes theo response/question/run/snapshot cho lookup và khóa ngoại. ID tự cấp nằm trong `HIST-EVAL-001`–`HIST-EVAL-999999999`; numeric ID import ngoài khoảng này được giữ như external ID và không làm tràn SQLite counter. Việc khởi động sửa counter không hợp lệ từ bản cũ mà không đổi question IDs.

Không tự chạy benchmark hoặc gọi LLM. Một experiment chứa metadata do người dùng cung cấp; giá trị chưa biết để trống/null. Trong **Experiments**, tạo experiment từ snapshot rồi thêm output thủ công, hoặc dùng import batch. Không dán API key, mật khẩu, token, private chain-of-thought hay secret vào metadata, output hoặc evidence. Input v1.1 và backup từ chối một số mẫu credential phổ biến; kiểm tra này không thể phát hiện mọi bí mật. Cấu hình experiment bị khóa khi đã có run; run bị khóa khi có review hoặc blind assignment. Xóa chỉ khả dụng khi không có dữ liệu phụ thuộc.

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

Không có LLM judge, tự xác minh sử liệu, benchmark runner hay capture tự động từ chatbot. Evidence/telemetry phải được nhập từ run thực. Không có cách ly quyền reviewer độc lập cho blind A/B; không bảo đảm phát hiện mọi dấu hiệu định danh trong chính output. Review đã lưu chỉ đọc; tạo review mới hoặc phiên blind mới để đánh giá lại. JSONL/CSV export question không tự chứa manifest; xuất JSON cùng bộ lọc. Chưa kiểm tra deploy hoặc restore Modal thực tế.
