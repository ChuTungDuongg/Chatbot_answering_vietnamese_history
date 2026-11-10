# Triển khai Sử Việt trên Modal

Một function ASGI `fastapi_app` phục vụ giao diện React và API tại cùng một URL HTTPS. Không cần backend URL riêng hay máy local sau khi chạy `modal deploy`. Mọi API, `/ready`, tài liệu API và giao diện đều yêu cầu phiên đăng nhập; `/health` chỉ là liveness công khai. Modal bắt buộc `AUTH_ENABLED=true`, và ứng dụng từ chối khởi động khi thiếu hoặc sai secret. Các endpoint cũ từ triển khai trước cần được xóa hoặc dừng riêng trên Modal Dashboard; deploy mới không tự xóa app cũ có tên khác.

## Chuẩn bị

1. Đăng nhập Modal, tạo/khôi phục các Volume `vn-history-artifacts`, `vn-history-hf-cache` và `vn-history-chat-data` theo [Artifacts](ARTIFACTS.md). Corpus và index V1 phải có sẵn. SQLite hội thoại và phiên lưu trên `/data`; deployment giới hạn một container. Không chạy nhiều replicas dùng cùng SQLite trên Modal Volume.
2. Tạo secret mật khẩu bằng lệnh sau trong PowerShell tại root repo, sau khi cài dependencies runtime và Modal CLI:

   ```powershell
   python scripts/create_modal_auth_secret.py
   ```

   Nhập mật khẩu riêng hai lần tại prompt ẩn, tối thiểu 12 ký tự. Script tạo hash Argon2id và secret ký session ngẫu nhiên, rồi gửi trực tiếp vào Modal Secret `vn-history-auth`; không đưa mật khẩu vào lệnh, lịch sử shell, Git hoặc chat. Có thể đặt tên khác bằng `--name` và cấu hình `MODAL_AUTH_SECRET_NAME` tương ứng. Nếu đã có secret cùng tên, tạo secret tên mới để xoay vòng.
3. Để bật OpenAI, tạo Modal Secret riêng `vn-history-openai` chứa `OPENAI_API_KEY` qua Modal Dashboard, và đặt các biến không bí mật sau trong `.env` (file này được gitignore):

   ```dotenv
   OPENAI_ENABLED=true
   MODAL_OPENAI_SECRET_NAME=vn-history-openai
   MODAL_AUTH_SECRET_NAME=vn-history-auth
   MODAL_GPU_CLASS=A100-40GB
   ```

   Chỉ nhập key trong giao diện Secret an toàn, không đưa vào code, frontend hay command line. Không cần `OPENAI_ACCESS_TOKEN` cho người dùng đã đăng nhập. OpenAI có giới hạn request, concurrency và số output token; hãy đặt budget tài khoản OpenAI phù hợp. Một số cấu hình legacy service-to-service vẫn yêu cầu token riêng; xem [cấu hình](CONFIGURATION.md#openai-responses-api).
4. FAISS là mặc định. Để thêm Qdrant, tạo Secret chứa `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, chọn `RETRIEVAL_AVAILABLE_BACKENDS=faiss,qdrant` và `MODAL_QDRANT_SECRET_NAME`. Không sửa collections hiện có. Để dùng vLLM, xem [vLLM](VLLM_LATENCY_EXPERIMENTS.md): runtime hiện chỉ hỗ trợ một trong hai chat mode cùng lúc. Transformers hỗ trợ cả Hybrid và Central.

## Deploy và sử dụng

```powershell
modal deploy modal_app.py
modal app logs vn-history-rag-api --since 2h
```

Mở URL `fastapi_app` được CLI và Modal Dashboard hiển thị, lưu URL đó vào bookmark, và đăng nhập. Không cần `VITE_API_BASE_URL` hay `MODAL_GPU_API_URL`. Khởi động lạnh có thể mất nhiều phút do tải retrieval và model; GPU phát sinh chi phí khi container hoạt động. `A100-40GB` được đề xuất cho cấu hình Transformers kết hợp hai model; kiểm tra quota và giá Modal trước khi sử dụng. Scale-to-zero sau khoảng 120 giây không hoạt động. `LOCAL_INFERENCE_ENABLED=false` chọn CPU, không tải Qwen, nhưng vẫn cần corpus và retrieval models; chế độ đó chỉ phục vụ inference từ xa.

Đăng xuất bằng nút trong giao diện. Cookie HttpOnly/Secure/SameSite=Lax hết hạn sau tối đa bảy ngày; khi hết hạn cần đăng nhập lại. Sau năm lần nhập sai, đăng nhập bị giới hạn 15 phút; giới hạn và session lưu cùng SQLite để tồn tại qua restart. Xoay mật khẩu bằng cách tạo Modal Secret mới với `python scripts/create_modal_auth_secret.py --name TEN_MOI`, đổi `MODAL_AUTH_SECRET_NAME` và deploy lại; secret ký mới vô hiệu hóa các session cũ. Sau khi kiểm tra, có thể xóa secret cũ trên Dashboard.

## Kiểm tra

Ở cửa sổ riêng tư chưa đăng nhập, `/` phải chuyển đến `/login`, còn `/ready`, `/api/v1/conversations`, `/docs`, `/openapi.json` phải trả 401. Kiểm tra trên Dashboard không còn endpoint cũ truy cập trực tiếp không xác thực. Đăng nhập rồi thử tạo hội thoại, Hybrid, Central, stream và Stop; thử cả Luna/Terra và reasoning với OpenAI. Các request OpenAI thực tính phí. Các model, Qdrant và GPU chỉ khả dụng khi tài khoản và tài nguyên tương ứng đã được cấu hình; lỗi model/quota được báo thay vì tự chuyển provider. Monitoring vẫn yêu cầu admin token riêng.

Các lệnh chẩn đoán (GPU smoke có thể tính phí):

```powershell
modal volume ls vn-history-artifacts /corpus_v1/runtime --json
modal run --detach --write-result reports/restore/runtime_smoke.json modal_app.py::runtime_smoke --generate --central
```

Không gửi mật khẩu, cookie, key hoặc secret vào logs hay ảnh chụp màn hình. Nếu deployment lỗi, kiểm tra secret bắt buộc, artifacts, quota GPU, model access và log đã loại bỏ dữ liệu nhạy cảm. `modal serve` chỉ là phiên phát triển tạm thời.
