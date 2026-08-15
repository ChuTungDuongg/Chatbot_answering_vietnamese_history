# Modal smoke tests

[Về README gốc](../README.md)

Thư mục `modal_test/` chứa các bài kiểm tra nhỏ để xác nhận Modal account, GPU và Volume hoạt động trước khi chạy full RAG. Đây là smoke/learning scripts, không phải automated test suite của ứng dụng.

## Điều kiện

```powershell
python -m pip install modal
modal setup
```

GPU và Volume có thể phát sinh chi phí Modal. Dừng run/container sau khi kiểm tra xong.

## Các script

| File | Lệnh | Kiểm tra | Kết quả mong đợi |
|---|---|---|---|
| [`modal_hello.py`](modal_hello.py) | `modal run modal_test/modal_hello.py` | Function cơ bản và remote OS | In `Hello from Modal!` cùng platform |
| [`modal_gpu_test.py`](modal_gpu_test.py) | `modal run modal_test/modal_gpu_test.py` | L4, CUDA, PyTorch và `nvidia-smi` | `CUDA available: True` và thông tin GPU |
| [`modal_volume_test.py`](modal_volume_test.py) | `modal run modal_test/modal_volume_test.py` | Mount/read `vn-history-artifacts` | Đọc được `/artifacts/test_modal.txt` |
| [`test_modal.txt`](test_modal.txt) | Không chạy trực tiếp | File mẫu để đưa lên Volume | Nội dung được script volume in ra |

`modal_volume_test.py` dùng `create_if_missing=False`; Volume `vn-history-artifacts` và file `/test_modal.txt` phải tồn tại trước khi chạy.

## Sau khi smoke test thành công

Các kiểm tra sát với ứng dụng thật nằm ở root:

| File | Phạm vi |
|---|---|
| [`../modal_artifact_sanity.py`](../modal_artifact_sanity.py) | Cấu trúc/count/checksum logic của deployment bundle |
| [`../modal_runtime_sanity.py`](../modal_runtime_sanity.py) | Load retrieval runtime và chạy truy vấn mẫu |
| [`../full_modal_runtime_sanity.py`](../full_modal_runtime_sanity.py) | Load model trên GPU và chạy full chat pipeline |
| [`../modal_app.py`](../modal_app.py) | ASGI deployment thật, không phải test |

Chạy theo thứ tự từ nhẹ tới nặng: hello, volume, GPU, artifact sanity, retrieval sanity, rồi full runtime sanity.

## Khi chỉnh sửa

- Giữ mỗi script độc lập và có `@app.local_entrypoint()` để chạy trực tiếp bằng Modal CLI.
- Không đưa API key/token vào source hoặc file mẫu.
- Một smoke test nên kiểm tra một capability rõ ràng và fail bằng exception/exit code khi capability đó không hoạt động.
- Logic production phải nằm trong `app/` hoặc root deployment scripts, không đặt trong thư mục này.
