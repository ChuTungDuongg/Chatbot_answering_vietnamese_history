# Khôi phục artifact bundle và đưa lên Modal Volume

Thư mục `vn_history_deployment/` là **portable deployment artifact bundle**: có thể backup nguyên thư mục lên Google Drive, tải về máy khác rồi đưa nội dung lên Modal Volume. Bundle chứa adapter, corpus, index và metadata; mã nguồn ứng dụng, môi trường Python và Qwen base weights được quản lý riêng.

**Volume khuyên dùng: `vn-history-artifacts`. Upload dữ liệu trước, `manifest.json` sau, và `artifact_lock.json` CUỐI CÙNG (LAST).**

Các lệnh dưới đây dành cho người dùng thực hiện khi restore. Việc viết README này không chạy Modal, không upload, không deploy và không sửa dữ liệu artifact.

## 1. Cấu trúc thực tế

```text
vn_history_deployment/
├── adapters/
│   ├── central/                  # Central V1 legacy còn được giữ trong backup
│   ├── evidence/
│   ├── history/
│   └── research/
├── config/
│   ├── inference_config.json
│   └── model_registry.json
├── corpus/
│   └── vn_history_rag_chunks_enriched.jsonl
├── retrieval/
│   ├── bm25s_index/
│   └── faiss/
├── artifact_lock.json
├── manifest.json
├── EXPORT_SUCCESS.txt
└── README_MODAL_VOLUME.md
```

- `adapters/`: các adapter artifacts; giữ toàn bộ nội dung khi backup/restore.
- `config/`: cấu hình inference và registry/metadata của các model.
- `corpus/`: corpus dùng cho RAG.
- `retrieval/`: index FAISS, BM25 và metadata retrieval đi kèm.
- `manifest.json`: mô tả bundle và `deployment_id`.
- `artifact_lock.json`: integrity lock với hash/cấu hình/số lượng để kiểm tra bundle; luôn upload cuối cùng.
- `EXPORT_SUCCESS.txt`: tên marker thực tế, có đuôi `.txt`. Runtime trong `app/` và validator local không yêu cầu marker này. Helper sanity cũ có kiểm tra marker, nên vẫn giữ và upload để mang đủ bản backup.
- README này: hướng dẫn đi kèm bundle, không phải cấu hình runtime.

Đừng chỉ copy riêng manifest hoặc lock: hai file đó không chứa trọng số adapter, corpus hay index.

## 2. Khôi phục từ Google Drive

1. Download/copy **toàn bộ** `vn_history_deployment/`; nếu backup là ZIP, giải nén đủ các phần trước.
2. Giữ nguyên cây thư mục con và tên file; chờ Google Drive tải xong, không dùng các file placeholder chưa có dữ liệu local.
3. Trên máy mới, chuẩn bị checkout mã nguồn project tương ứng và Python có các dependency cần thiết. Bundle không chứa `scripts/validate_artifact_bundle.py` hay `modal_app.py`; hai file này thuộc repository.
4. Mở PowerShell và vào thư mục bundle. Đây chỉ là ví dụ, hãy thay đường dẫn theo máy của bạn:

```powershell
Set-Location "C:\Users\PC\Chatbot_answering_vietnamese_history\artifacts\vn_history_deployment"
Get-ChildItem
```

Nếu restore đúng vị trí `repo/artifacts/vn_history_deployment`, repository root nằm ở `..\..`. Nếu đặt bundle nơi khác, dùng đường dẫn tuyệt đối tới repository cho các lệnh validator.

## 3. Kiểm tra Python / Modal CLI

Cú pháp trong README đã được đối chiếu với mã CLI Modal đang cài tại máy viết tài liệu và `scripts/upload_modal_volume.py`: `volume list`, `create`, `put --force`, `ls`, `get`. Trên máy restore, kiểm tra đúng interpreter và help trước khi upload:

```powershell
python -m modal --version
python -m modal volume --help
python -m modal volume put --help
python -m modal volume get --help
```

Nếu project dùng interpreter Conda, từ **repository root**:

```powershell
.\.conda\python.exe -m modal --version
.\.conda\python.exe -m modal volume --help
```

Từ **thư mục bundle** ở vị trí mặc định, interpreter đó là:

```powershell
..\..\.conda\python.exe -m modal --version
```

Dùng cùng một interpreter cho validator và các lệnh Modal; có thể thay `python` trong các ví dụ bằng đường dẫn Conda tương ứng. Nếu chưa có Modal, cài bằng `python -m pip install modal`; nếu chưa đăng nhập, dùng `python -m modal setup`. Chọn đúng Modal workspace/profile và environment dùng cho ứng dụng. Các lệnh mặc định dùng environment đang cấu hình; nếu chọn environment khác, đặt `MODAL_ENVIRONMENT` nhất quán hoặc thêm `--env TEN_ENV` cho từng lệnh Volume.

## 4. Validate local trước khi push

Khuyên chạy validator trước mọi lần upload. Từ **repository root**:

```powershell
python scripts/validate_artifact_bundle.py artifacts/vn_history_deployment
if ($LASTEXITCODE -ne 0) { throw "Bundle khong hop le; dung upload." }
```

Hoặc từ **thư mục bundle** ở vị trí mặc định:

```powershell
python ..\..\scripts\validate_artifact_bundle.py .
if ($LASTEXITCODE -ne 0) { throw "Bundle khong hop le; dung upload." }
```

Kết quả thành công có `ARTIFACT_BUNDLE_VALID`. Validator chỉ kiểm tra local, không chạy Modal hay inference. Nếu báo hash/lock/metadata không khớp, kiểm tra lại bản tải từ Drive; không tự sửa hash trong lock để bỏ qua lỗi.

## 5. Tên Volume và tạo Volume

`modal_app.py` hiện dùng `modal.Volume.from_name("vn-history-artifacts", create_if_missing=False)` và mount Volume vào `/artifacts`. Vì vậy hãy dùng đúng tên **`vn-history-artifacts`** và tạo nó nếu chưa có:

```powershell
python -m modal volume list
# Chi chay dong create neu danh sach chua co vn-history-artifacts:
python -m modal volume create vn-history-artifacts
```

Nếu Volume đã tồn tại thì bỏ qua `create`; không xoá Volume cũ để xử lý lỗi “đã tồn tại”. `--force` ở bước upload chỉ ghi đè file trùng đường dẫn, không xoá toàn bộ Volume hoặc các file cũ không có trong backup. Khi thay bundle đang phục vụ, bố trí dừng ứng dụng đọc bundle trong lúc cập nhật, chỉ chạy lại sau khi verify; nhiều lệnh upload không tạo một giao dịch nguyên tử.

Không đổi tên Volume nếu không cần. Nếu dùng tên khác, phải đổi tên trong `modal_app.py`, các helper có hardcode tên Volume và mọi lệnh CLI liên quan. Hiện tên Volume được ghi cứng trong `modal_app.py`; chỉ thêm một biến vào `.env` sẽ không đổi tên đó, trừ khi bạn đồng thời sửa mã để đọc biến môi trường.

## 6. Upload toàn bộ bundle, giữ đúng remote path

Chạy từ **thư mục `vn_history_deployment`**. Remote path tính từ root của Volume, không có tiền tố `/artifacts`:

| Local | Remote trong Volume | Path khi ứng dụng chạy |
| --- | --- | --- |
| `.\adapters` | `/adapters` | `/artifacts/adapters` |
| `.\config` | `/config` | `/artifacts/config` |
| `.\corpus` | `/corpus` | `/artifacts/corpus` |
| `.\retrieval` | `/retrieval` | `/artifacts/retrieval` |
| `.\manifest.json` | `/manifest.json` | `/artifacts/manifest.json` |
| `.\artifact_lock.json` | `/artifact_lock.json` | `/artifacts/artifact_lock.json` |

**Không thêm dấu `/` cuối remote directory** trong các lệnh dưới đây: CLI coi remote kết thúc bằng `/` là thư mục cha và nối thêm tên local; `/adapters/` có thể thành `/adapters/adapters`. Không tạo thêm tầng `/vn_history_deployment` hoặc `/artifacts` bên trong Volume.

`--force` là cú pháp overwrite đang được script upload của project sử dụng. Nếu bất kỳ lệnh nào lỗi, dừng và xử lý lỗi trước khi upload manifest/lock. Mục **Quick restore** có kiểm tra exit code tự động.

```powershell
& {
    # 1. Du lieu, adapter va config truoc.
    python -m modal volume put --force vn-history-artifacts .\adapters /adapters
    if ($LASTEXITCODE -ne 0) { throw "Upload adapters that bai." }
    python -m modal volume put --force vn-history-artifacts .\config /config
    if ($LASTEXITCODE -ne 0) { throw "Upload config that bai." }
    python -m modal volume put --force vn-history-artifacts .\corpus /corpus
    if ($LASTEXITCODE -ne 0) { throw "Upload corpus that bai." }
    python -m modal volume put --force vn-history-artifacts .\retrieval /retrieval
    if ($LASTEXITCODE -ne 0) { throw "Upload retrieval that bai." }

    # Marker va README cung duoc giu de mang du bundle.
    python -m modal volume put --force vn-history-artifacts .\EXPORT_SUCCESS.txt /EXPORT_SUCCESS.txt
    if ($LASTEXITCODE -ne 0) { throw "Upload marker that bai." }
    python -m modal volume put --force vn-history-artifacts .\README_MODAL_VOLUME.md /README_MODAL_VOLUME.md
    if ($LASTEXITCODE -ne 0) { throw "Upload README that bai." }

    # 2. Manifest sau khi cac muc tren thanh cong.
    python -m modal volume put --force vn-history-artifacts .\manifest.json /manifest.json
    if ($LASTEXITCODE -ne 0) { throw "Upload manifest that bai; khong upload lock." }

    # 3. LOCK LAST: day la lenh upload CUOI CUNG.
    python -m modal volume put --force vn-history-artifacts .\artifact_lock.json /artifact_lock.json
    if ($LASTEXITCODE -ne 0) { throw "Upload lock that bai; chua hoan tat restore." }
}
```

Lock mới chỉ được công bố sau dữ liệu/config và manifest tương ứng. Không upload lock trước manifest và không xoá lock cũ để “chuẩn bị” upload. Nếu sau này bundle có thêm file/thư mục ngoài cây hiện tại, upload chúng trước manifest/lock nữa.

## 7. Verify sau khi upload

Liệt kê root và các thư mục chính:

```powershell
python -m modal volume ls vn-history-artifacts /
python -m modal volume ls vn-history-artifacts /config
python -m modal volume ls vn-history-artifacts /adapters
python -m modal volume ls vn-history-artifacts /corpus
python -m modal volume ls vn-history-artifacts /retrieval
```

Root phải có `manifest.json`, `artifact_lock.json` cùng các thư mục phía trên. Kiểm tra nội dung hai file bằng `get`; đích `-` nghĩa là in ra stdout:

```powershell
python -m modal volume get vn-history-artifacts /manifest.json -
python -m modal volume get vn-history-artifacts /artifact_lock.json -
```

`deployment_id` trong manifest phải trùng lock và bản local. Chỉ liệt kê file hoặc đọc lock **chưa xác minh hash của dữ liệu trên Volume**. Để kiểm tra đầy đủ sau khi chuyển máy, có thể tải toàn bộ Volume về một thư mục tạm mới rồi chạy validator local trên bản vừa tải; cần đủ dung lượng và thời gian tải:

```powershell
# Chay tu thu muc bundle, repository o ..\..; sua duong dan validator neu can.
& {
    $bundleVerifyDir = Join-Path ([System.IO.Path]::GetTempPath()) ("vn-history-verify-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $bundleVerifyDir -ErrorAction Stop | Out-Null
    python -m modal volume get vn-history-artifacts / $bundleVerifyDir
    if ($LASTEXITCODE -ne 0) { throw "Tai ban verify that bai." }
    python ..\..\scripts\validate_artifact_bundle.py $bundleVerifyDir
    if ($LASTEXITCODE -ne 0) { throw "Artifact tren Volume khong qua validator." }
    Write-Host "Ban verify duoc giu tai: $bundleVerifyDir"
}
```

Không cần chạy `modal run`, deploy hay inference để thực hiện cách verify này. Helper `scripts/modal_artifact_sanity.py` hiện còn một số kiểm tra legacy (trong đó có `/adapters/central`); không dùng riêng yêu cầu legacy đó để kết luận Central V2 base bắt buộc adapter V1.

## 8. Trạng thái Central V2 hiện tại

- Central có thể chạy **`Qwen/Qwen3-8B` base**, adapter Central là **optional**.
- Bundle hiện có `central.adapter_path=null` trong manifest/registry và `llm.central.adapter_path=null` trong inference config; lock ghi `central=null`, `central_adapter_present=false`.
- `modal_app.py` hiện đặt `CENTRAL_AGENT_ADAPTER_PATH` thành chuỗi rỗng, được cấu hình ứng dụng đọc là `None`; mặc định `central_adapter_loaded=false`.
- `/adapters/central` nếu còn trong backup là **Central V1 legacy baseline**. Giữ nó để backup là được; không cần xoá weights cũ. Sự hiện diện của thư mục này không được tự động kích hoạt adapter khi config là `None`.
- Central V2 vẫn dùng luồng PREPARE → INITIAL_GROUNDING → ACTION / TOOL_EXECUTION khi cần → SYNTHESIS → QUALITY_REPAIR tùy chọn → FINAL; tách biệt ba role agent 4B.

## 9. Sau khi train Central V2 adapter

Adapter mới cho `Qwen/Qwen3-8B` nên có local path `adapters/central-v2/` trong bundle và remote path **`/adapters/central-v2`**, không ghi đè Central V1 legacy.

Trước khi push bản mới, export lại một bundle đồng bộ bằng công cụ export của project: `training/scripts/export_artifacts.py` hỗ trợ `--central-agent` và `--central-adapter-relative-path adapters/central-v2`. Config, registry, manifest, `deployment_id` và integrity lock phải cùng phản ánh adapter mới. Không chỉ chép weights mới rồi giữ lock cũ; chạy lại validator trước upload.

Ví dụ khi đứng trong bundle đã export/validate, lệnh upload adapter mới là:

```powershell
python -m modal volume put --force vn-history-artifacts .\adapters\central-v2 /adapters/central-v2
```

Nếu thư mục đầu vào ở ngay `.\central-v2`, có thể dùng `python -m modal volume put --force vn-history-artifacts .\central-v2 /adapters/central-v2`; để backup đầy đủ, bản bundle cuối cùng vẫn cần chứa adapter trong `adapters/central-v2/`.

Sau đó upload config/dữ liệu của bundle mới theo mục 6, `manifest.json` rồi `artifact_lock.json` **LAST**. Đặt runtime config:

```text
CENTRAL_AGENT_ADAPTER_PATH=/artifacts/adapters/central-v2
```

**Lưu ý cấu hình hiện tại:** phải sửa giá trị `"CENTRAL_AGENT_ADAPTER_PATH": ""` trong `image.env(...)` của `modal_app.py` sang path trên, hoặc sửa nó để đọc env tương ứng. Chỉ gán `$env:CENTRAL_AGENT_ADAPTER_PATH` ở PowerShell hoặc sửa `.env` local sẽ không thay thế giá trị đang ghi cứng trong image. Sau khi verify bundle và đổi runtime config, redeploy/restart ứng dụng theo quy trình project. **Không cần refactor lại kiến trúc Central.**

## 10. Artifact Volume, HF cache và secrets

`vn-history-artifacts` là **artifact Volume**, mount tại `/artifacts`. Nó không phải Hugging Face model cache Volume. Project hiện dùng Volume riêng `vn-history-hf-cache`, mount tại `/hf-cache`, với hub cache ở `/hf-cache/hub`. **Không upload Qwen base model weights/cache vào `vn-history-artifacts`.**

Không lưu API key, Tavily key, token hay runtime secrets trong bundle hoặc bản ZIP backup; không copy `.env` chứa secrets vào đây. Cấp secrets qua Modal Secret / environment của runtime. `modal_app.py` có hook `MODAL_WEB_SEARCH_SECRET_NAME` để chọn Modal Secret; README không yêu cầu ghi giá trị key vào artifact config.

## 11. Backup lên Google Drive

Có thể upload nguyên **`vn_history_deployment/`** hoặc ZIP toàn bộ thư mục rồi đưa lên Google Drive. Đảm bảo công cụ nén hỗ trợ kích thước thực tế của bundle và tải lên hoàn tất.

Khi restore phải giữ nguyên cấu trúc folder, đầy đủ adapters/config/corpus/retrieval, manifest, lock và các file đi kèm. Không chỉ copy riêng `manifest.json` hoặc `artifact_lock.json`. Giữ mã nguồn đúng phiên bản ở nơi backup riêng để có validator và cấu hình deploy tương ứng. Sau khi tải về, chạy validator trước khi push lên Volume.

## Quick restore

Khối dưới đây giả định bundle ở `repo/artifacts/vn_history_deployment`, repository/dependency/Modal account đã sẵn sàng, và đã chọn đúng Modal environment. **Thay đường dẫn ở dòng đầu**; nếu bundle đặt ngoài repo, sửa `$bundleRepo`. Khối tự chọn `.conda\python.exe` của repo nếu có, nếu không dùng `python` trên PATH. Khi thay Volume đang chạy, thực hiện trong thời gian ứng dụng ngừng đọc bundle.

```powershell
& {
    $ErrorActionPreference = "Stop"
    Set-Location "C:\Users\PC\Chatbot_answering_vietnamese_history\artifacts\vn_history_deployment"
    $bundleRoot = (Get-Location).Path
    $bundleRepo = (Resolve-Path "..\..").Path
    $bundlePython = Join-Path $bundleRepo ".conda\python.exe"
    if (-not (Test-Path -LiteralPath $bundlePython)) { $bundlePython = (Get-Command python -ErrorAction Stop).Source }
    $bundleVolume = "vn-history-artifacts"
    function Invoke-BundlePython {
        & $bundlePython @args
        if ($LASTEXITCODE -ne 0) { throw "Lenh Python/Modal that bai; dung truoc buoc tiep theo." }
    }

    # Validate local truoc khi tao/upload Volume.
    Invoke-BundlePython (Join-Path $bundleRepo "scripts\validate_artifact_bundle.py") $bundleRoot
    $bundleInventoryJson = Invoke-BundlePython -m modal volume list --json
    $bundleVolumes = ($bundleInventoryJson | Out-String) | ConvertFrom-Json
    if (-not ($bundleVolumes | Where-Object { $_.name -eq $bundleVolume })) {
        Invoke-BundlePython -m modal volume create $bundleVolume
    }
    Invoke-BundlePython -m modal volume list

    Invoke-BundlePython -m modal volume put --force $bundleVolume .\adapters /adapters
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\config /config
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\corpus /corpus
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\retrieval /retrieval
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\EXPORT_SUCCESS.txt /EXPORT_SUCCESS.txt
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\README_MODAL_VOLUME.md /README_MODAL_VOLUME.md
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\manifest.json /manifest.json
    # LOCK LAST: khong con lenh upload nao sau dong nay.
    Invoke-BundlePython -m modal volume put --force $bundleVolume .\artifact_lock.json /artifact_lock.json

    foreach ($bundlePath in @("/", "/config", "/adapters", "/corpus", "/retrieval")) {
        Invoke-BundlePython -m modal volume ls $bundleVolume $bundlePath
    }
    Invoke-BundlePython -m modal volume get $bundleVolume /manifest.json -
    Invoke-BundlePython -m modal volume get $bundleVolume /artifact_lock.json -
}
```

Sau quick restore, đối chiếu `deployment_id`; dùng bước tải về và validate ở mục 7 nếu cần kiểm tra hash đầy đủ của artifact trên Volume.
