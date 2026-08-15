# Training and Dataset Pipeline

[Về README gốc](../README.md)

`Training/` lưu notebook archives, context packs cho RAG-SFT và dependency của pipeline nghiên
cứu Phase 1-10. Runtime production không import code từ notebook; backend chỉ đọc deployment
bundle đã export.

## Cấu trúc hiện tại

```text
Training/
├── README.md
├── InvestigatingDataset.zip
├── requirement.txt
├── Dataset/
│   ├── Chunk_id/                  # 31 JSONL packs
│   └── merged_jsonl/
│       └── all_chunk_id.jsonl     # 520 records
└── Training/
    └── Training.zip               # 10 notebooks Phase 1-10
```

`all_chunk_id.jsonl` hiện có 520 records và 511 `chunk_id` khác nhau. Đây là tập
context/sample phục vụ RAG-SFT, không phải deployment corpus.

## Notebook audit

`InvestigatingDataset.zip` chứa:

| Notebook | Mục đích |
|---|---|
| `InvestigatingDataset.ipynb` | Khảo sát phân bố, schema và chất lượng dataset |
| `ReadingJSONCorpus.ipynb` | Đọc, tìm kiếm và audit JSON/JSONL corpus |

## Pipeline Phase 1-10

`Training/Training/Training.zip` chứa đúng các notebook sau. Tên được giữ nguyên theo archive,
kể cả chính tả hiện tại.

| Phase | Notebook | Output chính |
|---|---|---|
| 1 | `Phase1_SFT.ipynb` | QLoRA adapter SFT nền |
| 2 | `Phase2_CreatingCorpus.ipynb` | Corpus lịch sử đã lọc/làm sạch |
| 3 | `Phase3_ChunkExporter.ipynb` | Chunk packs theo chủ đề |
| 4 | `Phase4_ChunkExporter_extra_topics.ipynb` | Chunk packs cho chủ đề bổ sung |
| 5 | `Phase5.ipynb` | Dữ liệu chunk/messages đã hợp nhất |
| 6 | `Phase6_RAG_SFT_Qwen2_5_LoRA.ipynb` | RAG-grounded QLoRA adapter và evaluation |
| 7 | `Phase7_InferneceTesting.ipynb` | Inference/FAISS sanity checks |
| 8 | `Phase8_VN_History_Chunk_Metadata_Enrichment_v4.ipynb` | Corpus được làm giàu metadata |
| 9 | `Phase9_VN_History_Hybrid_RAG_ToolUse_v2_Grounded_Direct.ipynb` | FAISS, BM25S, config và benchmark |
| 10 | `Phase10_VN_History_FastAPI_Export_From_Phase9 (1).ipynb` | Merged model và deployment bundle |

## Dòng dữ liệu và model

```text
1.29M Vietnamese Wikipedia documents
  -> corpus cleaning/filtering
  -> topic chunks và RAG-SFT samples
  -> Phase 1 instruction QLoRA
  -> Phase 6 RAG-grounded QLoRA
  -> Phase 8 metadata enrichment
  -> Phase 9 Hybrid RAG indexes + benchmark
  -> Phase 10 merged model + deployment bundle
```

Adapter flow:

```text
Qwen2.5-3B-Instruct
  + Phase 1 adapter
  -> merged Stage 1 model
  + Phase 6 adapter
  -> merged Stage 1 + Stage 2 model
```

## Các bộ dữ liệu dễ nhầm

| Dữ liệu | Quy mô hiện tại | Vai trò |
|---|---:|---|
| `Training/Dataset/merged_jsonl/all_chunk_id.jsonl` | 520 records, 511 unique `chunk_id` | Context/sample corpus cho RAG-SFT |
| [`../Dataset/merged_jsonl/all_messages.jsonl`](../Dataset/README.md) | 1.000 messages | Chat-format training samples |
| Deployment corpus Phase 8-10 | 58.603 chunks | Global corpus cho FastAPI Hybrid RAG |

Không build production FAISS/BM25S từ nhầm tập 520 records. Deployment corpus, FAISS vectors và
BM25S records phải cùng count và dùng cùng `chunk_id` contract.

## Môi trường notebook

Dùng environment riêng cho training:

```powershell
python -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Training/requirement.txt
python -m ipykernel install --user --name vn-history-training
```

`requirement.txt` là environment nghiên cứu, không phải dependency lock production. Nó gồm
Jupyter, Hugging Face training stack, PEFT/TRL, bitsandbytes, sentence-transformers và FAISS.
QLoRA/bitsandbytes cần GPU, CUDA và PyTorch tương thích; notebook có thể yêu cầu package khác
tùy Colab/Modal runtime.

Giải nén archive vào workspace tạm bên ngoài repository trước khi chạy:

```powershell
Expand-Archive Training/InvestigatingDataset.zip -DestinationPath .work/investigation
Expand-Archive Training/Training/Training.zip -DestinationPath .work/phases
```

Không commit notebook checkpoints, cache, adapters, merged weights hoặc output lớn sau khi giải
nén.

## Output contract

Phase 9 tạo:

- enriched retrieval config;
- multilingual E5/FAISS index;
- BM25S index;
- benchmark results/summary;
- inference config dùng tiếp ở Phase 10.

Phase 10 tạo:

- merged Qwen2.5 Stage 1 + Stage 2 model;
- corpus và retrieval indexes;
- `config/inference_config.json`;
- evaluation outputs;
- `manifest.json` và `EXPORT_SUCCESS.txt`.

Bundle cuối được mô tả tại [`../artifacts/README.md`](../artifacts/README.md). Toàn bộ số liệu
Phase 6 và benchmark Phase 9 được ghi trong [README gốc](../README.md).

## Reproducibility checklist

- Ghi seed, base model revision, GPU, CUDA, package versions và dataset hash cho mỗi run.
- Lưu train/eval split và kiểm tra leakage theo question/evidence.
- Validate JSONL, role order, source IDs và evidence trước khi train.
- Không coi sample `id` trong [`../Dataset/`](../Dataset/README.md) là unique toàn cục.
- Không ghi đè checkpoint tốt nhất trước khi lưu metrics và config.
- Lưu benchmark output cạnh đúng retrieval/generation config đã dùng.
- So sánh corpus count, FAISS count, BM25S count và unique `chunk_id` trước khi export.
- Khi artifact contract thay đổi, cập nhật Phase 10, backend loader, sanity scripts và docs.
