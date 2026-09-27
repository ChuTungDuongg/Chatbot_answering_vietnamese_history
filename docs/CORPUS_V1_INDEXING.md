# Corpus V1 indexing readiness

Corpus V0 is the frozen historical retrieval baseline. Corpus V1 is a local
UVW-2026 full candidate whose indexing and evaluation are pending. The current
workspace has only `artifacts/corpus_v1/documents.jsonl` (926,690,133 bytes)
and `artifacts/corpus_v1/chunks.jsonl` (1,331,391,190 bytes). These generated
files and the user's `artifacts/old_corpus/` archive are ignored by Git.
Neither corpus quality nor superiority to V0 has been established.

The intended pipeline is UVW-2026 → history filter v3 → documents → 384-token
chunks with 48-token overlap → `chunks.jsonl` → multilingual E5 →
FAISS IndexFlatIP plus BM25S → existing Hybrid retrieval → human-reviewed
V0/V1 evaluation → promotion decision. This change prepares indexing only.

## Local artifact layout

```text
artifacts/corpus_v1/
  documents.jsonl
  chunks.jsonl
  manifest.json             # copy from full Colab build if available
  source_manifest.json      # copy if available
  hashes.json               # copy if available
  stats.json                # copy if available
  filter_audit.json         # copy if available
  samples/                  # optional; not needed for indexing
  retrieval/
    faiss/
      chunks.index
      manifest.json
    qdrant/
      manifest.json         # local proof for a remote or persistent collection
    bm25s_index/
      data.csc.index.npy
      indices.csc.index.npy
      indptr.csc.index.npy
      params.index.json
      vocab.index.json
      phase9_manifest.json
    index_manifest.json
```

The small provenance files were **not** present in this local copy at inspection.
Copy genuine files from the full Colab build when available; do not recreate them
from guesses. The index manifest records the corpus configuration fingerprint
only when the corpus `manifest.json` is present.

## Read-only preflight

Preflight scans JSONL one row at a time, validates IDs/text/schema, checks exact
duplicate IDs, computes corpus and ordered-ID SHA-256 hashes, and estimates
float32 FAISS vector storage at 768 dimensions. It does not create an output
directory or load/download an embedding model. Exact duplicate detection keeps
the unique ID set in RAM; it never retains the JSON rows or text corpus.
Successful scans report zero missing IDs, duplicate IDs, malformed rows, and
missing/empty text; invalid input aborts at the first affected line. Preflight
records the requested model revision, but deliberately does not contact the
model hub to resolve it. The dimension is an
estimate during preflight; the actual build reads and records the loaded model's
dimension.

PowerShell:

```powershell
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --device cpu --preflight
```

Colab/Linux:

```bash
python -m scripts.retrieval.build_index --corpus /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-full-v1/chunks.jsonl --output-dir /content/drive/MyDrive/vn_history_llm/corpus_v1/uvw-2026-full-v1/retrieval --embedding-model intfloat/multilingual-e5-base --device auto --preflight
```

Review the chunk count, corpus hash, ordered-ID hash, size, and estimated vector
bytes before allocating GPU or RAM. Flat FAISS holds every vector in system RAM;
streaming removes the separate full JSON/text/embedding-matrix peak, but cannot
make the final index constant-memory. Allow additional RAM for the model, each
batch, libraries, and the OS. BM25S has a separate and potentially larger
memory peak.

The local read-only preflight on 2026-09-27 counted **624,288** chunks, with
corpus SHA-256
`4d1700c20c25e8f2c77b801e6ef6242224a8f59a37469ab13fdbe1dc62266255`
and ordered chunk-ID SHA-256
`b42f792bc8b59795861a5329df5fc0835abdfd3449028dcf2eef8e86f3b55d8f`.
At 768 dimensions, raw float32 vectors require **1,917,812,736 bytes**
(1.786 GiB), before model or process overhead. The shortest nonempty chunk
was one character; **329 chunks have fewer than 40 non-whitespace
characters**. Sampled cases include reference headings and residual infobox
fragments. Review these before indexing; preflight does not change corpus bytes.

## Future component builds

These commands are for a **later** indexing job. `--component all` retains the
local FAISS+BM25S baseline. `dense-all` encodes each E5 batch once and sends
the identical float32 batch to FAISS and Qdrant; `all-backends` also builds
BM25S. Individual `faiss`, `qdrant`, and `bm25` modes remain available. Use a
fresh output directory. The builder refuses to overwrite an existing component
or Qdrant collection. It cannot resume an interrupted component.
The paired mode records one SHA-256 over the ordered float32 embedding stream
in both dense manifests; independent builds record their own stream hashes.

PowerShell, separate FAISS and BM25S phases:

```powershell
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --model-revision main --embedding-batch-size 64 --device cuda --component faiss
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --embedding-batch-size 64 --device cpu --component bm25
```

PowerShell, both components in one fresh output directory:

```powershell
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --model-revision main --embedding-batch-size 64 --device cuda --component all
```

PowerShell, both V1 dense backends with one embedding stream, then shared BM25S
in a fresh CPU session:

```powershell
$env:QDRANT_URL = "http://localhost:6333" # persistent local server; use a private Cloud URL when chosen
$env:QDRANT_COLLECTION = "vn_history_v1_e5"
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --model-revision main --embedding-batch-size 64 --device cuda --component dense-all
python -m scripts.retrieval.build_index --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --embedding-model intfloat/multilingual-e5-base --embedding-batch-size 64 --device cpu --component bm25
```

Set `QDRANT_API_KEY` privately in the environment for a secured server or
Qdrant Cloud. The builder does not accept it as a CLI argument or write it to
manifests/logs. A standalone Qdrant build uses `--component qdrant`. A fresh
one-job build of all three uses `--component all-backends`; BM25S may need more
system RAM than the GPU session provides. No real collection is created by
this readiness change.

Use `--device auto` when CUDA availability is uncertain. The FAISS phase
resolves the requested Hugging Face model revision to an immutable SHA and
loads that SHA with SentenceTransformer. It encodes
`passage: {title}\n{text}` in ordered batches, normalizes embeddings, and
adds float32 vectors to IndexFlatIP. Runtime queries still use `query: ...`.
The sidecar records model ID, requested/resolved revision, actual dimension,
normalization, literal passage/query prefixes, batch size, device, corpus
identity, builder Git SHA/version, and timestamp. The build resolves `main`
once, then loads the immutable SHA; for an explicitly pinned rebuild, pass
that SHA through `--model-revision`. Keep the same model and revision for
V0/V1 comparison.

BM25S 0.3.10 does not offer a bounded-memory incremental index writer. The
builder streams JSON rows and normalizes/tokenizes bounded batches, but retains
all token-ID lists and the sparse score arrays during `BM25.index`. It uses the
same `match_norm`, duplicated title, `bm25s.tokenize` with no stopwords or
stemmer, and BM25 defaults as the prior builder. Run this phase separately on
a high-RAM runtime if needed. No dense model or matrix is retained in that
phase. The manifest records BM25 configuration and the same ordered chunk hash.

Each local component is written to a `.partial` directory and renamed only
after its index and sidecar are complete. An interrupted Qdrant upload also
leaves a local `qdrant.partial` marker and may leave an incomplete remote
collection. Inspect and explicitly clean up that collection before retrying;
the builder never silently deletes or reuses it. A later run refuses leftover
partial or incomplete component paths. There is no shard checkpoint or
automatic resume for index generation. The Corpus V1 construction pipeline's
resume support does not apply to indexing.

## Two controlled retrieval lanes

The primary comparison changes **only the dense backend**:

| Lane | Dense search | Shared later stages |
| --- | --- | --- |
| A, local hybrid | normalized E5 → exact FAISS IndexFlatIP | BM25S → existing weighted RRF → existing BGE reranker → context selection |
| B, Qdrant hybrid | same E5 → Qdrant HNSW (`exact=false`) | same BM25S → same RRF, reranker, and context selection |

Qdrant uses named vector `dense_e5`, COSINE distance, full-precision vectors,
server-default HNSW parameters, and no quantization. The build manifest captures
effective `m`, `ef_construct`, and `full_scan_threshold` returned by the server,
the server/client versions, count, corpus hashes, and resolved E5 revision.
`point_id` is the zero-based `chunks.jsonl` row number. Thus FAISS row 12345,
Qdrant point 12345, and corpus row 12345 identify the same chunk. Qdrant stores
a compact payload containing IDs, title, URL, source split, filter decision,
relevance score, and token count. It omits full text; the ordered corpus remains
the text source. The runtime verifies returned `chunk_id` against its row map.

Use a persistent local Qdrant server or Qdrant Cloud for the full collection.
The Python client's in-process local mode is not used for the 624k-point build.
The local FAISS lane needs no running Qdrant service. Qdrant runtime selection
requires `QDRANT_URL`; `QDRANT_API_KEY` is optional and must stay private.
`QDRANT_COLLECTION` defaults to `vn_history_v1_e5` and `QDRANT_HNSW_EF` is an
explicit optional query parameter. When unset, server search defaults apply;
record its chosen value in each controlled evaluation. Exact diagnostic search
uses `exact=true` and is separate from the normal HNSW lane.

After all indexes exist, validate the real counts and corpus identity before
either hybrid evaluation:

```powershell
python -m scripts.retrieval.validate_indexes --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --components all-backends
```

### Evaluation levels

**Level 1, dense backend:** Run the same normalized `query: ` E5 vectors
through FAISS exact, Qdrant exact, and Qdrant HNSW. The diagnostic utility
reports exact top-k overlap and score difference, HNSW ANN Recall@10/20/50
against FAISS exact, and p50/p95 dense-search latency. Keep build time and
index/storage size alongside that report. Qdrant storage size must be measured
on its persistent volume or Cloud dashboard; the comparison utility leaves it
unset. Record whether Qdrant is local or remote because network transit affects
latency. Ties and floating-point differences
can prevent identical ordering; inspect exact-search mismatches rather than
assuming all are historical relevance errors.

```powershell
python -m scripts.retrieval.compare_dense --corpus artifacts/corpus_v1/chunks.jsonl --output-dir artifacts/corpus_v1/retrieval --queries path/to/fixed_questions.jsonl --hnsw-ef 128
```

**Level 2, end-to-end hybrid:** Run the same human-reviewed history questions
through two isolated app processes with the same V1 bundle, E5 revision,
BM25S index, RRF configuration, BGE reranker, candidate depths, query analysis,
and context selection. Set only `RETRIEVAL_DENSE_BACKEND=faiss` versus
`RETRIEVAL_DENSE_BACKEND=qdrant` (plus Qdrant connection settings). Save each
run's predictions and compare HitRate@1/3/5/10, MRR@10, nDCG@10, source recall
when labeled, reranked overlap, and retrieval latency. The existing
`evaluation.runner` scores saved predictions. ANN recall and historical
relevance metrics belong in separate reports. Do not promote a winner from
unlabeled fixture questions.

Qdrant-native sparse retrieval, Query API fusion, and `qdrant_native_hybrid`
remain a possible **third, later experiment**. They are absent from the two
primary lanes, which both use the same BM25S and weighted RRF.

## Runtime and V0 comparison

The current application still defaults `ARTIFACT_ROOT` to
`artifacts/vn_history_deployment` and expects the V0 `corpus/` and
`config/` layout. Its loader holds every parsed chunk row and an ID-to-row
dictionary in RAM, then loads FAISS and memory-mapped BM25S. Merely changing
`ARTIFACT_ROOT` to `artifacts/corpus_v1` will not work: its corpus path,
runtime manifest shape, and configuration still refer to V0. A 1.3 GB JSONL
would also occupy considerably more RAM as Python dictionaries and strings.

A future, isolated V1 deployment bundle must provide the expected
`corpus/vn_history_rag_chunks_enriched.jsonl` path (link or copy the V1 chunks
without changing their row order), `retrieval/bm25s_index/`, either
`retrieval/faiss/` or `retrieval/qdrant/manifest.json` for the selected lane,
a compatible `config/inference_config.json`, and a
runtime `manifest.json` with `corpus.count = 624288`. Validate its corpus hash,
ordered-ID hash, model ID/revision, and index counts before setting
`ARTIFACT_ROOT` to that bundle. The current loader accepts schema 2 rows but
holds all 624,288 parsed chunk objects plus an ID dictionary in RAM. This
packaging and memory check remain separate work; neither the V1 root nor the
current V0 runtime default is changed here.

For Lane A, set `RETRIEVAL_DENSE_BACKEND=faiss`; for Lane B, set
`RETRIEVAL_DENSE_BACKEND=qdrant` with `QDRANT_URL`, `QDRANT_COLLECTION`, and
optionally `QDRANT_API_KEY` and `QDRANT_HNSW_EF`. Both lanes point to the same
BM25S files in their isolated V1 bundles. The app loads the resolved E5 SHA
from the dense manifest where present. The runtime's older `faiss_ms` telemetry
field measures the selected dense call in either lane; interpret it as dense
latency for Lane B.

For an initial V1 smoke test, an isolated high-RAM adapter can load V1 rows
in memory while retaining the ordered-ID checks. A scalable runtime should
store compact chunk metadata keyed by stable FAISS/BM25 row number, with
deterministic offsets or a local SQLite sidecar, and fetch full text only for
retrieved candidates. The sidecar must bind to the corpus SHA-256 and ordered
chunk-ID SHA-256. This runtime work and a human-reviewed retrieval gold set
must precede any claim that V1 improves on V0. Do not change retrieval
weights, reranker, or query behavior in the comparison.

The user-managed local V0 archive is at `artifacts/old_corpus/`. The
preservation audit still checks the original V0 paths and correctly fails
while protected files are absent there. Restore a verified V0 working layout
for comparison without changing the preservation manifest or committing the
archive.
