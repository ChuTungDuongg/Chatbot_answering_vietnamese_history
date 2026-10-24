# Qdrant Cloud to local Docker migration (Windows PowerShell)

This standalone tool copies existing points; it never generates embeddings or
imports the FastAPI application. The repository's current default collection is
`vn_history_v1_e5` and its baseline named dense vector is `dense_e5`. Actual Cloud
schema is authoritative: unnamed/named dense, sparse and multivectors are copied.

The existing `scripts/retrieval/qdrant_index.py` is an embedding-stream sink with
a fixed E5 schema and restricted payload fields. It is deliberately not used to
export existing points, because that would lose arbitrary vectors and metadata.
The migration instead uses its existing `qdrant-client==1.19.1` dependency.

`compose.day03.yaml`, `qdrant_data`, application settings and `APP_MODE=api-only`
remain unchanged. The migrated collection is available through Qdrant REST; this
does not switch the application's retrieval backend or enable local inference.

## Design and safety

- Cloud access uses a read-only wrapper; use a Cloud **read-only API key** where
  possible. Only schema reads, exact count, scroll and similarity queries run
  against Cloud. No Cloud writes, deletes or snapshots are issued.
- Destination must be a loopback URL. A **new** collection is required;
  `vinai_day03_lab` is always protected. Existing collections are never adopted,
  replaced, recreated or deleted.
- Resume requires the original checkpoint and its matching random ownership token
  in destination collection metadata (Qdrant >=1.17). The token is an ownership
  guard, not an authentication credential. Collection metadata and all original
  payloads are retained; the ownership entry is added at collection level only.
- Raw REST preserves collection/index settings that a typed client could otherwise
  omit. Unsupported top-level settings and custom sharding block migration.
  Requested options are checked against the server's resulting configuration
  before points are copied. Incompatible settings fail explicitly.
  `destination_configuration_differences` identifies settings the target did not
  reproduce, including unexpected vector families or enabled quantization.
- Cloud replication/write-consistency and distributed read tuning may need a
  single-node adjustment. Such adjustments appear in the report and require
  `--allow-config-adjustments`. That flag does not permit unknown settings or
  vector/index changes.
- A process lock protects the checkpoint. Each acknowledged `wait=True` upsert is
  followed by a flushed checkpoint replacement. If the process stops after a
  write but before checkpointing, `--resume` replays the batch idempotently.
- Memory is bounded by batch size plus a fixed-size sample. Checkpoints store
  URLs without credentials, collection names, IDs used as pagination offsets,
  counters, ownership and content hashes; no point vectors, payloads or API keys.
  Generated reports/checkpoints live under the already Git-ignored `reports/`.

Arrange a quiet source window: pause other writers operationally while migrating
and verifying. Scroll is not a transactionally frozen Cloud snapshot. The tool
detects count/schema changes and compares a full content digest from the transfer
with a fresh source scan, including same-count changes to already copied points.
Keep the target exclusively for this migration until validation passes.

## Setup

Run from the repository root. A small separate environment avoids installing
Torch or the whole RAG stack just to copy Qdrant points:

```powershell
py -3 -m venv .venv\qdrant-migration
$python = Join-Path $PWD '.venv\qdrant-migration\Scripts\python.exe'
& $python -m pip install 'qdrant-client==1.19.1' pytest

# Start the existing Qdrant service only if needed; keep the named volume.
docker compose -f compose.day03.yaml up -d qdrant
Invoke-RestMethod 'http://127.0.0.1:6333/'
Invoke-RestMethod 'http://127.0.0.1:6333/collections'

$env:QDRANT_CLOUD_URL = 'https://YOUR-CLUSTER.REGION.cloud.qdrant.io:6333'
```

Use the actual Cloud REST endpoint from your dashboard, including its port.
Credentials embedded in URLs, URL queries and plaintext Cloud HTTP are rejected.
Leave `QDRANT_CLOUD_API_KEY` unset to enter it through Python `getpass`, which
does not echo it. A secret manager may supply that variable to the current process
instead. Never put the API key in a command, script, report, Git file or chat.
The tool does not read credentials from `.env` or reuse `QDRANT_API_KEY`.
If local Qdrant needs authentication, provide `QDRANT_LOCAL_API_KEY` securely.

## 1. Discover and dry-run

```powershell
& $python scripts/migrate_qdrant_cloud_to_local.py --dry-run
```

Collection discovery follows `QDRANT_COLLECTION` in the process, then the
repository's `.env` collection setting, then the literal default in
`app/config.py`. It chooses that name only when the collection exists in Cloud.
Otherwise it lists available names and requires explicit selection (or typing the
exact name in an interactive terminal). It never silently picks the first
collection. For subsequent commands, set explicit source and destination names:

```powershell
$migrationArgs = @(
    'scripts/migrate_qdrant_cloud_to_local.py'
    '--source-collection', 'vn_history_v1_e5'
    '--target-collection', 'vn_history_v1_e5__cloud_copy'
    '--checkpoint', '.\reports\qdrant-migrations\history.checkpoint.json'
    '--batch-size', '128'
)

& $python @migrationArgs --dry-run `
    --report '.\reports\qdrant-migrations\history.dry-run.json'
if ($LASTEXITCODE -ne 0) { throw 'Dry-run failed. Inspect the report before continuing.' }
```

Dry-run checks both connections, exact source count, full vector settings, payload
indexes, configuration compatibility, destination availability and sample-based
transfer/memory estimates. It writes only its local report, with **no database
modifications and no checkpoint**. Estimates exclude HNSW, WAL, filesystem and
Python overhead; leave substantial disk/RAM headroom. Full validation makes a
second complete source pass, so account for Cloud reads and network transfer.

If the report lists necessary topology adjustments, inspect each one first. If
acceptable for this local server, append the explicit flag and repeat dry-run:

```powershell
$migrationArgs += '--allow-config-adjustments'
& $python @migrationArgs --dry-run `
    --report '.\reports\qdrant-migrations\history.dry-run.json'
```

Use the same flag for migration, resume and verification. Unknown configurations,
custom sharding, incompatible vector settings and index options remain blockers.

## 2. Copy into the new collection

After dry-run passes and external source writes are paused:

```powershell
& $python @migrationArgs `
    --report '.\reports\qdrant-migrations\history.migration.json'
if ($LASTEXITCODE -ne 0) { throw 'Migration incomplete. Keep the checkpoint and inspect the report.' }
```

The tool copies original integer/UUID IDs, all vectors and entire payloads with
`scroll(with_vectors=True, with_payload=True)` and bounded, retried upserts. It
creates payload indexes before copying and reports progress after each batch.
Read/write batch limits and similarity-query limits are capped to the source's
enabled strict-mode constraints; the effective batch size appears in the estimate.
No existing collection or source point is altered. A failure leaves the new
collection and checkpoint intact for inspection/recovery.

## 3. Independent verification

```powershell
& $python @migrationArgs --verify-only `
    --report '.\reports\qdrant-migrations\history.verification.json'
if ($LASTEXITCODE -ne 0) { throw 'Verification failed. Inspect the PASS/FAIL checks.' }
Get-Content '.\reports\qdrant-migrations\history.verification.json'
```

The report gives PASS/FAIL for exact counts; vector names, dimensions and settings;
payload index types/options; deterministic sampled IDs, vectors and full payloads;
**every** source point versus a destination retrieve; source schema/count/content
stability; and exact similarity queries on both endpoints. Similarity compares
ranked scores because tied results can have different IDs. If the sample contains
no nonempty vector (including an empty collection), that query is explicitly
marked not applicable. `--sample-size` defaults to 16 and is bounded at 1000.

Payloads and IDs are compared exactly. Vector comparison allows float32 round-trip
noise (`rtol=1e-5`, `atol=1e-6`), including Qdrant cosine normalization; sparse
indices are exact. The tool neither embeds nor deliberately rounds vectors.
Verification performs no database writes. It needs the original checkpoint and
access to both endpoints; a missing checkpoint never authorizes adopting a target.

## Recovery from interruption

Pressing Ctrl+C or losing connectivity preserves the last saved position. In a
new PowerShell session, restore `$python`, `$env:QDRANT_CLOUD_URL` and the same
`$migrationArgs` values from above, then:

```powershell
& $python @migrationArgs --resume `
    --report '.\reports\qdrant-migrations\history.resume.json'
```

Authentication, ownership, schema and original exact count are checked again.
An ambiguous collection-creation timeout is recoverable when the server stored
the matching metadata token. A crash after upsert replays only the uncheckpointed
batch. Even a completed transfer is revalidated on `--resume`.

- Keep the checkpoint and collection metadata unchanged. Copy the checkpoint to
  safe storage if moving the operation to another machine, and keep the same
  endpoint identities and collection names.
- Locks are released by the operating system when a process exits. A leftover
  `.lock` file is harmless; do not remove a lock held by a running migration.
- A missing, malformed or mismatched checkpoint/marker is a blocker. Do not edit
  state to force adoption, clear metadata or overwrite an existing collection.
- If the source changed, or the partial destination was removed/replaced, choose
  a different new destination and a different checkpoint and repeat dry-run.
  Keep the old partial collection for inspection; the tool never deletes it.
- If an option/index was rejected or not reproduced, inspect the report and
  server compatibility. Do not remove an option from state to claim equivalence.
- Do not use `docker compose down -v`, `docker volume rm`, collection deletes or
  snapshot deletes. Normal container restart retains `qdrant_data`.

Exit codes: `0` = selected mode passed; `1` = migration/report checks failed or
were interrupted; `2` = setup/selection failed before migration; `130` = interrupt
before the migration runner started. Setup failures are printed with sanitized
messages; migration runs generate the requested report even on validation failure.

## Unit tests (no Cloud credentials)

```powershell
& $python -m pytest tests/test_qdrant_cloud_to_local.py -q
```

Tests mock the HTTP boundary while exercising qdrant-client serialization and the
real migration/checkpoint logic, using isolated temporary files. They do not
connect to Docker or Cloud.

API references: [Qdrant 1.17 collection creation and metadata](https://api.qdrant.tech/v-1-17-x/api-reference/collections/create-collection),
[collection configuration](https://api.qdrant.tech/v-1-17-x/api-reference/collections/get-collection),
[payload indexes](https://api.qdrant.tech/v-1-17-x/api-reference/indexes/create-field-index).
