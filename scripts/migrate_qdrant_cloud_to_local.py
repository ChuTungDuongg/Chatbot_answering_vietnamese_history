"""Read-only Cloud export to a new, owned local Qdrant collection.

Run with --help. No application imports, embedding generation, deletes, snapshots,
collection replacement, or production configuration changes are performed.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
import getpass
import hashlib
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit
import uuid
import warnings

import httpx
import portalocker
from qdrant_client import QdrantClient, models


ROOT = Path(__file__).resolve().parents[1]
MARKER = "_qdrant_cloud_to_local_migration"
FORMAT_VERSION = 1
EMPTY_DIGEST = "0" * 64
VALIDATION_CHECKS = (
    "point_count", "vector_dimensions_names", "payload_indexes",
    "sampled_point_ids", "sampled_vectors", "sampled_payloads",
    "all_points", "source_unchanged", "sample_similarity",
)


class MigrationError(RuntimeError):
    """Only fixed, credential-free messages belong in this exception."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def redact(value: Any, secrets: tuple[str, ...]) -> Any:
    serialized = json.dumps(value, ensure_ascii=True)
    for secret in secrets:
        if secret:
            # Replace JSON-escaped as well as ordinary spellings.
            serialized = serialized.replace(json.dumps(secret, ensure_ascii=True)[1:-1], "[REDACTED]")
    return json.loads(serialized)


def validate_endpoint_url(url: str, *, local: bool) -> str:
    try:
        parsed = urlsplit(url.strip())
        _ = parsed.port
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError
        if not parsed.hostname or parsed.path not in ("", "/"):
            raise ValueError
        if local:
            hostname = parsed.hostname.casefold()
            loopback = hostname == "localhost"
            if not loopback:
                loopback = ipaddress.ip_address(hostname).is_loopback
            if not loopback or parsed.scheme not in ("http", "https"):
                raise ValueError
        elif parsed.scheme != "https":
            raise ValueError
        return urlunsplit((parsed.scheme, parsed.netloc.lower(), "", "", ""))
    except (ValueError, AttributeError):
        raise MigrationError("Use a credential-free HTTPS Cloud URL and a loopback local URL, without a path, query or fragment.") from None


def cloud_api_key(environment: dict[str, str]) -> str:
    if environment.get("QDRANT_CLOUD_API_KEY"):
        return environment["QDRANT_CLOUD_API_KEY"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            value = getpass.getpass("Qdrant Cloud API key (hidden): ")
    except (EOFError, getpass.GetPassWarning):
        raise MigrationError("A secure terminal is required for getpass; run in PowerShell or supply QDRANT_CLOUD_API_KEY securely.") from None
    if not value:
        raise MigrationError("A Cloud API key is required.")
    return value


def validate_collection_name(name: str) -> None:
    if not name or name.strip() != name or any(c in name for c in ("/", "\\")) or any(ord(c) < 32 for c in name):
        raise MigrationError("Collection names must be nonempty and contain no path separators or control characters.")


def select_collection(available: list[str], explicit: str | None, repo: Path,
                      environment: dict[str, str], *, interactive: bool) -> str:
    """Read only the collection setting; never instantiate application Settings."""
    configured = environment.get("QDRANT_COLLECTION")
    if not configured and (repo / ".env").is_file():
        for line in (repo / ".env").read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.removeprefix("export ").partition("=")
            if separator and key.strip() == "QDRANT_COLLECTION":
                value = value.strip()
                if value.startswith(("'", '"')):
                    try:
                        configured = ast.literal_eval(value)
                    except (SyntaxError, ValueError):
                        configured = None
                else:
                    configured = value.split(" #", 1)[0].strip()
    if not configured and (repo / "app/config.py").is_file():
        tree = ast.parse((repo / "app/config.py").read_text(encoding="utf-8-sig"))
        candidates = {node.value.value for node in ast.walk(tree)
                      if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                      and node.target.id == "qdrant_collection" and isinstance(node.value, ast.Constant)
                      and isinstance(node.value.value, str)}
        if len(candidates) == 1:
            configured = candidates.pop()
    if explicit:
        if explicit not in available:
            raise MigrationError("The explicit source collection was not found. Use a name from the Cloud collection list.")
        return explicit
    if configured in available:
        return configured
    if interactive:
        selected = input("Type the exact Cloud collection name from the displayed list: ").strip()
        if selected in available:
            return selected
    raise MigrationError("Collection selection is ambiguous; specify --source-collection using the displayed Cloud collection list.")


class Endpoint:
    """REST only. The Cloud instance refuses all write operations."""

    def __init__(self, url: str, api_key: str | None = None, *, readonly: bool,
                 retries: int = 3, timeout: int = 60, transport=None,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print):
        self.url = validate_endpoint_url(url, local=not readonly)
        self.readonly, self.retries, self.sleep, self.log = readonly, retries, sleep, log
        self.secrets = (api_key,) if api_key else ()
        options = {"trust_env": False, "follow_redirects": False}
        if transport is not None:
            options["transport"] = transport
        self.rest = httpx.Client(base_url=self.url, headers={"api-key": api_key} if api_key else {}, timeout=timeout, **options)
        self.sdk = QdrantClient(url=self.url, port=None, api_key=api_key, prefer_grpc=False,
                                timeout=timeout, check_compatibility=False, **options)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.sdk.close()
        self.rest.close()

    def writable(self) -> None:
        if self.readonly:
            raise MigrationError("Cloud write operations are forbidden.")

    def invoke(self, operation: str, call: Callable, *, retry: bool = True):
        for attempt in range(self.retries + 1):
            try:
                return call()
            except MigrationError:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                inner = getattr(exc, "source", exc)
                transient = status in (408, 429) or (status is not None and status >= 500) or isinstance(inner, (httpx.TransportError, OSError))
                code = f"HTTP {status}" if status else type(inner).__name__
                if retry and transient and attempt < self.retries:
                    self.log(f"{operation}: retry {attempt + 1}/{self.retries} ({code})")
                    self.sleep(min(2 ** attempt, 15))
                    continue
                raise MigrationError(f"{operation} failed ({code}); no server error text is logged. Check connectivity, permissions and the checkpoint before resuming.") from None

    def request(self, method: str, path: str, body=None, *, optional=False, retry=True,
                before_attempt: Callable = lambda: None):
        if self.readonly and method != "GET":
            raise MigrationError("Cloud writes through the schema API are forbidden.")

        def call():
            before_attempt()
            response = self.rest.request(method, path, json=body)
            if optional and response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            return data if path == "/" else data["result"]

        return self.invoke("Cloud REST" if self.readonly else "Local REST", call, retry=retry)

    @staticmethod
    def collection_path(name: str) -> str:
        validate_collection_name(name)
        return "/collections/" + quote(name, safe="")

    def collection(self, name: str, *, optional=False):
        return self.request("GET", self.collection_path(name), optional=optional)

    def collections(self) -> list[str]:
        return [item["name"] for item in self.request("GET", "/collections")["collections"]]

    def count(self, name: str) -> int:
        return int(self.invoke("Exact count", lambda: self.sdk.count(name, exact=True)).count)

    def scroll(self, name: str, limit: int, offset=None):
        return self.invoke("Scroll", lambda: self.sdk.scroll(collection_name=name, limit=limit,
                           offset=offset, with_vectors=True, with_payload=True))

    def retrieve(self, name: str, ids: list):
        return self.invoke("Retrieve", lambda: self.sdk.retrieve(collection_name=name, ids=ids,
                           with_vectors=True, with_payload=True))

    def upsert(self, name: str, records: list, *, before_attempt: Callable = lambda: None):
        self.writable()
        points = [models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in records]

        def call():
            before_attempt()
            return self.sdk.upsert(collection_name=name, points=points, wait=True)

        result = self.invoke("Upsert", call)
        if str(result.status).split(".")[-1].lower() != "completed":
            raise MigrationError("Upsert was not confirmed completed; the checkpoint has not advanced.")

    def query(self, name: str, vector, using: str | None, limit: int):
        return self.invoke("Similarity query", lambda: self.sdk.query_points(
            collection_name=name, query=vector, using=using, limit=limit,
            search_params=models.SearchParams(exact=True, quantization=models.QuantizationSearchParams(ignore=True)),
            with_vectors=False, with_payload=False)).points


def nonnull(value):
    if isinstance(value, dict):
        return {k: nonnull(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [nonnull(v) for v in value]
    return value


def subset(expected: Any, actual: Any) -> bool:
    """Compare every requested option; tolerate additional server defaults."""
    expected, actual = nonnull(expected), nonnull(actual)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and subset(v, actual[k]) for k, v in expected.items())
    return expected == actual


def configuration_plan(info: dict, allow_adjustments: bool) -> tuple[dict, dict, list, list]:
    config = deepcopy(info["config"])
    params = config.pop("params")
    body, issues, adjustments = {}, [], []
    fields = {"vectors", "sparse_vectors", "shard_number", "sharding_method", "replication_factor", "write_consistency_factor", "on_disk_payload", "payload"}
    for key, value in params.items():
        if value is None:
            continue
        if key in fields:
            body[key] = value
        elif key in ("read_fan_out_factor", "read_fan_out_delay_ms"):
            adjustments.append(f"params.{key}: distributed read tuning has no create-collection equivalent on this local server.")
        else:
            issues.append(f"Unsupported collection parameter: {key}")
    for key, value in config.items():
        if value is None:
            continue
        if key in ("hnsw_config", "wal_config", "quantization_config", "strict_mode_config", "metadata"):
            body[key] = value
        elif key == "optimizer_config":
            body["optimizers_config"] = value
        else:
            issues.append(f"Unsupported collection configuration: {key}")
    if body.get("sharding_method") not in (None, "auto"):
        issues.append("Custom sharding cannot be reproduced by an unkeyed scroll/upsert migration; shard routing would be lost.")
    for key in ("replication_factor", "write_consistency_factor"):
        if body.get(key, 1) != 1:
            adjustments.append(f"{key}: {body[key]} -> 1 for a single-node destination.")
            body[key] = 1
    if adjustments and not allow_adjustments:
        issues.append("Configuration adjustments require explicit --allow-config-adjustments after inspecting --dry-run.")
    if MARKER in (body.get("metadata") or {}):
        issues.append("Source collection already contains the reserved migration ownership metadata key.")
    indexes = {}
    for name, schema in info.get("payload_schema", {}).items():
        unknown = set(schema) - {"data_type", "params", "points"}
        if unknown:
            issues.append(f"Unsupported payload index information for {name}: {', '.join(sorted(unknown))}")
        definition = schema.get("params") or {"type": schema["data_type"]}
        if definition.get("type") != schema["data_type"]:
            issues.append(f"Inconsistent payload index type for {name}")
        indexes[name] = definition
    return body, indexes, issues, adjustments


def point_data(point) -> dict:
    return point.model_dump(mode="json", include={"id", "vector", "payload"})


def extend_digest(previous: str, records: list) -> str:
    for point in records:
        previous = hashlib.sha256((previous + canonical(point_data(point))).encode()).hexdigest()
    return previous


def vectors_equal(left: Any, right: Any) -> bool:
    if hasattr(left, "model_dump"):
        left = left.model_dump(mode="json")
    if hasattr(right, "model_dump"):
        right = right.model_dump(mode="json")
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            (left[k] == right[k] if k == "indices" else vectors_equal(left[k], right[k])) for k in left)
    if isinstance(left, list):
        return isinstance(right, list) and len(left) == len(right) and all(vectors_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
    return left == right


def check_vector_shape(point, body: dict) -> bool:
    vectors = body.get("vectors") or {}
    unnamed = isinstance(vectors.get("size"), int)
    vector = point_data(point)["vector"]
    dense = {"": vectors} if unnamed else vectors
    sparse = body.get("sparse_vectors") or {}
    if vector is None:
        return not dense and not sparse
    values = {"": vector} if isinstance(vector, list) else vector
    if not isinstance(values, dict):
        return False
    for name, value in values.items():
        if name in sparse:
            if not isinstance(value, dict) or set(value) != {"indices", "values"}:
                return False
            if len(value["indices"]) != len(value["values"]) or any(not isinstance(i, int) or i < 0 for i in value["indices"]):
                return False
            if not all(math.isfinite(v) for v in value["values"]):
                return False
        elif name in dense:
            schema = dense[name]
            rows = value if schema.get("multivector_config") else [value]
            if not isinstance(rows, list) or not all(isinstance(row, list) and len(row) == schema["size"] and all(isinstance(v, (int, float)) and math.isfinite(v) for v in row) for row in rows):
                return False
        else:
            return False
    return True


def atomic_json(path: Path, value: dict, *, exclusive=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Options:
    source_collection: str
    target_collection: str
    checkpoint: Path
    report: Path
    batch_size: int = 128
    sample_size: int = 16
    dry_run: bool = False
    resume: bool = False
    verify_only: bool = False
    allow_config_adjustments: bool = False


class Migration:
    def __init__(self, source: Endpoint, target: Endpoint, options: Options, *, log=print):
        self.source, self.target, self.options, self.log = source, target, options, log
        self.secrets = source.secrets + target.secrets
        self.report = {"format_version": FORMAT_VERSION, "mode": "dry-run" if options.dry_run else "verify-only" if options.verify_only else "migration",
                       "source_collection": options.source_collection, "target_collection": options.target_collection,
                       "source_count": None, "destination_count": None, "checks": {},
                       "configuration_issues": [], "configuration_adjustments": []}
        self.state = None

    def check(self, name: str, passed: bool, detail: str, **extra):
        self.report["checks"][name] = {"status": "PASS" if passed else "FAIL", "detail": detail, **extra}

    def run(self) -> dict:
        try:
            self._run()
        except KeyboardInterrupt:
            self.check("execution", False, "Interrupted; rerun the same command with --resume. The last uncheckpointed batch is safe to replay.")
        except MigrationError as exc:
            self.check("execution", False, str(exc))
        except Exception as exc:
            self.check("execution", False, f"Stopped safely ({type(exc).__name__}); raw exception and data omitted. Keep the checkpoint and use --resume after resolving the cause.")
        if not self.options.dry_run:
            for name in VALIDATION_CHECKS:
                if name not in self.report["checks"]:
                    self.check(name, False, "Not performed; migration or verification did not complete.")
        self.report["overall"] = "FAIL" if any(check["status"] == "FAIL" for check in self.report["checks"].values()) else "PASS"
        self.report = redact(self.report, self.secrets)
        # Invalid output arguments must never destroy a checkpoint while trying
        # to report why those same arguments were rejected.
        if self.options.report.resolve() != self.options.checkpoint.resolve():
            atomic_json(self.options.report, self.report)
        return self.report

    def _run(self):
        opt = self.options
        validate_collection_name(opt.source_collection)
        validate_collection_name(opt.target_collection)
        if opt.target_collection.casefold() == "vinai_day03_lab":
            raise MigrationError("vinai_day03_lab is protected and cannot be a migration target.")
        if self.source.url == self.target.url or not self.source.readonly or self.target.readonly:
            raise MigrationError("Source must be read-only Cloud and destination must be a different writable local endpoint.")
        if not 1 <= opt.batch_size <= 10000 or not 1 <= opt.sample_size <= 1000:
            raise MigrationError("Use batch-size 1..10000 and sample-size 1..1000.")
        if opt.dry_run and (opt.resume or opt.verify_only):
            raise MigrationError("--dry-run cannot be combined with --resume or --verify-only.")
        if opt.checkpoint.resolve() == opt.report.resolve():
            raise MigrationError("Checkpoint and report paths must be different.")
        source_version = self.source.request("GET", "/").get("version", "unknown")
        target_version = self.target.request("GET", "/").get("version", "unknown")
        self.source.collections()
        self.target.collections()
        self.check("connectivity", True, "Both REST endpoints responded.")
        self.report["server_versions"] = {"source": source_version, "destination": target_version}
        if tuple(int(x) for x in target_version.split(".")[:2]) < (1, 17):
            raise MigrationError("Destination Qdrant >=1.17 is required for collection-metadata ownership and safe resume.")
        info = self.source.collection(opt.source_collection)
        count = self.source.count(opt.source_collection)
        self.report["source_count"] = count
        self.body, self.indexes, issues, adjustments = configuration_plan(info, opt.allow_config_adjustments)
        self.report["configuration_issues"] = issues
        self.report["configuration_adjustments"] = adjustments
        self.report["source_schema"] = {"vectors": info["config"]["params"].get("vectors"),
                                        "sparse_vectors": info["config"]["params"].get("sparse_vectors"),
                                        "settings": {k: v for k, v in self.body.items() if k != "metadata"}}
        self.report["payload_indexes"] = info.get("payload_schema", {})
        # Full raw settings are hashed, not persisted; unknown options cannot hide
        # in a typed SDK response that silently omitted them.
        self.fingerprint = digest({"config": info["config"], "indexes": self.indexes})
        strict = self.body.get("strict_mode_config") or {}
        self.query_limit = strict.get("max_query_limit") if strict.get("enabled") else None
        self.read_limit = min(opt.batch_size, self.query_limit or opt.batch_size)
        self.transfer_limit = min(self.read_limit, strict.get("upsert_max_batchsize") or opt.batch_size)
        sample, _ = self.source.scroll(opt.source_collection, min(opt.sample_size, self.read_limit))
        average = sum(len(canonical(point_data(point)).encode()) for point in sample) / len(sample) if sample else 0
        self.report["estimates"] = {"json_transfer_bytes": math.ceil(average * count),
                                    "batch_working_bytes": math.ceil(average * self.read_limit * 4),
                                    "effective_transfer_batch_size": self.transfer_limit,
                                    "transfer_batches": math.ceil(count / self.transfer_limit),
                                    "verification": "One additional complete source scroll and destination retrieve per batch.",
                                    "note": "Sample-based JSON/memory estimates; vector indexes, Python overhead, WAL and disk allocation are additional."}
        self.check("configuration", not issues, "All settings are reproducible." if not issues else "Resolve configuration_issues before copying.")
        if issues:
            raise MigrationError("Source configuration cannot be safely reproduced; see configuration_issues and configuration_adjustments.")
        existing = self.target.collection(opt.target_collection, optional=True)
        if opt.dry_run:
            self.check("target_safety", existing is None, "New target name is available." if existing is None else "Target exists; choose a new collection or use the matching checkpoint with --resume.")
            return
        opt.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            with portalocker.Lock(str(opt.checkpoint) + ".lock", mode="a", timeout=0):
                self.prepare_state(existing, count)
                if not opt.verify_only:
                    self.prepare_target()
                    self.transfer(count)
                self.validate(count)
                if not opt.verify_only and not any(item["status"] == "FAIL" for item in self.report["checks"].values()):
                    self.state["complete"] = True
                    self.save_state()
        except portalocker.exceptions.LockException:
            raise MigrationError("Another process holds this checkpoint lock; wait for it to exit before resuming.") from None

    def marker(self) -> dict:
        return {"format_version": FORMAT_VERSION, "migration_id": self.state["migration_id"],
                "identity": self.state["identity"], "schema_fingerprint": self.fingerprint}

    def save_state(self, *, exclusive=False):
        serialized = canonical(self.state)
        if any(secret in serialized for secret in self.secrets):
            raise MigrationError("Refusing to persist a credential in checkpoint state.")
        atomic_json(self.options.checkpoint, self.state, exclusive=exclusive)

    def prepare_state(self, existing, count):
        opt = self.options
        identity = {"source_url": self.source.url, "target_url": self.target.url,
                    "source_collection": opt.source_collection, "target_collection": opt.target_collection}
        if opt.checkpoint.exists():
            if not (opt.resume or opt.verify_only):
                raise MigrationError("Checkpoint exists; use --resume or --verify-only with the same source and target.")
            try:
                self.state = json.loads(opt.checkpoint.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                raise MigrationError("Checkpoint is unreadable; do not adopt an existing collection without its original checkpoint.") from None
            if self.state.get("format_version") != FORMAT_VERSION or self.state.get("identity") != identity:
                raise MigrationError("Checkpoint belongs to a different migration or format.")
            if self.state.get("schema_fingerprint") != self.fingerprint or self.state.get("source_count") != count:
                raise MigrationError("Source schema or exact count changed since the migration began; use a new target after stabilizing the source.")
            if existing is None and (self.state.get("phase") != "planned" or self.state.get("copied") != 0 or opt.verify_only):
                raise MigrationError("Owned target is missing; refusing to recreate a partially transferred collection.")
            if existing is not None:
                self.assert_owned(existing)
        else:
            if opt.resume or opt.verify_only:
                raise MigrationError("Original checkpoint is required for resume or verify-only; an existing target is never adopted.")
            if existing is not None:
                raise MigrationError("Target already exists; no overwrite is permitted.")
            self.state = {"format_version": FORMAT_VERSION, "migration_id": uuid.uuid4().hex,
                          "identity": identity, "schema_fingerprint": self.fingerprint, "source_count": count,
                          "phase": "planned", "offset": None, "copied": 0, "source_digest": EMPTY_DIGEST,
                          "transfer_finished": False, "complete": False}
            self.save_state(exclusive=True)
        self.check("target_safety", True, "Checkpoint and collection ownership agree, or target is new.")

    def assert_owned(self, info=None):
        if info is None:
            info = self.target.collection(self.options.target_collection)
        if (info.get("config", {}).get("metadata") or {}).get(MARKER) != self.marker():
            raise MigrationError("Target ownership marker does not match the checkpoint; no writes are allowed.")

    def schema_matches(self, info) -> bool:
        config = info["config"]
        differences = []
        params = config["params"]

        def dense_map(value):
            value = value or {}
            return {"": value} if isinstance(value.get("size"), int) else value

        expected_dense = dense_map(self.body.get("vectors"))
        actual_dense = dense_map(params.get("vectors"))
        if expected_dense.keys() != actual_dense.keys():
            differences.append("params.vectors: dense vector names differ")
        if set(self.body.get("sparse_vectors") or {}) != set(params.get("sparse_vectors") or {}):
            differences.append("params.sparse_vectors: sparse vector names differ")
        # Null quantization means disabled, rather than an unspecified option.
        if self.body.get("quantization_config") is None and config.get("quantization_config") is not None:
            differences.append("quantization_config: unexpectedly enabled")
        if self.body.get("strict_mode_config") is None and (config.get("strict_mode_config") or {}).get("enabled"):
            differences.append("strict_mode_config: unexpectedly enabled")
        for name in expected_dense.keys() & actual_dense.keys():
            if expected_dense[name].get("quantization_config") is None and actual_dense[name].get("quantization_config") is not None:
                differences.append(f"params.vectors.{name}.quantization_config: unexpectedly enabled")
        for key, expected in self.body.items():
            if key == "metadata":
                if not subset(expected, config.get("metadata") or {}):
                    differences.append("metadata: original collection metadata differs")
            elif key in ("vectors", "sparse_vectors", "shard_number", "sharding_method", "replication_factor", "write_consistency_factor", "on_disk_payload", "payload"):
                actual = config["params"].get(key)
                if key in ("vectors", "sparse_vectors"):
                    unnamed = key == "vectors" and isinstance(expected.get("size"), int)
                    if not unnamed and set(expected) != set(actual or {}):
                        differences.append(f"params.{key}: configured names differ")
                if not subset(expected, actual):
                    differences.append(f"params.{key}: requested options were not reproduced")
            elif not subset(expected, config.get("optimizer_config" if key == "optimizers_config" else key)):
                differences.append(f"{key}: requested options were not reproduced")
        self.report["destination_configuration_differences"] = differences
        return not differences

    def indexes_match(self, info) -> bool:
        actual = info.get("payload_schema", {})
        return actual.keys() == self.indexes.keys() and all(
            actual[name]["data_type"] == expected["type"] and
            subset(expected, actual[name].get("params") or {"type": actual[name]["data_type"]})
            for name, expected in self.indexes.items())

    def prepare_target(self):
        name = self.options.target_collection
        # Recheck under the lock, including the create-time race window.
        existing = self.target.collection(name, optional=True)
        if existing is None:
            if self.state["phase"] != "planned" or self.state["copied"]:
                raise MigrationError("Target disappeared; refusing to recreate it.")
            body = deepcopy(self.body)
            body["metadata"] = {**(body.get("metadata") or {}), MARKER: self.marker()}
            # An ambiguous PUT result is recovered only on the next --resume,
            # after GET confirms this exact metadata token.
            self.target.request("PUT", self.target.collection_path(name), body, retry=False)
            existing = self.target.collection(name)
        self.assert_owned(existing)
        if not self.schema_matches(existing):
            raise MigrationError("Destination did not reproduce every requested configuration option; collection kept for inspection, no points copied.")
        self.state["phase"] = "copying"
        self.save_state()
        for field, definition in self.indexes.items():
            self.assert_owned()
            current = self.target.collection(name).get("payload_schema", {}).get(field)
            if current:
                if current["data_type"] != definition["type"] or not subset(definition, current.get("params") or {"type": current["data_type"]}):
                    raise MigrationError("An existing target payload index differs; no index is overwritten.")
                continue
            self.target.request("PUT", self.target.collection_path(name) + "/index?wait=true",
                                {"field_name": field, "field_schema": definition},
                                before_attempt=self.assert_owned)
        if not self.indexes_match(self.target.collection(name)):
            raise MigrationError("Destination did not reproduce all payload index options.")

    def transfer(self, count):
        opt = self.options
        while not self.state["transfer_finished"]:
            records, next_offset = self.source.scroll(opt.source_collection, self.transfer_limit, self.state["offset"])
            if next_offset is not None and (not records or next_offset == self.state["offset"]):
                raise MigrationError("Source scroll made no progress; checkpoint preserved.")
            if not all(check_vector_shape(record, self.body) for record in records):
                raise MigrationError("Source vector names or dimensions do not match the collection schema.")
            if records:
                self.assert_owned()
                if not self.schema_matches(self.target.collection(opt.target_collection)):
                    raise MigrationError("Target schema changed while copying; stopping before upsert.")
                self.target.upsert(opt.target_collection, records, before_attempt=self.assert_owned)
            self.state["source_digest"] = extend_digest(self.state["source_digest"], records)
            self.state["copied"] += len(records)
            self.state["offset"] = next_offset
            self.state["transfer_finished"] = next_offset is None
            self.save_state()
            percentage = 100 * self.state["copied"] / count if count else 100
            self.log(f"Copied {self.state['copied']}/{count} points ({percentage:.1f}%). Checkpoint saved.")
        if self.state["copied"] != count:
            raise MigrationError("Transferred point count differs from the original exact count; source may have changed.")

    def validate(self, original_count):
        opt = self.options
        self.assert_owned()
        source_count = self.source.count(opt.source_collection)
        destination_count = self.target.count(opt.target_collection)
        self.report["source_count"], self.report["destination_count"] = source_count, destination_count
        self.check("point_count", source_count == destination_count == original_count == self.state["copied"],
                   f"Original={original_count}, source={source_count}, destination={destination_count}, transferred={self.state['copied']}.")
        target_info = self.target.collection(opt.target_collection)
        schema_ok = self.schema_matches(target_info)
        self.check("payload_indexes", self.indexes_match(target_info), "Compare every payload index type and original index option.")
        ids_ok = vectors_ok = payloads_ok = shapes_ok = True
        sampled_ids_ok = sampled_vectors_ok = sampled_payloads_ok = True
        rolling, offset, verified, samples = EMPTY_DIGEST, None, 0, []
        # Keep a deterministic sample distributed across the ID space using the
        # smallest ID hashes; no vectors/payloads enter persistent state.
        while True:
            records, next_offset = self.source.scroll(opt.source_collection, self.read_limit, offset)
            if next_offset is not None and (not records or next_offset == offset):
                raise MigrationError("Source verification scroll made no progress.")
            destination = self.target.retrieve(opt.target_collection, [record.id for record in records]) if records else []
            lookup = {record.id: record for record in destination}
            ids_ok &= len(lookup) == len(records)
            for record in records:
                other = lookup.get(record.id)
                same_id = other is not None
                same_vector = same_id and vectors_equal(record.vector, other.vector)
                same_payload = same_id and canonical(record.payload) == canonical(other.payload)
                shape_ok = check_vector_shape(record, self.body) and same_id and check_vector_shape(other, self.body)
                ids_ok &= same_id
                vectors_ok &= same_vector
                payloads_ok &= same_payload
                shapes_ok &= shape_ok
                samples.append((digest(record.id), record, same_id, same_vector, same_payload))
                samples.sort(key=lambda item: item[0])
                del samples[opt.sample_size:]
            rolling = extend_digest(rolling, records)
            verified += len(records)
            self.log(f"Verified {verified}/{original_count} points.")
            if next_offset is None:
                break
            offset = next_offset
        for _, _, same_id, same_vector, same_payload in samples:
            sampled_ids_ok &= same_id
            sampled_vectors_ok &= same_vector
            sampled_payloads_ok &= same_payload
        self.check("vector_dimensions_names", schema_ok and shapes_ok, "Compare collection vector settings and every transferred point's vector names and dimensions.")
        self.check("sampled_point_ids", sampled_ids_ok, f"Compared {len(samples)} deterministic sampled point IDs.")
        self.check("sampled_vectors", sampled_vectors_ok, "Dense, multivector and sparse values compared with float32 tolerance (rtol=1e-5, atol=1e-6); sparse indices exact.")
        self.check("sampled_payloads", sampled_payloads_ok, "Sampled payloads compared as complete canonical JSON, including value types.")
        self.check("all_points", ids_ok and vectors_ok and payloads_ok and verified == original_count,
                   f"Streamed comparison of all {verified} source IDs, complete vectors and payloads against destination retrieves.")
        final_info = self.source.collection(opt.source_collection)
        _, final_indexes, final_issues, _ = configuration_plan(final_info, opt.allow_config_adjustments)
        stable = (not final_issues and digest({"config": final_info["config"], "indexes": final_indexes}) == self.fingerprint
                  and self.source.count(opt.source_collection) == original_count
                  and rolling == self.state["source_digest"] and self.state["transfer_finished"])
        self.check("source_unchanged", stable, "Compare final source schema/count and full rolling content digest with the original transfer.")
        self.similarity(samples, original_count)

    def similarity(self, samples, count):
        candidate = None
        for _, record, *_ in samples:
            vector = record.vector
            if isinstance(vector, list) and vector:
                candidate = (vector, None)
                break
            if isinstance(vector, dict):
                for name, value in vector.items():
                    if isinstance(value, list) and value or isinstance(value, models.SparseVector) and value.indices:
                        candidate = (value, name)
                        break
            if candidate:
                break
        if candidate is None:
            self.check("sample_similarity", True, "Not applicable: no nonempty query vector in the bounded sample.", applicable=False)
            return
        vector, using = candidate
        limit = min(5, count, self.query_limit or 5)
        source = self.source.query(self.options.source_collection, vector, using, limit)
        destination = self.target.query(self.options.target_collection, vector, using, limit)
        scores_equal = len(source) == len(destination) and all(math.isclose(a.score, b.score, rel_tol=1e-5, abs_tol=1e-6) for a, b in zip(source, destination))
        self.check("sample_similarity", bool(source) and scores_equal,
                   "Executed exact similarity queries against both endpoints; compared ranked scores (tied IDs can differ).", applicable=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-collection", help="Explicit Cloud collection; otherwise use unambiguous repo QDRANT_COLLECTION.")
    result.add_argument("--target-collection", help="New local collection (default: SOURCE__cloud_copy).")
    result.add_argument("--local-url", default=os.getenv("QDRANT_LOCAL_URL", "http://127.0.0.1:6333"))
    result.add_argument("--dry-run", action="store_true", help="Inspect source and both connections; no database writes or checkpoints.")
    result.add_argument("--resume", action="store_true", help="Resume only with matching checkpoint and target ownership metadata.")
    result.add_argument("--verify-only", action="store_true", help="Read-only database verification using the original checkpoint.")
    result.add_argument("--allow-config-adjustments", action="store_true", help="Explicitly allow reported single-node topology adjustments; never unknown options.")
    result.add_argument("--checkpoint", type=Path, help="Default: ignored reports/qdrant-migrations/IDENTITY.checkpoint.json.")
    result.add_argument("--report", type=Path, help="Default: ignored reports/qdrant-migrations/IDENTITY.report.json.")
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--sample-size", type=int, default=16)
    result.add_argument("--retries", type=int, default=3, help="Additional retries on timeout, rate limiting and transient server failures.")
    result.add_argument("--timeout", type=int, default=60)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for logger in ("httpx", "httpcore", "qdrant_client"):
        logging.getLogger(logger).setLevel(logging.CRITICAL)
    secrets = ()
    try:
        cloud_url = os.getenv("QDRANT_CLOUD_URL")
        if not cloud_url:
            raise MigrationError("Set QDRANT_CLOUD_URL to the credential-free HTTPS Cloud REST endpoint.")
        cloud_url = validate_endpoint_url(cloud_url, local=False)
        local_url = validate_endpoint_url(args.local_url, local=True)
        if not 0 <= args.retries <= 10 or not 1 <= args.timeout <= 600:
            raise MigrationError("Use retries 0..10 and timeout 1..600 seconds.")
        key = cloud_api_key(dict(os.environ))
        local_key = os.getenv("QDRANT_LOCAL_API_KEY")
        secrets = tuple(value for value in (key, local_key) if value)
        with Endpoint(cloud_url, key, readonly=True, retries=args.retries, timeout=args.timeout) as source, Endpoint(local_url, local_key, readonly=False, retries=args.retries, timeout=args.timeout) as target:
            # Fail connectivity before asking the operator to choose a collection.
            available = source.collections()
            target.collections()
            configured = args.source_collection
            try:
                name = select_collection(available, configured, ROOT, dict(os.environ), interactive=False)
            except MigrationError:
                print(json.dumps(redact({"available_cloud_collections": available}, secrets), ensure_ascii=True))
                name = select_collection(available, configured, ROOT, dict(os.environ), interactive=sys.stdin.isatty())
            destination = args.target_collection or name + "__cloud_copy"
            identity = digest([cloud_url, local_url, name, destination])[:20]
            directory = ROOT / "reports" / "qdrant-migrations"
            options = Options(name, destination, args.checkpoint or directory / (identity + ".checkpoint.json"),
                              args.report or directory / (identity + ".report.json"), args.batch_size, args.sample_size,
                              args.dry_run, args.resume, args.verify_only, args.allow_config_adjustments)
            report = Migration(source, target, options).run()
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return 0 if report["overall"] == "PASS" else 1
    except KeyboardInterrupt:
        print("Interrupted; keep the original checkpoint and rerun with --resume.", file=sys.stderr)
        return 130
    except MigrationError as exc:
        print(redact(str(exc), secrets), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Stopped safely ({type(exc).__name__}); credentials and raw error details omitted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
