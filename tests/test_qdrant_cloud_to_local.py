"""Hermetic REST integration tests: no credentials, Docker or network required."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import httpx
import pytest


def migration_module():
    try:
        return importlib.import_module("scripts.migrate_qdrant_cloud_to_local")
    except ModuleNotFoundError as exc:
        if exc.name == "scripts.migrate_qdrant_cloud_to_local":
            pytest.fail("Standalone Qdrant migration tool has not been implemented")
        raise


def source_schema(vectors=None, sparse=None):
    return {
        "status": "green", "optimizer_status": "ok",
        "config": {
            "params": {
                "vectors": vectors if vectors is not None else {"dense_e5": {"size": 2, "distance": "Dot"}},
                "sparse_vectors": sparse,
                "shard_number": 1, "sharding_method": "auto",
                "replication_factor": 1, "write_consistency_factor": 1,
                "on_disk_payload": True,
            },
            "hnsw_config": {"m": 16, "ef_construct": 100, "full_scan_threshold": 10000},
            "optimizer_config": {"deleted_threshold": 0.2, "indexing_threshold": 10000},
            "wal_config": {"wal_capacity_mb": 32, "wal_segments_ahead": 0},
            "quantization_config": None, "strict_mode_config": None,
            "metadata": {"purpose": "history"},
        },
        "payload_schema": {"title": {"data_type": "keyword", "params": {"type": "keyword", "is_tenant": True}, "points": 3}},
    }


class FakeServers:
    """Mock only HTTP; exercise the real SDK serializers and migration logic."""

    def __init__(self, schema=None, points=None):
        self.source = {"vn_history_v1_e5": deepcopy(schema or source_schema())}
        self.source_points = deepcopy(points if points is not None else [
            {"id": 0, "vector": {"dense_e5": [1.0, 0.0]}, "payload": {"title": "A", "text": "full text", "nested": {"keep": [None, True, 42]}}},
            {"id": 7, "vector": {"dense_e5": [0.0, 1.0]}, "payload": {"title": "B", "extra": "do not truncate"}},
            {"id": 99, "vector": {"dense_e5": [0.5, 0.5]}, "payload": {}},
        ])
        self.local = {}
        self.local_points = {}
        self.calls = []
        self.upsert_attempts = 0
        self.fail_upsert_at = None
        self.fail_after_write = False
        self.fail_create_after_write = False
        self.transient_upserts = 0
        self.corrupt_retrieval = False
        self.fail_status = None

    def response(self, result, status=200):
        return httpx.Response(status, json={"result": result, "status": "ok", "time": 0.0})

    def handle(self, request):
        cloud = request.url.host == "cloud.example"
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.calls.append((cloud, request.method, path, deepcopy(body)))
        if self.fail_status:
            return httpx.Response(self.fail_status, json={"status": {"error": "credential-super-secret-123"}})
        database = self.source if cloud else self.local
        if path == "/":
            return httpx.Response(200, json={"title": "qdrant", "version": "1.17.1"})
        if path == "/collections":
            return self.response({"collections": [{"name": name} for name in database]})
        parts = path.strip("/").split("/")
        assert parts[0] == "collections", path
        name = parts[1]
        if len(parts) == 2 and request.method == "GET":
            return self.response(deepcopy(database[name])) if name in database else self.response(None, 404)
        if len(parts) == 2 and request.method == "PUT":
            assert not cloud and name not in database, "Attempt to overwrite a collection"
            config = deepcopy(body)
            params = {k: config.pop(k) for k in list(config) if k in ("vectors", "sparse_vectors", "shard_number", "sharding_method", "replication_factor", "write_consistency_factor", "on_disk_payload")}
            config["optimizer_config"] = config.pop("optimizers_config", {})
            config["params"] = params
            database[name] = {"config": config, "payload_schema": {}, "status": "green", "optimizer_status": "ok"}
            self.local_points[name] = {}
            if self.fail_create_after_write:
                self.fail_create_after_write = False
                raise httpx.ReadTimeout("credential-super-secret-123")
            return self.response(True)
        if len(parts) == 3 and parts[2] == "index":
            assert not cloud
            field = body["field_schema"]
            if isinstance(field, str):
                field = {"type": field}
            database[name]["payload_schema"][body["field_name"]] = {"data_type": field["type"], "params": field, "points": 0}
            return self.response({"operation_id": 1, "status": "completed"})
        points = self.source_points if cloud else list(self.local_points.get(name, {}).values())
        if parts[2:] == ["points", "count"]:
            assert body["exact"] is True
            return self.response({"count": len(points)})
        if parts[2:] == ["points", "scroll"]:
            assert body["with_vector"] is True and body["with_payload"] is True
            ordered = sorted(points, key=lambda p: (isinstance(p["id"], str), p["id"]))
            offset = body.get("offset")
            start = next((i for i,p in enumerate(ordered) if p["id"] == offset), len(ordered)) if offset is not None else 0
            page = ordered[start:start + body["limit"]]
            end = start + len(page)
            next_offset = ordered[end]["id"] if end < len(ordered) else None
            return self.response({"points": deepcopy(page), "next_page_offset": next_offset})
        if parts[2:] == ["points"] and request.method == "PUT":
            assert not cloud and request.url.params.get("wait") == "true"
            self.upsert_attempts += 1
            if self.transient_upserts:
                self.transient_upserts -= 1
                return self.response(None, 503)
            should_fail = self.upsert_attempts == self.fail_upsert_at
            if should_fail and not self.fail_after_write:
                raise httpx.ReadTimeout("credential-super-secret-123")
            for point in body["points"]:
                self.local_points[name][point["id"]] = deepcopy(point)
            if should_fail:
                raise httpx.ReadTimeout("credential-super-secret-123")
            return self.response({"operation_id": 2, "status": "completed"})
        if parts[2:] == ["points"] and request.method == "POST":
            records = deepcopy([p for p in points if p["id"] in body["ids"]])
            if self.corrupt_retrieval and records:
                records[0]["payload"] = {"corrupt": True}
            return self.response(records)
        if parts[2:] == ["points", "query"]:
            return self.response({"points": [{"id": p["id"], "score": 1.0, "version": 1} for p in points[:body["limit"]]]})
        raise AssertionError((request.method, path, body))

    def clients(self, module, retries=0):
        transport = httpx.MockTransport(self.handle)
        return (
            module.Endpoint("https://cloud.example", "credential-super-secret-123", readonly=True, retries=retries, transport=transport, sleep=lambda _: None),
            module.Endpoint("http://127.0.0.1:6333", readonly=False, retries=retries, transport=transport, sleep=lambda _: None),
        )


def run_migration(tmp_path, servers, **overrides):
    module = migration_module()
    retries = overrides.pop("retries", 0)
    source, target = servers.clients(module, retries)
    options = module.Options(source_collection="vn_history_v1_e5", target_collection="history_cloud_copy", checkpoint=tmp_path / "state.json", report=tmp_path / "report.json", batch_size=2, sample_size=2, **overrides)
    with source, target:
        return module.Migration(source, target, options, log=lambda _: None).run()


def test_dry_run_is_read_only_and_inspects_full_schema(tmp_path):
    servers = FakeServers()
    before = deepcopy(servers.source)
    report = run_migration(tmp_path, servers, dry_run=True)
    assert report["overall"] == "PASS", report
    assert report["source_count"] == 3
    assert report["source_schema"]["vectors"]["dense_e5"]["size"] == 2
    assert report["payload_indexes"]["title"]["params"]["is_tenant"] is True
    assert report["estimates"]["json_transfer_bytes"] > 0
    assert not (tmp_path / "state.json").exists()
    assert servers.local == {} and servers.source == before
    assert all(method == "GET" or path.endswith(("/scroll", "/count")) for _, method, path, _ in servers.calls)


def test_copy_keeps_every_point_and_passes_validation(tmp_path):
    servers = FakeServers()
    before = deepcopy(servers.source_points)
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "PASS", report
    assert report["source_count"] == report["destination_count"] == 3
    assert list(servers.local_points["history_cloud_copy"].values()) == before
    assert servers.source_points == before
    assert report["checks"]["all_points"]["status"] == "PASS"
    assert report["checks"]["sample_similarity"]["status"] == "PASS"
    assert servers.local["history_cloud_copy"]["payload_schema"]["title"]["params"]["is_tenant"] is True
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["complete"] is True and state["copied"] == 3
    assert "credential-super-secret-123" not in (tmp_path / "state.json").read_text()
    assert "credential-super-secret-123" not in (tmp_path / "report.json").read_text()


@pytest.mark.parametrize("after_write", [False, True])
def test_resume_replays_uncheckpointed_batch_idempotently(tmp_path, after_write):
    servers = FakeServers()
    servers.fail_upsert_at = 2
    servers.fail_after_write = after_write
    first = run_migration(tmp_path, servers)
    assert first["overall"] == "FAIL"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["copied"] == 2 and state["offset"] == 99
    servers.fail_upsert_at = None
    resumed = run_migration(tmp_path, servers, resume=True)
    assert resumed["overall"] == "PASS", resumed
    assert len(servers.local_points["history_cloud_copy"]) == 3
    assert len([c for c in servers.calls if c[1] == "PUT" and c[2] == "/collections/history_cloud_copy"]) == 1


def test_resume_after_ambiguous_create_checks_server_marker(tmp_path):
    servers = FakeServers()
    servers.fail_create_after_write = True
    assert run_migration(tmp_path, servers)["overall"] == "FAIL"
    assert run_migration(tmp_path, servers, resume=True)["overall"] == "PASS"


@pytest.mark.parametrize("target", ["history_cloud_copy", "vinai_day03_lab"])
def test_never_overwrites_preexisting_or_protected_collection(tmp_path, target):
    servers = FakeServers()
    servers.local[target] = source_schema()
    servers.local_points[target] = {55: {"id": 55, "vector": {"dense_e5": [1.0, 0.0]}, "payload": {"existing": True}}}
    before = deepcopy(servers.local_points)
    module = migration_module()
    source, destination = servers.clients(module)
    options = module.Options(source_collection="vn_history_v1_e5", target_collection=target, checkpoint=tmp_path / "state.json", report=tmp_path / "report.json")
    with source, destination:
        report = module.Migration(source, destination, options, log=lambda _: None).run()
    assert report["overall"] == "FAIL"
    assert servers.local_points == before
    assert not any(not c[0] and c[1] in ("PUT", "PATCH", "DELETE") for c in servers.calls)


def test_resume_refuses_replaced_collection(tmp_path):
    servers = FakeServers()
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    servers.local["history_cloud_copy"]["config"]["metadata"] = {}
    count = servers.upsert_attempts
    report = run_migration(tmp_path, servers, resume=True)
    assert report["overall"] == "FAIL"
    assert servers.upsert_attempts == count


def test_retry_is_idempotent_and_reports_progress(tmp_path):
    servers = FakeServers()
    servers.transient_upserts = 1
    report = run_migration(tmp_path, servers, retries=2)
    assert report["overall"] == "PASS", report
    assert servers.upsert_attempts == 3


def test_named_dense_sparse_multivectors_and_uuid_ids(tmp_path):
    schema = source_schema(vectors={"text": {"size": 2, "distance": "Dot"}, "late": {"size": 2, "distance": "Dot", "multivector_config": {"comparator": "max_sim"}}}, sparse={"sparse": {"modifier": "idf", "index": {"on_disk": False}}})
    points = [{"id": "8c77a5e4-e507-452e-9e15-e97a1ee08da9", "vector": {"text": [1.0, 0.0], "late": [[1.0, 0.0], [0.0, 1.0]], "sparse": {"indices": [1, 70000], "values": [0.4, 2.0]}}, "payload": {"title": "full", "text": "preserve entire text"}}]
    servers = FakeServers(schema, points)
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "PASS", report
    assert list(servers.local_points["history_cloud_copy"].values()) == points
    assert report["checks"]["vector_dimensions_names"]["status"] == "PASS"


def test_unnamed_dense_vectors(tmp_path):
    servers = FakeServers(source_schema(vectors={"size": 2, "distance": "Dot"}), [{"id": 1, "vector": [0.1, 0.2], "payload": {"keep": "all"}}])
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    assert servers.local_points["history_cloud_copy"][1]["vector"] == [0.1, 0.2]


def test_verify_only_detects_payload_corruption_without_writes(tmp_path):
    servers = FakeServers()
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    servers.corrupt_retrieval = True
    servers.calls.clear()
    report = run_migration(tmp_path, servers, verify_only=True)
    assert report["overall"] == "FAIL"
    assert report["checks"]["all_points"]["status"] == "FAIL"
    assert not any(c[1] in ("PUT", "PATCH", "DELETE") for c in servers.calls)


def test_resume_rejects_same_count_source_mutation(tmp_path):
    servers = FakeServers()
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    servers.source_points[0]["payload"]["text"] = "changed while count stayed constant"
    report = run_migration(tmp_path, servers, verify_only=True)
    assert report["overall"] == "FAIL"
    assert report["checks"]["source_unchanged"]["status"] == "FAIL"


@pytest.mark.parametrize("unknown", ["params", "config", "index"])
def test_unknown_configuration_blocks_before_creation(tmp_path, unknown):
    schema = source_schema()
    if unknown == "params":
        schema["config"]["params"]["future_storage_mode"] = True
    elif unknown == "config":
        schema["config"]["future_config"] = {"important": True}
    else:
        schema["payload_schema"]["title"]["future_index_option"] = True
    servers = FakeServers(schema)
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "FAIL"
    assert servers.local == {}
    assert report["configuration_issues"]


def test_topology_adjustment_requires_explicit_flag(tmp_path):
    schema = source_schema()
    schema["config"]["params"]["replication_factor"] = 2
    servers = FakeServers(schema)
    assert run_migration(tmp_path, servers, dry_run=True)["overall"] == "FAIL"
    report = run_migration(tmp_path, servers, allow_config_adjustments=True)
    assert report["overall"] == "PASS", report
    assert report["configuration_adjustments"]
    assert servers.local["history_cloud_copy"]["config"]["params"]["replication_factor"] == 1


def test_strict_mode_small_read_and_write_limits_are_respected(tmp_path):
    class LimitedServers(FakeServers):
        def handle(self, request):
            body = json.loads(request.content) if request.content else {}
            if request.url.path.endswith(("/scroll", "/query")) and body.get("limit", 0) > 1:
                return self.response(None, 400)
            if request.method == "POST" and request.url.path.endswith("/points") and len(body.get("ids", [])) > 1:
                return self.response(None, 400)
            return super().handle(request)

    schema = source_schema()
    schema["config"]["strict_mode_config"] = {"enabled": True, "max_query_limit": 1, "upsert_max_batchsize": 1}
    servers = LimitedServers(schema)
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "PASS", report
    assert report["destination_count"] == 3
    assert report["estimates"]["transfer_batches"] == 3


def test_getpass_refuses_an_echoing_fallback(monkeypatch):
    module = migration_module()

    def insecure_prompt(prompt):
        module.warnings.warn("Cannot disable echo", module.getpass.GetPassWarning)
        return "must-never-read-this"

    monkeypatch.setattr(module.getpass, "getpass", insecure_prompt)
    with pytest.raises(module.MigrationError, match="secure terminal"):
        module.cloud_api_key({})


def test_http_errors_never_expose_credentials(tmp_path):
    servers = FakeServers()
    servers.fail_status = 401
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "FAIL"
    assert "credential-super-secret-123" not in json.dumps(report)
    assert "credential-super-secret-123" not in (tmp_path / "report.json").read_text()


def test_collection_discovery_uses_repository_default_and_requires_explicit_ambiguity(tmp_path):
    module = migration_module()
    (tmp_path / "app").mkdir()
    (tmp_path / "app/config.py").write_text('class Settings:\n    qdrant_collection: str = "vn_history_v1_e5"\n')
    assert module.select_collection(["other", "vn_history_v1_e5"], None, tmp_path, {}, interactive=False) == "vn_history_v1_e5"
    with pytest.raises(module.MigrationError, match="source-collection"):
        module.select_collection(["one", "two"], None, tmp_path, {}, interactive=False)
    assert module.select_collection(["one", "two"], "two", tmp_path, {}, interactive=False) == "two"


@pytest.mark.parametrize("url", ["https://user:secret@cloud.example", "https://cloud.example?api_key=secret", "http://cloud.example"])
def test_cloud_url_rejects_embedded_credentials_or_plaintext(url):
    module = migration_module()
    with pytest.raises(module.MigrationError):
        module.validate_endpoint_url(url, local=False)


def test_destination_must_be_loopback():
    module = migration_module()
    with pytest.raises(module.MigrationError):
        module.validate_endpoint_url("https://another-cloud.example", local=True)


def test_missing_key_uses_secure_getpass(monkeypatch):
    module = migration_module()
    monkeypatch.setattr(module.getpass, "getpass", lambda prompt: "prompted-secret")
    assert module.cloud_api_key({}) == "prompted-secret"


def test_empty_collection_is_valid_and_similarity_is_not_applicable(tmp_path):
    servers = FakeServers(points=[])
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "PASS", report
    assert report["source_count"] == report["destination_count"] == 0
    assert report["checks"]["sample_similarity"]["applicable"] is False


def test_report_path_cannot_destroy_the_checkpoint(tmp_path):
    module = migration_module()
    servers = FakeServers()
    source, destination = servers.clients(module)
    state = tmp_path / "state.json"
    state.write_text('{"original_checkpoint": true}')
    options = module.Options("vn_history_v1_e5", "new", state, state, dry_run=True)
    with source, destination:
        report = module.Migration(source, destination, options, log=lambda _: None).run()
    assert report["overall"] == "FAIL"
    assert state.read_text() == '{"original_checkpoint": true}'


def test_extra_target_vector_names_are_not_accepted(tmp_path):
    servers = FakeServers()
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    servers.local["history_cloud_copy"]["config"]["params"]["vectors"]["unexpected"] = {"size": 3, "distance": "Dot"}
    report = run_migration(tmp_path, servers, verify_only=True)
    assert report["overall"] == "FAIL"
    assert report["checks"]["vector_dimensions_names"]["status"] == "FAIL"


@pytest.mark.parametrize("family", ["sparse", "quantization", "per_vector_quantization", "strict_mode"])
def test_absent_source_configuration_is_not_silently_enabled_on_target(tmp_path, family):
    servers = FakeServers()
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    config = servers.local["history_cloud_copy"]["config"]
    if family == "sparse":
        config["params"]["sparse_vectors"] = {"unexpected": {}}
    elif family == "quantization":
        config["quantization_config"] = {"scalar": {"type": "int8", "always_ram": True}}
    elif family == "per_vector_quantization":
        config["params"]["vectors"]["dense_e5"]["quantization_config"] = {"scalar": {"type": "int8"}}
    else:
        config["strict_mode_config"] = {"enabled": True, "max_query_limit": 1}
    report = run_migration(tmp_path, servers, verify_only=True)
    assert report["overall"] == "FAIL"
    assert report["checks"]["vector_dimensions_names"]["status"] == "FAIL"


def test_ownership_is_rechecked_on_each_upsert_retry(tmp_path):
    class ReplaceOnRetry(FakeServers):
        def handle(self, request):
            response = super().handle(request)
            if request.method == "PUT" and request.url.path.endswith("/points") and response.status_code == 503:
                self.local["history_cloud_copy"]["config"]["metadata"] = {}
            return response

    servers = ReplaceOnRetry()
    servers.transient_upserts = 1
    report = run_migration(tmp_path, servers, retries=2)
    assert report["overall"] == "FAIL"
    assert servers.upsert_attempts == 1
    assert servers.local_points["history_cloud_copy"] == {}


def test_ownership_is_rechecked_on_each_index_retry(tmp_path):
    class ReplaceOnIndexRetry(FakeServers):
        index_attempts = 0

        def handle(self, request):
            if request.method == "PUT" and request.url.path.endswith("/index"):
                self.index_attempts += 1
                if self.index_attempts == 1:
                    self.local["history_cloud_copy"]["config"]["metadata"] = {}
                    return self.response(None, 503)
            return super().handle(request)

    servers = ReplaceOnIndexRetry()
    report = run_migration(tmp_path, servers, retries=2)
    assert report["overall"] == "FAIL"
    assert servers.index_attempts == 1
    assert servers.local["history_cloud_copy"]["payload_schema"] == {}


def test_missing_checkpoint_never_adopts_existing_target(tmp_path):
    servers = FakeServers()
    servers.local["history_cloud_copy"] = source_schema()
    servers.local_points["history_cloud_copy"] = {}
    report = run_migration(tmp_path, servers, resume=True)
    assert report["overall"] == "FAIL"
    assert servers.upsert_attempts == 0


def test_missing_partially_copied_target_is_not_recreated(tmp_path):
    servers = FakeServers()
    servers.fail_upsert_at = 2
    assert run_migration(tmp_path, servers)["overall"] == "FAIL"
    del servers.local["history_cloud_copy"]
    servers.calls.clear()
    assert run_migration(tmp_path, servers, resume=True)["overall"] == "FAIL"
    assert not any(call[1] == "PUT" for call in servers.calls)


def test_rejected_nested_config_is_reported_before_any_points_copy(tmp_path):
    class DropOption(FakeServers):
        def handle(self, request):
            response = super().handle(request)
            if request.method == "PUT" and request.url.path == "/collections/history_cloud_copy":
                self.local["history_cloud_copy"]["config"]["hnsw_config"].pop("future_important_option", None)
            return response

    schema = source_schema()
    schema["config"]["hnsw_config"]["future_important_option"] = True
    servers = DropOption(schema)
    report = run_migration(tmp_path, servers)
    assert report["overall"] == "FAIL"
    assert servers.upsert_attempts == 0


@pytest.mark.parametrize("corruption", ["vector", "id", "payload", "index"])
def test_verification_reports_each_kind_of_corruption(tmp_path, corruption):
    servers = FakeServers()
    assert run_migration(tmp_path, servers)["overall"] == "PASS"
    if corruption == "vector":
        servers.local_points["history_cloud_copy"][0]["vector"]["dense_e5"] = [0.0, 1.0]
    elif corruption == "id":
        servers.local_points["history_cloud_copy"][1234] = servers.local_points["history_cloud_copy"].pop(0)
        servers.local_points["history_cloud_copy"][1234]["id"] = 1234
    elif corruption == "payload":
        servers.local_points["history_cloud_copy"][0]["payload"]["nested"]["keep"][1] = 1
    else:
        servers.local["history_cloud_copy"]["payload_schema"]["title"]["params"]["is_tenant"] = False
    report = run_migration(tmp_path, servers, verify_only=True)
    assert report["overall"] == "FAIL"
    check = "payload_indexes" if corruption == "index" else "all_points"
    assert report["checks"][check]["status"] == "FAIL"
