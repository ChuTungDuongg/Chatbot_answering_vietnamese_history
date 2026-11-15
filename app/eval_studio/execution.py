"""Request-scoped benchmark variants around the existing chat execution boundary.

Only question text enters inference. Prompt configurations are explicit artifacts;
model messages, native replay, reasoning and tool arguments never leave this module.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from copy import copy, deepcopy
import hashlib
import json
import math
import os
import platform
import re
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from app.api import routes
from app.central import planning
from app.central import runtime as central_prompts
from app.chat_modes import ChatMode
from app.config import settings
from app.mcp.schemas import ToolSteering
from app.models.base import ModelDone
from app.models.tool_calls import HermesFunctionCallCodec
from app.prompts.loader import PROMPT_NAMES, PROMPT_ROOT, load_prompt
from app.rag import prompting
from app.schemas import ChatRequest
from app.services.metadata import _git_commit, _hash_file
from app.telemetry import RequestTrace
from .privacy import SECRET_PATTERN, reject_secrets
from .v11 import EvidenceInput, RunInput

LIVE_PROMPTS = tuple(name for name in PROMPT_NAMES if name.startswith(("shared/", "hybrid/", "central/")))


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _request(app):
    return SimpleNamespace(app=app, state=SimpleNamespace(owner_authenticated=True), headers={})


def _payload(frozen, question="Benchmark configuration"):
    central = frozen["chat_mode"] == "central"
    return ChatRequest(
        provider=frozen["provider"],
        model=frozen["model_id"] if frozen["provider"] == "openai" else None,
        reasoning_effort=frozen["reasoning_effort"] if frozen["provider"] == "openai" else None,
        conversation_id=uuid4(),
        question=question,
        mode=frozen["chat_mode"],
        retrieval_backend=frozen["dense_backend"],
        final_k=frozen["final_k"],
        steering=ToolSteering.model_validate(frozen["steering"]) if central else None,
        debug=False,
    )


def _live_prompts():
    resources = {name: load_prompt(name) for name in LIVE_PROMPTS}
    resources.update(
        {
            "shared/grounded_system": prompting.SYSTEM_PROMPT,
            "hybrid/live_style": prompting.LIVE_STYLE_INSTRUCTION,
            "central/planner": planning.PLANNER_INSTRUCTION,
            "central/completion_parameter": planning.COMPLETION_DESCRIPTION,
            "central/independent_parameter": planning.INDEPENDENT_DESCRIPTION,
            "central/external_domain": central_prompts.EXTERNAL_DOMAIN_INSTRUCTION,
            "central/grounded_final": central_prompts.GROUNDED_FINAL_INSTRUCTION,
            "central/execution_limit": central_prompts.EXECUTION_LIMIT_INSTRUCTION,
            "central/recover_empty_plan": central_prompts.EMPTY_PLAN_RECOVERY,
        }
    )
    return resources


def _provenance(app, runtime, mode):
    service = getattr(app.state, "rag_service", None)
    retriever = getattr(runtime, "retriever", None) or getattr(app.state, "retriever", None)
    retrieval = getattr(retriever, "retrieval_config", None)
    if callable(retrieval):
        retrieval = retrieval()
    manifest = getattr(service, "manifest", None) or {}
    indexes = manifest.get("manifest_fingerprints") or None
    model = runtime.model
    result = {
        "git_commit": os.getenv("DEPLOYMENT_GIT_COMMIT") or _git_commit(),
        "deployment_source_sha256": os.getenv("DEPLOYMENT_SOURCE_SHA256"),
        "corpus_hash": getattr(service, "corpus_sha256", None),
        "ordered_chunk_id_hash": getattr(service, "ordered_chunk_id_sha256", None),
        "available_dense_backends": sorted(getattr(service, "dense_retrievers", {})),
        "retrieval_index_hash": _hash(indexes) if indexes else None,
        "index_manifest_fingerprints": indexes,
        "artifact_manifest_hashes": {
            key: _hash_file(path)
            for key, path in (
                ("runtime", settings.manifest_path),
                ("faiss", settings.faiss_manifest_path),
                ("bm25", settings.bm25_manifest_path),
                ("index", settings.index_manifest_path),
                ("qdrant", settings.qdrant_manifest_path),
            )
        },
        "model_id": model.model_id,
        "model_revision": getattr(model, "resolved_revision", None),
        "adapter_fingerprint": getattr(model, "adapter_fingerprint", None),
        "generation_settings": deepcopy(getattr(model, "generation_settings", None)),
        "retrieval_config": deepcopy(retrieval),
        "engine": deepcopy(getattr(model, "engine_metadata", None)),
        "runtime_environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "app_mode": settings.app_mode,
            "app_env": settings.app_env,
            "local_inference_enabled": settings.local_inference_enabled,
            "openai_max_output_tokens": settings.openai_max_output_tokens,
            "final_max_new_tokens": (
                settings.hybrid_max_new_tokens
                if mode == ChatMode.HYBRID
                else settings.central_final_max_new_tokens
            ),
        },
        "central_config": None,
    }
    if mode == ChatMode.CENTRAL:
        limits = getattr(runtime, "limits", None)
        result["central_config"] = {
            "max_action_rounds": runtime.max_action_rounds,
            "action_max_new_tokens": runtime.action_max_new_tokens,
            "planning_reasoning_effort": runtime.planning_reasoning_effort,
            "max_mcp_tools": runtime.max_mcp_tools,
            "mcp_schema_budget": runtime.mcp_schema_budget,
            "limits": dict(vars(limits)) if limits is not None else None,
            "mcp_capabilities": runtime.mcp_manager.capabilities() if runtime.mcp_manager else None,
        }
    # Metadata is an artifact too: fail closed rather than persisting credentials.
    reject_secrets(result)
    return result


def _environment(app, runtime, mode):
    descriptors = deepcopy(runtime.tools.describe()) if mode == ChatMode.CENTRAL else []
    return {
        "prompts": _live_prompts(),
        "prompt_file_hashes": {name: _hash_file(PROMPT_ROOT / f"{name}.md") for name in LIVE_PROMPTS},
        "composed_prompts": {
            "live_system": prompting.LIVE_SYSTEM_PROMPT,
            "agentic_planner": central_prompts.AGENTIC_PLANNER_INSTRUCTION,
        },
        "tools": descriptors,
        "runtime": _provenance(app, runtime, mode),
    }


def freeze_configuration(app, experiment: dict, options: dict) -> dict:
    """Resolve an explicit experiment against the current, read-only runtime."""
    unknown = set(options) - {"prompt_overrides", "tool_descriptions", "mcp_enabled", "allowed_mcp_servers"}
    if unknown:
        raise ValueError("Unsupported benchmark configuration options")
    required = ("id", "snapshot_id", "provider", "model_id", "chat_mode", "dense_backend", "final_k")
    if any(experiment.get(key) is None or experiment.get(key) == "" for key in required):
        raise ValueError("Benchmark execution requires explicit experiment settings")
    frozen = {
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "snapshot_id": experiment["snapshot_id"],
        **{key: experiment[key] for key in required if key not in {"id", "snapshot_id"}},
        "reasoning_effort": experiment.get("reasoning_effort"),
        "planning_reasoning_effort": experiment.get("planning_reasoning_effort"),
        "tool_allowlist": deepcopy(experiment.get("tool_allowlist")),
    }
    declared_mcp = None
    if experiment.get("mcp_config"):
        try:
            declared_mcp = ToolSteering.model_validate_json(experiment["mcp_config"])
        except (ValueError, TypeError):
            raise ValueError("Experiment MCP metadata must be strict ToolSteering JSON to execute") from None
        for field in ("mcp_enabled", "allowed_mcp_servers"):
            if field in options and options[field] != getattr(declared_mcp, field):
                raise ValueError("Experiment MCP metadata conflicts with benchmark options")
        if declared_mcp.allowed_tools is not None:
            if frozen["tool_allowlist"] is not None and set(declared_mcp.allowed_tools) != set(
                frozen["tool_allowlist"]
            ):
                raise ValueError("Experiment MCP metadata conflicts with tool allowlist")
            frozen["tool_allowlist"] = list(declared_mcp.allowed_tools)
    frozen["declared_mcp_config"] = declared_mcp.model_dump(mode="json") if declared_mcp else None
    if frozen["provider"] == "openai" and not frozen["reasoning_effort"]:
        raise ValueError("OpenAI benchmark requires explicit reasoning effort")
    if frozen["provider"] == "local" and frozen["reasoning_effort"] is not None:
        raise ValueError("Local benchmark cannot select OpenAI reasoning effort")
    steering = {"allowed_tools": frozen["tool_allowlist"], "mcp_enabled": False, "allowed_mcp_servers": []}
    if declared_mcp is not None:
        steering.update(declared_mcp.model_dump(mode="json"))
    steering.update({key: options[key] for key in ("mcp_enabled", "allowed_mcp_servers") if key in options})
    if frozen["chat_mode"] == "hybrid" and (
        frozen["tool_allowlist"] is not None or steering["mcp_enabled"] or steering["allowed_mcp_servers"]
    ):
        raise ValueError("Tool steering requires Central mode")
    frozen["steering"] = ToolSteering.model_validate(steering).model_dump(mode="json")
    request = _request(app)
    payload = _payload(frozen)
    mode = ChatMode(frozen["chat_mode"])
    runtime = routes._runtime(request, mode, payload)
    if runtime.model.model_id != frozen["model_id"]:
        raise ValueError("Experiment model differs from the configured runtime")
    routes._resolve_backend(request, frozen["dense_backend"], prefer_qdrant=mode == ChatMode.CENTRAL)
    if mode == ChatMode.CENTRAL:
        effort = frozen["planning_reasoning_effort"]
        if effort not in {"inherit", "low", "medium", "high", "xhigh"}:
            raise ValueError("Central benchmark requires explicit planning reasoning effort")
        runtime = copy(runtime)
        runtime.planning_reasoning_effort = effort
    environment = _environment(app, runtime, mode)
    frozen["baseline_prompts"] = environment["prompts"]
    frozen["prompts"] = deepcopy(environment["prompts"])
    for name, value in options.get("prompt_overrides", {}).items():
        if name not in LIVE_PROMPTS or not isinstance(value, str) or not value.strip() or len(value) > 40000:
            raise ValueError("Invalid live prompt override")
        reject_secrets(value)
        frozen["prompts"][name] = value
    view = routes._resolve_tools(runtime, payload, mode)
    allowed = view.allowed_names if view is not None else frozenset()
    frozen["tool_allowlist"] = sorted(allowed) if mode == ChatMode.CENTRAL else None
    frozen["steering"]["allowed_tools"] = frozen["tool_allowlist"]
    frozen["tool_definitions"] = [deepcopy(row) for row in environment["tools"] if row["name"] in allowed]
    descriptions = options.get("tool_descriptions", {})
    if not set(descriptions).issubset(allowed):
        raise ValueError("Description override requires an allowed tool")
    for descriptor in frozen["tool_definitions"]:
        if descriptor["name"] in descriptions:
            value = descriptions[descriptor["name"]]
            if not isinstance(value, str) or not value.strip() or len(value) > 12000:
                raise ValueError("Invalid tool description override")
            reject_secrets(value)
            descriptor["description"] = value
    frozen["runtime_provenance"] = environment["runtime"]
    frozen["environment_hash"] = _hash(environment)
    frozen["prompt_hash"] = _hash(frozen["prompts"])
    frozen["tool_hash"] = _hash(frozen["tool_definitions"])
    reject_secrets(frozen)
    frozen["configuration_hash"] = _hash(frozen)
    return frozen


def verify_configuration(app, frozen: dict) -> None:
    """Reject a modified artifact or a changed live baseline before resuming."""
    if frozen.get("configuration_hash") != _hash(
        {k: v for k, v in frozen.items() if k != "configuration_hash"}
    ):
        raise ValueError("Benchmark configuration drift: artifact integrity mismatch")
    mode = ChatMode(frozen["chat_mode"])
    runtime = routes._runtime(_request(app), mode, _payload(frozen))
    if mode == ChatMode.CENTRAL:
        runtime = copy(runtime)
        runtime.planning_reasoning_effort = frozen["planning_reasoning_effort"]
    if frozen["environment_hash"] != _hash(_environment(app, runtime, mode)):
        raise ValueError("Benchmark configuration drift: live runtime baseline changed")


def _safe_url(value):
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return None
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except (ValueError, TypeError):
        return None


def _text(value):
    value = SECRET_PATTERN.sub("[redacted]", str(value))
    return re.sub(r'https?://[^\s<>"\']+', lambda match: _safe_url(match.group()) or "[redacted-url]", value)


def _code(value):
    if (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,79}", value)
        and not SECRET_PATTERN.search(value)
    ):
        return value
    return None


def _number(value):
    return value if type(value) in {int, float} and math.isfinite(value) and value >= 0 else None


def _score(value):
    return value if type(value) in {int, float} and math.isfinite(value) else None


def _fingerprint(arguments):
    # json.loads accepts non-finite numbers; an observation must not turn such
    # an invalid call into a new planner failure before the real validator runs.
    return hashlib.sha256(
        json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _Capture:
    def __init__(self):
        self.prepared = None
        self.retrieved = []
        self.tools = []
        self.phases = []
        self.retrieval_observed = False

    def calls(self, parsed, registry, schemas):
        controlled = {
            schema["function"]["name"]
            for schema in schemas
            if planning.COMPLETION_FIELD in schema["function"].get("parameters", {}).get("properties", {})
        }
        for call in parsed.tool_calls:
            arguments = dict(call.arguments)
            control_error = False
            if call.name in controlled:
                for field in (planning.COMPLETION_FIELD, planning.INDEPENDENT_FIELD):
                    if field in arguments and type(arguments[field]) is not bool:
                        control_error = True
                    arguments.pop(field, None)
            valid = not bool(call.error or control_error)
            if valid:
                try:
                    registry.validate_arguments(call.name, arguments)
                except Exception:
                    valid = False
            self.tools.append(
                {
                    "name": _code(call.name) or "unknown_tool",
                    "call_id": _code(call.id),
                    "arguments_valid": valid,
                    "argument_fingerprint": _fingerprint(arguments),
                    "status": "pending",
                    "error_code": _code(call.error),
                    "duration_ms": None,
                    "result_count": None,
                }
            )

    def tool(self, name, arguments):
        fingerprint = _fingerprint(arguments)
        for row in self.tools:
            if (
                row["name"] == name
                and row["argument_fingerprint"] == fingerprint
                and row["status"] == "pending"
            ):
                return row
        row = {
            "name": _code(name) or "unknown_tool",
            "call_id": None,
            "arguments_valid": None,
            "argument_fingerprint": fingerprint,
            "status": "pending",
            "error_code": None,
            "duration_ms": None,
            "result_count": None,
        }
        self.tools.append(row)
        return row

    def start_usage(self, phase):
        row = {"phase": phase, "input_tokens": None, "output_tokens": None}
        self.phases.append(row)
        return row

    def usage(self, row, metrics):
        row.update({field: _number(metrics.get(field)) for field in ("input_tokens", "output_tokens")})

    def finalize_tools(self):
        if self.prepared is None:
            return
        remaining = list(self.tools)
        for item in self.prepared.tool_calls:
            row = next(
                (row for row in remaining if row["call_id"] == _code(item.get("call_id")) and row["call_id"]),
                None,
            )
            if row is None:
                row = next((row for row in remaining if row["name"] == item["name"]), None)
            if row is None:
                row = {
                    "name": _code(item["name"]) or "unknown_tool",
                    "call_id": None,
                    "arguments_valid": None,
                    "argument_fingerprint": None,
                }
                self.tools.append(row)
            elif row in remaining:
                remaining.remove(row)
            failed = bool(item.get("error"))
            error = (_code(item.get("error")) or "tool_failed") if failed else None
            row.update(
                call_id=_code(item.get("call_id")),
                status=(
                    "rejected"
                    if item.get("execution_mode") == "rejected"
                    else "failed" if failed else "completed"
                ),
                error_code=error,
                duration_ms=_number(item.get("latency_ms")),
                result_count=_number(item.get("result_count")),
            )


class _Registry:
    """Keep the original validators, permission checks and shared execution locks."""

    def __init__(self, registry, frozen, capture):
        self._registry, self._descriptors, self._capture = registry, frozen["tool_definitions"], capture

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def describe(self):
        return deepcopy(self._descriptors)

    async def call(self, name, arguments, *, context=None):
        row = self._capture.tool(name, arguments)
        try:
            self._registry.validate_arguments(name, arguments)
            row["arguments_valid"] = True
        except Exception:
            row["arguments_valid"] = False
        started = time.perf_counter_ns()
        try:
            output, record = await self._registry.call(name, arguments, context=context)
            row.update(
                status="failed" if record.error else "completed",
                error_code=(_code(record.error) or "tool_failed") if record.error else None,
                result_count=_number(record.result_count),
            )
            self._capture.retrieval_observed = True
            if not record.error:
                self._capture.retrieved.extend(deepcopy(planning.tool_evidence(name, output)))
            return output, record
        except asyncio.CancelledError:
            row.update(status="cancelled", error_code="cancelled")
            raise
        except Exception as exc:
            row.update(status="failed", error_code=_code(getattr(exc, "category", type(exc).__name__)))
            raise
        finally:
            row["duration_ms"] = (time.perf_counter_ns() - started) / 1e6


class _Retriever:
    def __init__(self, retriever, capture):
        self._retriever, self._capture = retriever, capture

    def __getattr__(self, name):
        return getattr(self._retriever, name)

    def retrieve(self, *args, **kwargs):
        result = self._retriever.retrieve(*args, **kwargs)
        self._capture.retrieval_observed = True
        self._capture.retrieved.extend(deepcopy(result.get("candidates20") or []))
        self._capture.retrieved.extend(deepcopy(result.get("final_context") or []))
        return result


class _Model:
    def __init__(self, model, frozen, capture, registry=None):
        self._model, self._capture, self._registry = model, capture, registry
        replacements = {
            value: frozen["prompts"][name]
            for name, value in frozen["baseline_prompts"].items()
            if value != frozen["prompts"][name]
        }
        self._replacements = replacements
        self._pattern = (
            re.compile("|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)))
            if replacements
            else None
        )

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _replace(self, value):
        return (
            self._pattern.sub(lambda match: self._replacements[match.group()], value)
            if self._pattern
            else value
        )

    def _messages(self, messages):
        # Native reasoning and function outputs are passed through in memory only.
        return [
            (
                {**item, "content": self._replace(item["content"])}
                if item.get("role") == "system" and isinstance(item.get("content"), str)
                else item
            )
            for item in messages
        ]

    def _schemas(self, value):
        if isinstance(value, list):
            return [self._schemas(child) for child in value]
        if isinstance(value, dict):
            return {
                key: (
                    self._replace(child)
                    if key == "description" and isinstance(child, str)
                    else self._schemas(child)
                )
                for key, child in value.items()
            }
        return value

    async def plan(self, messages, **kwargs):
        kwargs["tools"] = self._schemas(kwargs.get("tools", []))
        usage = self._capture.start_usage("planning")
        try:
            parsed, replay, done = await self._model.plan(self._messages(messages), **kwargs)
        except Exception as exc:
            self._capture.usage(usage, getattr(exc, "metrics", {}))
            raise
        self._capture.usage(usage, done.metrics)
        self._capture.calls(parsed, self._registry, kwargs["tools"])
        return parsed, replay, done

    async def generate(self, messages, **kwargs):
        kwargs["tools"] = self._schemas(kwargs.get("tools", []))
        usage = self._capture.start_usage("planning")
        try:
            text, done = await self._model.generate(self._messages(messages), **kwargs)
        except Exception as exc:
            self._capture.usage(usage, getattr(exc, "metrics", {}))
            raise
        self._capture.usage(usage, done.metrics)
        self._capture.calls(HermesFunctionCallCodec().decode(text), self._registry, kwargs["tools"])
        return text, done

    async def stream(self, messages, **kwargs):
        usage = self._capture.start_usage("final")
        try:
            async with aclosing(self._model.stream(self._messages(messages), **kwargs)) as stream:
                async for item in stream:
                    if isinstance(item, ModelDone):
                        self._capture.usage(usage, item.metrics)
                    yield item
        except Exception as exc:
            self._capture.usage(usage, getattr(exc, "metrics", {}))
            raise


class _Runtime:
    def __init__(self, runtime, capture):
        self._runtime, self._capture = runtime, capture

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    async def prepare(self, *args, **kwargs):
        result = await self._runtime.prepare(*args, **kwargs)
        self._capture.prepared = result
        return result


class _EphemeralConversation:
    def get_recent_history(self, *args):
        return []

    def add_message(self, *args):
        return {"id": str(uuid4())}


def _evidence(capture, sources):
    selected = {str(row["chunk_id"]): row for row in sources if row.get("chunk_id")}
    if capture.prepared is not None:
        for rank, chunk in enumerate(capture.prepared.contexts, 1):
            selected.setdefault(str(chunk["chunk_id"]), {**chunk, "final_rank": rank})
    merged = {}
    for row in [*capture.retrieved, *selected.values()]:
        if not isinstance(row, dict) or not row.get("chunk_id"):
            continue
        chunk_id = str(row["chunk_id"])
        current = merged.setdefault(chunk_id, {})
        current.update(row)
    output = []
    for chunk_id, row in merged.items():
        final = selected.get(chunk_id, {})
        output.append(
            EvidenceInput(
                chunk_id=_text(chunk_id),
                title=_text(row["title"]) if row.get("title") else None,
                source_kind=_code(row.get("source_kind")),
                source_url=_safe_url(row.get("url")),
                excerpt=_text(row.get("text", "")),
                dense_rank=_number(row.get("dense_rank")),
                dense_score=_score(row.get("best_dense_score")),
                bm25_rank=_number(row.get("bm25_rank")),
                bm25_score=_score(row.get("best_bm25_score")),
                rrf_score=_score(row.get("rrf_score")),
                reranker_score=_score(row.get("reranker_score")),
                final_rank=_number(final.get("final_rank")),
                selected=chunk_id in selected,
                cited=bool(final.get("cited")) if sources else None,
            ).model_dump(mode="json")
        )
    return output


class ChatBenchmarkExecutor:
    def __init__(self, app):
        self.app = app

    async def execute(self, question_text: str, frozen: dict, cancel: threading.Event) -> dict:
        verify_configuration(self.app, frozen)
        payload = _payload(frozen, question_text)
        mode = ChatMode(frozen["chat_mode"])
        request = _request(self.app)
        runtime = copy(routes._runtime(request, mode, payload))
        capture = _Capture()
        if mode == ChatMode.CENTRAL:
            runtime.planning_reasoning_effort = frozen["planning_reasoning_effort"]
            runtime.tools = _Registry(runtime.tools, frozen, capture)
        else:
            runtime.retriever = _Retriever(runtime.retriever, capture)
        runtime.model = _Model(runtime.model, frozen, capture, getattr(runtime, "tools", None))
        observed = _Runtime(runtime, capture)
        view = routes._resolve_tools(observed, payload, mode)
        sources, done, failure = [], {}, {}
        started = time.perf_counter_ns()

        async def collect():
            nonlocal sources, done, failure
            async with aclosing(
                routes._execute(
                    payload,
                    request,
                    "eval-benchmark-owner",
                    _EphemeralConversation(),
                    observed,
                    mode,
                    RequestTrace(str(uuid4()), mode.value),
                    watch_disconnect=False,
                    tool_view=view,
                )
            ) as events:
                async for event, data in events:
                    if event == "sources":
                        sources = data["items"]
                    elif event == "done":
                        done = data
                    elif event == "error":
                        failure = data

        task = None
        if not cancel.is_set():
            task = asyncio.create_task(collect())
            try:
                while not task.done() and not cancel.is_set():
                    await asyncio.wait({task}, timeout=0.05)
                if cancel.is_set() and not task.done():
                    task.cancel()
                # A manager timeout can overlap cooperative cancellation. Do not
                # inject a second cancellation into the provider's close/finally.
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done() and not task.cancelling():
                    task.cancel()
                await asyncio.shield(asyncio.gather(task, return_exceptions=True))
                if not cancel.is_set():
                    raise
            except Exception as exc:
                failure = {
                    "category": _code(getattr(exc, "category", type(exc).__name__)) or "execution_failed"
                }
        capture.finalize_tools()
        succeeded = done.get("status") == "done" and not failure
        status = "completed" if succeeded else "cancelled" if cancel.is_set() else "failed"
        error = (
            None
            if succeeded
            else "cancelled" if cancel.is_set() else _code(failure.get("category")) or "execution_failed"
        )
        usage = {}
        for phase in ("planning", "final"):
            rows = [row for row in capture.phases if row["phase"] == phase]
            usage[f"{phase}_calls"] = len(rows)
            for field in ("input_tokens", "output_tokens"):
                usage[f"{phase}_{field}"] = (
                    sum(row[field] for row in rows)
                    if (rows or phase == "planning") and all(row[field] is not None for row in rows)
                    else None
                )
        totals = {
            field: (
                sum(usage[f"{phase}_{field}"] for phase in ("planning", "final"))
                if all(usage[f"{phase}_{field}"] is not None for phase in ("planning", "final"))
                else None
            )
            for field in ("input_tokens", "output_tokens")
        }
        prepared = capture.prepared
        planning_data = prepared.planning if prepared else {}
        diagnostics = {
            "tool_calls": capture.tools,
            "planning": {
                "exit_reason": _code(planning_data.get("exit_reason")),
                "action_rounds": _number(prepared.action_rounds) if prepared else None,
                "repeated_calls_avoided": _number(planning_data.get("repeats_avoided")),
                "tools_executed": _number(planning_data.get("tools_executed")),
            },
            "capture": {
                "tool_trace": mode == ChatMode.CENTRAL and (prepared is not None or bool(capture.tools)),
                "retrieved_evidence": capture.retrieval_observed,
                "final_context": prepared is not None,
            },
            "usage": {**usage, **totals},
        }
        trace = {
            "retrieval_backend": frozen["dense_backend"],
            "final_context_chunk_ids": [
                _text(str(row["chunk_id"])) for row in (prepared.contexts if prepared else [])
            ],
            "cited_chunk_ids": [_text(str(row["chunk_id"])) for row in sources if row.get("cited")],
            "tools": [
                {key: row.get(key) for key in ("name", "status", "duration_ms", "error_code")}
                for row in capture.tools
            ],
        }
        run = RunInput(
            experiment_id=frozen["experiment_id"],
            question_id="benchmark-item",
            answer=_text(done.get("answer", "")) if succeeded else "",
            completion_status=status,
            error_reason=error,
            latency_ms=(time.perf_counter_ns() - started) / 1e6,
            **totals,
            tool_calls=(
                len(capture.tools)
                if mode == ChatMode.CENTRAL and (prepared is not None or capture.tools)
                else None
            ),
            cost=None,
            evidence=_evidence(capture, sources),
            trace_captured=prepared is not None,
            trace=trace,
        ).model_dump(mode="json")
        run.pop("experiment_id")
        run.pop("question_id")
        return {**run, "diagnostics": diagnostics}
