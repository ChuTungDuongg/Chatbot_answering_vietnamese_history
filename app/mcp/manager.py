"""Startup discovery, cached schemas and bounded, cancellable MCP execution."""

import asyncio
import json
import logging
from pathlib import Path
import time

from app.mcp.client import MCPConnection, sdk_client
from app.mcp.schemas import MCPConfig, MCPError, MCPToolError
from app.mcp.adapters import MCPToolAdapter, tool_name, validate_schema
from app.mcp.normalization import normalize, redact, clipped_redact

logger = logging.getLogger(__name__)


class MCPManager:
    def __init__(self, config=None, *, enabled=False, repo_root=None, factory=sdk_client):
        self.config = config or MCPConfig()
        self.enabled, self.repo_root, self.factory = enabled, Path(repo_root or Path.cwd()), factory
        self.connections, self.adapters, self.connection_ms, self.errors = {}, {}, {}, {}
        self.pending_calls = set()

    @classmethod
    def from_settings(cls, settings):
        from app.config import REPO_ROOT
        if not settings.mcp_enabled: return cls(repo_root=REPO_ROOT)
        path = settings.mcp_config_path
        if not path.is_absolute(): path = REPO_ROOT / path
        try:
            if path.stat().st_size > 262144: raise ValueError()
            config = MCPConfig.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            raise MCPError("MCP config thiếu hoặc không hợp lệ; kiểm tra MCP_CONFIG_PATH.") from None
        return cls(config, enabled=True, repo_root=REPO_ROOT)

    async def start(self):
        if not self.enabled: return
        async def load(server, config):
            if not config.enabled: return
            connection = MCPConnection(config, self.repo_root, self.factory)
            start = time.perf_counter_ns()
            try:
                await connection.start()
                adapters = []
                for tool in connection.tools:
                    if tool.name not in config.allowed_tools: continue
                    annotations = getattr(tool, "annotations", None)
                    if config.read_only_only and not (annotations and annotations.read_only_hint is True): continue
                    schema_text = json.dumps(tool.input_schema)
                    if any(secret in schema_text or secret in tool.name for secret in connection.secrets):
                        raise MCPError("MCP schema chứa thông tin riêng; không expose cho planner.")
                    output_schema = getattr(tool, "output_schema", None)
                    if output_schema is not None:
                        validate_schema(output_schema)
                    adapter = MCPToolAdapter(self, server, tool.name,
                        clipped_redact(tool.description or "External evidence", 300, connection.secrets),
                        tool.input_schema)
                    adapters.append(adapter)
                names = [adapter.name for adapter in adapters]
                if len(set(names)) != len(names) or set(names).intersection(self.adapters):
                    raise MCPError("MCP tool namespace collision.")
                self.connections[server] = connection
                self.connection_ms[server] = (time.perf_counter_ns()-start)/1e6
                for adapter in adapters:
                    self.adapters[adapter.name] = adapter
            except BaseException as exc:
                await connection.close()
                if isinstance(exc, asyncio.CancelledError): raise
                self.errors[server] = type(exc).__name__
                logger.warning("MCP unavailable: server=%s type=%s", server, type(exc).__name__)
                if config.required: raise MCPError(f"Required MCP server '{server}' không khả dụng.") from None
        tasks = [asyncio.create_task(load(server, config)) for server, config in self.config.servers.items()]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()
            raise

    def register(self, registry):
        for adapter in self.adapters.values(): registry.register(adapter)

    def capabilities(self):
        return {"enabled": self.enabled, "servers": [
            {"id": server, "label": redact(config.label, self.connections[server].secrets if server in self.connections else ()), "available": server in self.connections and self.connections[server].client is not None,
             "tools": [{"id": tool.name, "label": tool.remote_name[:64]} for tool in self.adapters.values() if tool.server == server]}
            for server, config in self.config.servers.items() if config.enabled]}

    async def execute(self, server, tool, arguments, *, context):
        name = tool_name(server, tool)
        if name not in self.adapters or context.allowed_tools is None or name not in context.allowed_tools:
            raise MCPToolError("MCP tool không được phép trong request này.")
        connection = self.connections.get(server)
        if not connection or connection.client is None: raise MCPToolError("MCP server hiện không khả dụng.")
        config = self.config.servers[server]
        label = redact(config.label, connection.secrets)
        callback = context.progress
        def emit(state, latency=None):
            if callback:
                callback({"stage": f"tool:{name}",
                    "state": state, "message": f"Tra cứu {label}", "provider": "mcp", "server": server,
                    "tool": tool, **({"latency_ms": latency} if latency is not None else {})})
        start = time.perf_counter_ns()
        emit("started")
        call = asyncio.create_task(connection.client.call_tool(tool, arguments))
        async def wait_cancel():
            while context.cancel is not None and not context.cancel.is_set(): await asyncio.sleep(.05)
            if context.cancel is None: await asyncio.Future()
        cancellation = asyncio.create_task(wait_cancel())
        success, count = False, None
        try:
            done, _ = await asyncio.wait({call, cancellation}, timeout=config.tool_timeout_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if cancellation in done: raise asyncio.CancelledError()
            if call not in done: raise MCPToolError(f"{label} không phản hồi trong thời hạn.")
            result = normalize(call.result(), server=server, tool=tool, label=label,
                               limit=config.max_result_chars, secrets=connection.secrets)
            success, count = True, len(result)
            emit("completed", (time.perf_counter_ns()-start)/1e6)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            emit("failed")
            if isinstance(exc, MCPToolError): raise
            raise MCPToolError(f"{label} không thể hoàn tất tra cứu.") from None
        finally:
            for task in (call, cancellation):
                if not task.done(): task.cancel()
            # A non-cooperative transport must not defeat the request deadline.
            # Detached cleanup cannot emit events or add evidence to this request.
            _, pending = await asyncio.wait({call, cancellation}, timeout=.2)
            for task in (call, cancellation):
                if task.done() and not task.cancelled(): task.exception()
            for task in pending:
                self.pending_calls.add(task)
                def consume(completed):
                    self.pending_calls.discard(completed)
                    if not completed.cancelled(): completed.exception()
                task.add_done_callback(consume)
            context.mcp_metrics.append({"provider": "mcp", "server": server, "tool": tool,
                "mcp_connection_ms": self.connection_ms[server], "mcp_tool_ms": (time.perf_counter_ns()-start)/1e6,
                "result_count": count, "success": success})

    async def close(self):
        await asyncio.gather(*(connection.close() for connection in self.connections.values()), return_exceptions=True)
        for task in self.pending_calls: task.cancel()
        if self.pending_calls:
            await asyncio.wait(self.pending_calls, timeout=.5)
        self.connections.clear(); self.adapters.clear()
