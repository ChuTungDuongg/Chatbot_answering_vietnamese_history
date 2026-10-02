"""One SDK client/transport per server, owned and closed by the same task."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import os
import logging
from pathlib import Path
import sys
from urllib.parse import urlparse

from app.mcp.schemas import MCPError


def referenced_environment(references):
    values = {}
    for name, reference in references.items():
        value = os.getenv(reference)
        if not value:
            raise MCPError("MCP server thiếu biến môi trường đã cấu hình.")
        values[name] = value
    return values


@asynccontextmanager
async def sdk_client(config, repo_root: Path):
    # SDK imports and network/process work happen only when explicitly enabled.
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
    # Library transport diagnostics may include endpoint URLs or raw envelopes.
    # Application logging below this boundary records only server ID/error type.
    for namespace in ("mcp", "httpx2", "httpcore2"):
        library_logger = logging.getLogger(namespace)
        library_logger.addHandler(logging.NullHandler())
        library_logger.propagate = False
        library_logger.setLevel(logging.CRITICAL)
    async with AsyncExitStack() as stack:
        secrets = []
        if config.transport == "stdio":
            env = referenced_environment(config.env_refs)
            secrets.extend(env.values())
            cwd = config.cwd or repo_root
            if not cwd.is_absolute(): cwd = repo_root / cwd
            command = sys.executable if config.command == "python" else config.command
            errlog = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            target = stdio_client(StdioServerParameters(command=command, args=config.args, env=env,
                                                        cwd=cwd), errlog=errlog)
        else:
            import httpx2
            url = os.getenv(config.url_env) if config.url_env else config.url
            parsed = urlparse(url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise MCPError("MCP HTTP endpoint chưa được cấu hình hợp lệ.")
            headers = referenced_environment(config.headers_env)
            secrets.extend([url, parsed.hostname, *headers.values()])
            secrets.extend(value.split()[-1] for value in headers.values() if " " in value)
            http = await stack.enter_async_context(httpx2.AsyncClient(headers=headers,
                timeout=httpx2.Timeout(config.tool_timeout_seconds, connect=config.connect_timeout_seconds),
                follow_redirects=False))
            target = streamable_http_client(url, http_client=http)
        client = await stack.enter_async_context(Client(target, read_timeout_seconds=config.tool_timeout_seconds,
                                                       cache=None))
        yield client, tuple(value for value in secrets if len(value) >= 4)


class MCPConnection:
    def __init__(self, config, repo_root, factory=sdk_client):
        self.config, self.repo_root, self.factory = config, repo_root, factory
        self.client = None
        self.secrets = ()
        self.tools = []
        self.task = None
        self.stop = asyncio.Event()

    async def start(self):
        ready = asyncio.get_running_loop().create_future()
        async def own_session():
            try:
                async with self.factory(self.config, self.repo_root) as (client, secrets):
                    self.client, self.secrets = client, secrets
                    cursor = None
                    while True:
                        page = await client.list_tools(cursor=cursor)
                        self.tools.extend(page.tools)
                        if len(self.tools) > 256: raise MCPError("MCP discovery vượt giới hạn công cụ.")
                        cursor = page.next_cursor
                        if not cursor: break
                    ready.set_result(None)
                    await self.stop.wait()
            except BaseException:
                if not ready.done(): ready.set_exception(MCPError("Không kết nối được MCP server."))
            finally:
                self.client = None
        self.task = asyncio.create_task(own_session())
        try:
            await asyncio.wait_for(asyncio.shield(ready), self.config.connect_timeout_seconds)
        except BaseException as exc:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            if ready.done() and not ready.cancelled(): ready.exception()
            else: ready.cancel()
            if isinstance(exc, asyncio.CancelledError): raise
            raise MCPError("Không kết nối được MCP server.") from None

    async def close(self):
        self.stop.set()
        if self.task:
            try: await asyncio.wait_for(self.task, 5)
            except (TimeoutError, asyncio.CancelledError):
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
