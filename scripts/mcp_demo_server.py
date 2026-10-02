"""Read-only MCP development server. Never enabled by application defaults."""

import argparse
import asyncio

from mcp import types
from mcp.server import MCPServer

server = MCPServer("History research demo")


@server.tool(annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def lookup(query: str) -> dict:
    """Return a small synthetic external evidence item for development/tests."""
    return {"text": f"Demo evidence for {query[:200]}; synthetic data, not historical verification.",
            "url": "https://example.org/research-demo", "source_id": "demo-research"}


@server.tool(annotations=types.ToolAnnotations(readOnlyHint=True))
async def slow_lookup(query: str, delay: float = 1.0) -> dict:
    """Exercise cancellation and deadlines in development/tests."""
    await asyncio.sleep(min(max(delay, 0), 10))
    return await lookup(query)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8383)
    args = parser.parse_args()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        import uvicorn
        uvicorn.run(server.streamable_http_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
