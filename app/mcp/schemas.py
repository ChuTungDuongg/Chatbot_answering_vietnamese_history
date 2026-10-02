"""Trusted deployment definitions and unprivileged request steering."""

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ServerID = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]
ToolID = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_.:-]{1,64}$")]


class ToolSteering(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mcp_enabled: bool = False
    allowed_mcp_servers: list[ServerID] = Field(default_factory=list, max_length=16)
    # None means all currently configured built-ins and explicitly selected MCP servers.
    # [] means no tools; it never means unrestricted access.
    allowed_tools: list[ToolID] | None = Field(default=None, max_length=64)
    mcp_failure_policy: Literal["continue", "fail"] = "continue"


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(default="MCP", max_length=64)
    enabled: bool = False
    required: bool = False
    transport: Literal["stdio", "http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list, max_length=32)
    cwd: Path | None = None
    # Environment mapping: destination variable/header -> existing env variable name.
    env_refs: dict[str, str] = Field(default_factory=dict)
    headers_env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    url_env: str | None = None
    allowed_tools: list[str] = Field(default_factory=list, max_length=64)
    read_only_only: bool = True
    connect_timeout_seconds: float = Field(default=15, ge=.1, le=120)
    tool_timeout_seconds: float = Field(default=30, ge=.1, le=300)
    max_result_chars: int = Field(default=6000, ge=256, le=16000)

    @model_validator(mode="after")
    def validate_transport(self):
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio requires a server-side command")
        if self.transport == "http" and not (self.url or self.url_env):
            raise ValueError("http requires url or url_env")
        if self.url:
            parsed = urlparse(self.url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Invalid MCP HTTP endpoint; credentials belong in env references")
        return self


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    servers: dict[ServerID, MCPServerConfig] = Field(default_factory=dict, max_length=16)


class MCPError(RuntimeError):
    """Only fixed, sanitized messages may cross this application boundary."""


class MCPToolError(MCPError):
    pass
