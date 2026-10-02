import { useState } from "react";

export function useToolSteering(capabilities = {}) {
  const [excluded, setExcluded] = useState([]);
  const [selectedServers, setSelectedServers] = useState([]);
  const tools = (capabilities.tools ?? []).filter((tool) => tool.available);
  const servers = capabilities.mcp?.enabled ? capabilities.mcp.servers ?? [] : [];
  const allowedServers = servers.filter((server) => server.available && selectedServers.includes(server.id));
  const external = allowedServers.flatMap((server) => server.tools ?? []);
  const limit = capabilities.tool_policy?.max_mcp_tools ?? 8;
  const selected = [...tools, ...external].filter((tool) => !excluded.includes(tool.id));
  const steering = { mcp_enabled: allowedServers.length > 0,
    allowed_mcp_servers: allowedServers.map((server) => server.id),
    allowed_tools: selected.map((tool) => tool.id), mcp_failure_policy: "continue" };
  return { tools, servers, steering, limit,
    isAllowed: (id) => !excluded.includes(id),
    toggleTool: (id) => setExcluded((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]),
    toggleServer: (id) => {
      if (!servers.some((server) => server.id === id && server.available)) return;
      setSelectedServers((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id]);
    },
    overBudget: external.filter((tool) => !excluded.includes(tool.id)).length > limit,
    // Old/no capability API keeps built-in behavior, never enables MCP.
    payload: tools.length ? steering : undefined,
  };
}
