import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import ToolSteering from "../../src/components/ToolSteering.jsx";
import ChatInput from "../../src/components/ChatInput.jsx";
import RetrievalProgress from "../../src/components/RetrievalProgress.jsx";
import RetrievedChunks from "../../src/components/RetrievedChunks.jsx";
import { useToolSteering } from "../../src/hooks/useToolSteering.js";
import { chatSessionReducer, initialChatSessionState } from "../../src/state/chatSessionReducer.js";

const capabilities = { tools: [{ id: "search_history", label: "Kho sử liệu", available: true },
  { id: "search_wikipedia", label: "Wikipedia", available: true }],
  mcp: { enabled: true, servers: [
    { id: "research", label: "Research MCP", available: true, tools: [{ id: "mcp__research__lookup", label: "lookup" }] },
    { id: "offline", label: "Offline MCP", available: false, tools: [] },
  ] }, tool_policy: { max_mcp_tools: 8 } };

test("secure default exposes built-ins only and capabilities normalize unavailable MCP", () => {
  const hook = renderHook(useToolSteering, { initialProps: capabilities });
  expect(hook.result.current.payload).toMatchObject({ mcp_enabled: false, allowed_mcp_servers: [],
    allowed_tools: ["search_history", "search_wikipedia"] });
  act(() => hook.result.current.toggleServer("offline"));
  expect(hook.result.current.payload.mcp_enabled).toBe(false);
  act(() => hook.result.current.toggleServer("research"));
  expect(hook.result.current.payload).toMatchObject({ mcp_enabled: true, allowed_mcp_servers: ["research"] });
  expect(hook.result.current.payload.allowed_tools).toContain("mcp__research__lookup");
  hook.rerender({ ...capabilities, mcp: { ...capabilities.mcp, servers: capabilities.mcp.servers.map((server) => ({ ...server, available: false })) } });
  expect(hook.result.current.payload.mcp_enabled).toBe(false);
  expect(hook.result.current.payload.allowed_tools).not.toContain("mcp__research__lookup");
});

test("server disabled cannot be unlocked and tool deselection restricts payload", () => {
  const hook = renderHook(useToolSteering, { initialProps: { ...capabilities, mcp: { enabled: false, servers: capabilities.mcp.servers } } });
  act(() => hook.result.current.toggleServer("research"));
  expect(hook.result.current.servers).toEqual([]);
  act(() => hook.result.current.toggleTool("search_wikipedia"));
  expect(hook.result.current.payload.allowed_tools).toEqual(["search_history"]);
  act(() => hook.result.current.toggleTool("search_history"));
  expect(hook.result.current.payload.allowed_tools).toEqual([]);
});

test("Tools popover uses advertised labels and disables unavailable server", () => {
  function Harness() { return <ToolSteering control={useToolSteering(capabilities)} />; }
  render(<Harness />);
  fireEvent.click(screen.getByRole("button", { name: "Công cụ Central" }));
  expect(screen.getByRole("checkbox", { name: /Offline MCP/ })).toBeDisabled();
  expect(screen.queryByRole("checkbox", { name: "lookup" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("checkbox", { name: "Research MCP" }));
  expect(screen.getByRole("checkbox", { name: "lookup" })).toBeChecked();
  fireEvent.click(screen.getByRole("checkbox", { name: "lookup" }));
  expect(screen.getByRole("checkbox", { name: "lookup" })).not.toBeChecked();
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.getByRole("button", { name: "Công cụ Central" })).toHaveFocus();
  expect(screen.queryByRole("group", { name: "Công cụ Central Agent" })).not.toBeInTheDocument();
});

test("composer shows MCP controls only in Central and preserves attachments/send/stop", () => {
  const hook = renderHook(useToolSteering, { initialProps: capabilities });
  const props = { question: "", onQuestionChange: vi.fn(), onSubmit: vi.fn(), onFilesSelected: vi.fn(),
    onModeChange: vi.fn(), retrievalBackend: "faiss", availableBackends: ["faiss"], toolControl: hook.result.current };
  const view = render(<ChatInput {...props} mode="hybrid" />);
  expect(screen.queryByRole("button", { name: "Công cụ Central" })).not.toBeInTheDocument();
  view.rerender(<ChatInput {...props} mode="central" />);
  expect(screen.getByRole("button", { name: "Công cụ Central" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Tải PDF hoặc hình ảnh" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Gửi câu hỏi" })).toBeInTheDocument();
  view.rerender(<ChatInput {...props} mode="central" isRunning />);
  expect(screen.getByRole("button", { name: "Công cụ Central" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Dừng tạo câu trả lời" })).toBeInTheDocument();
});

test("tool schema budget prevents oversized selection", () => {
  const hook = renderHook(useToolSteering, { initialProps: { ...capabilities, tool_policy: { max_mcp_tools: 0 } } });
  act(() => hook.result.current.toggleServer("research"));
  expect(hook.result.current.overBudget).toBe(true);
  act(() => hook.result.current.toggleTool("mcp__research__lookup"));
  expect(hook.result.current.overBudget).toBe(false);
});

test.each(["STREAM_ABORTED", "STREAM_ERROR", "STREAM_DONE"])("MCP progress is request scoped and settles on %s", (type) => {
  const state = { ...initialChatSessionState, activeConversationId: "c1", messages: [{ id: "a1", content: "", pipeline: [] }] };
  const event = { stage: "tool:mcp__research__lookup", state: "started", message: "Tra cứu Research MCP",
    provider: "mcp", server: "research", tool: "lookup", request_id: "r1" };
  const stale = chatSessionReducer(state, { type: "STREAM_STATUS", scopeId: "c2", messageId: "a1", progress: event });
  expect(stale.messages[0].pipeline).toEqual([]);
  const updated = chatSessionReducer(state, { type: "STREAM_STATUS", scopeId: "c1", messageId: "a1", progress: event });
  expect(updated.messages[0].pipeline[0]).toMatchObject({ provider: "mcp", server: "research", requestId: "r1" });
  const finished = chatSessionReducer(updated, { type, scopeId: "c1", messageId: "a1" });
  expect(finished.messages[0].pipeline[0].state).not.toBe("started");
});

test("MCP progress stays compact after first token and sources preserve external kind", () => {
  const pipeline = [{ stage: "tool:mcp__research__lookup", state: "completed", label: "Tra cứu Research MCP", provider: "mcp" }];
  render(<RetrievalProgress pipeline={pipeline} content="Answer" status="done" />);
  expect(screen.getByRole("status")).toHaveTextContent("Research MCP");
  expect(screen.getByRole("status")).not.toHaveAttribute("open");
  render(<RetrievedChunks sources={[{ chunk_id: "mcp:research:123", source_kind: "mcp", title: "Research", text: "Evidence", url: "https://example.org" }]} />);
  expect(screen.getByText("Nguồn MCP")).toBeInTheDocument();
  expect(screen.queryByText("Kho sử liệu")).not.toBeInTheDocument();
});
