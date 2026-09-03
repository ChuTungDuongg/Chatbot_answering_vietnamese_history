import { StrictMode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  listConversations: vi.fn(), getConversation: vi.fn(), createConversation: vi.fn(),
  updateConversation: vi.fn(), deleteConversation: vi.fn(), uploadAttachment: vi.fn(),
  deleteAttachment: vi.fn(), streamChat: vi.fn(),
}));

const api = await import("../../src/services/api.js");
const { useChatSession } = await import("../../src/hooks/useChatSession.js");
const { useChatStream } = await import("../../src/hooks/useChatStream.js");
const { useAttachments } = await import("../../src/hooks/useAttachments.js");

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

const detail = (id) => ({
  messages: [{ id: `m-${id}`, role: "assistant", content: id, sources: [{ id: `s-${id}` }] }],
  attachments: [{ id: `att-${id}`, filename: `${id}.png`, status: "ready" }],
});

async function setup() {
  const hook = renderHook(({ mode }) => {
    const session = useChatSession();
    const shared = { dispatch: session.dispatch, activeConversationId: session.state.activeConversationId,
      isRunning: session.isRunning, ensureActiveConversation: session.ensureActiveConversation };
    const uploads = useAttachments({ ...shared, attachments: session.state.attachments });
    const stream = useChatStream({ ...shared, mode, attachments: session.state.attachments,
      isUploading: session.state.isUploading, showDebugTrace: true });
    return { session, uploads, stream };
  }, { initialProps: { mode: "hybrid" } });
  await waitFor(() => expect(hook.result.current.session.state.isLoadingConversations).toBe(false));
  return hook;
}

beforeEach(() => {
  vi.resetAllMocks();
  api.listConversations.mockResolvedValue([{ id: "a" }, { id: "b" }, { id: "c" }]);
  api.getConversation.mockImplementation(async (id) => detail(id));
  api.createConversation.mockResolvedValue({ id: "new" });
  api.deleteConversation.mockResolvedValue(null);
  api.deleteAttachment.mockResolvedValue(null);
  api.uploadAttachment.mockResolvedValue({ id: "uploaded", status: "ready" });
  api.streamChat.mockResolvedValue(undefined);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});

test("latest selection owns messages, sources, attachments and loading even when an older load fails", async () => {
  const { result } = await setup();
  const b = deferred(), c = deferred();
  api.getConversation.mockImplementation((id) => (id === "b" ? b : c).promise);
  let older, latest;
  act(() => {
    older = result.current.session.loadConversation("b").catch(() => {});
    latest = result.current.session.loadConversation("c");
  });
  await act(async () => { b.reject(new Error("stale failure")); await older; });
  expect(result.current.session.state.isLoadingConversation).toBe(true);
  expect(result.current.session.state.error).toBe("");
  await act(async () => { c.resolve(detail("c")); await latest; });
  expect(result.current.session.state).toMatchObject({ activeConversationId: "c", ...detail("c"), sources: [{ id: "s-c" }] });
});

test("a slow earlier selection cannot overwrite the last selected conversation", async () => {
  const { result } = await setup();
  const b = deferred();
  api.getConversation.mockImplementation((id) => id === "b" ? b.promise : Promise.resolve(detail(id)));
  let older;
  act(() => { older = result.current.session.loadConversation("b"); });
  await act(async () => { await result.current.session.loadConversation("c"); });
  await act(async () => { b.resolve(detail("b")); await older; });
  expect(result.current.session.state.activeConversationId).toBe("c");
  expect(result.current.session.state.messages).toEqual(detail("c").messages);
});

test("StrictMode ignores an aborted bootstrap even when the transport resolves late", async () => {
  const first = deferred();
  api.listConversations.mockImplementationOnce(() => first.promise);
  const { result } = renderHook(() => useChatSession(), { wrapper: StrictMode });
  await waitFor(() => expect(result.current.state.activeConversationId).toBe("a"));
  await act(async () => { first.resolve([{ id: "obsolete" }]); });
  expect(result.current.state.activeConversationId).toBe("a");
  expect(api.getConversation).not.toHaveBeenCalledWith("obsolete", expect.anything());
});

test("concurrent first-conversation requests share one creation", async () => {
  api.listConversations.mockResolvedValue([]);
  const { result } = await setup();
  const creation = deferred();
  api.createConversation.mockReturnValue(creation.promise);
  let one, two;
  act(() => { one = result.current.session.ensureActiveConversation(); two = result.current.session.ensureActiveConversation(); });
  expect(api.createConversation).toHaveBeenCalledTimes(1);
  await act(async () => { creation.resolve({ id: "new" }); expect(await one).toBe("new"); expect(await two).toBe("new"); });
});

test("double submit during creation sends once and captures the mode at submit", async () => {
  api.listConversations.mockResolvedValue([]);
  const { result, rerender } = await setup();
  const creation = deferred();
  api.createConversation.mockReturnValue(creation.promise);
  let sending;
  act(() => { sending = result.current.stream.submit("Question"); result.current.stream.submit("Question"); });
  rerender({ mode: "central" });
  expect(result.current.session.isRunning).toBe(true);
  await act(async () => { creation.resolve({ id: "new" }); await sending; });
  expect(api.streamChat).toHaveBeenCalledTimes(1);
  expect(api.streamChat.mock.calls[0][0]).toMatchObject({ mode: "hybrid", conversationId: "new" });
});

test("central_loading and post-done synchronization keep the request busy", async () => {
  const { result } = await setup();
  const stream = deferred(), sync = deferred();
  let onEvent;
  api.streamChat.mockImplementation((options) => { onEvent = options.onEvent; return stream.promise; });
  let sending;
  await act(async () => { sending = result.current.stream.submit("Question"); });
  act(() => onEvent({ event: "status", data: { stage: "central_loading" } }));
  expect(result.current.session.isRunning).toBe(true);
  api.getConversation.mockReturnValue(sync.promise);
  await act(async () => { onEvent({ event: "done" }); stream.resolve(); });
  expect(result.current.session.isRunning).toBe(true);
  await act(async () => { await result.current.stream.submit("Duplicate"); });
  expect(api.streamChat).toHaveBeenCalledTimes(1);
  await act(async () => { sync.resolve(detail("a")); await sending; });
  expect(result.current.session.isRunning).toBe(false);
});

test("switching conversations aborts the stream and ignores late answer, sources and debug", async () => {
  const { result } = await setup();
  const held = deferred();
  let request, sending;
  api.streamChat.mockImplementation((options) => { request = options; return held.promise; });
  await act(async () => { sending = result.current.stream.submit("Question"); });
  await act(async () => { await result.current.session.loadConversation("b"); });
  expect(request.signal.aborted).toBe(true);
  await act(async () => {
    request.onEvent({ event: "sources", data: [{ id: "wrong" }] });
    request.onEvent({ event: "debug", data: { wrong: true } });
    request.onEvent({ event: "answer_delta", data: "wrong" });
    held.resolve(); await sending;
  });
  expect(result.current.session.state).toMatchObject({ activeConversationId: "b", ...detail("b"), sources: [{ id: "s-b" }], status: "idle" });
});

test("stop settles the UI immediately and unmount cancels the next request", async () => {
  const { result, unmount } = await setup();
  const held = deferred();
  let request, sending;
  api.streamChat.mockImplementation((options) => { request = options; return held.promise; });
  await act(async () => { sending = result.current.stream.submit("Question"); });
  act(() => result.current.stream.stop());
  expect(result.current.session.isRunning).toBe(false);
  expect(result.current.session.state.status).toBe("cancelled");
  await act(async () => { held.resolve(); await sending; });
  const second = deferred();
  api.streamChat.mockImplementation((options) => { request = options; return second.promise; });
  await act(async () => { sending = result.current.stream.submit("Next"); });
  unmount();
  expect(request.signal.aborted).toBe(true);
  const calls = api.getConversation.mock.calls.length;
  await act(async () => { second.resolve(); await sending; });
  expect(api.getConversation).toHaveBeenCalledTimes(calls);
});

test("a new answer clears old sources and EOF without done settles running state", async () => {
  const { result } = await setup();
  const held = deferred();
  api.streamChat.mockReturnValue(held.promise);
  let sending;
  await act(async () => { sending = result.current.stream.submit("Next"); });
  expect(result.current.session.state.sources).toEqual([]);
  api.getConversation.mockResolvedValue({ messages: [{ id: "empty-sources", role: "assistant", sources: [] }], attachments: [] });
  await act(async () => { held.resolve(); await sending; });
  expect(result.current.session.isRunning).toBe(false);
  expect(result.current.session.state.status).toBe("done");
});

test("image-only sends include only ready IDs and attachment labels", async () => {
  const { result } = await setup();
  const held = deferred();
  api.streamChat.mockReturnValue(held.promise);
  let sending;
  await act(async () => { sending = result.current.stream.submit(""); });
  expect(api.streamChat).toHaveBeenCalledWith(expect.objectContaining({ question: "", attachmentIds: ["att-a"] }));
  expect(result.current.session.state.messages.at(-2).sources).toEqual([expect.objectContaining({ attachment_id: "att-a", title: "a.png" })]);
  await act(async () => { held.resolve(); await sending; });
});

test("clipboard upload preserves origin and preview; cancelling pending deletes the server ID", async () => {
  const { result } = await setup();
  const held = deferred();
  api.uploadAttachment.mockReturnValue(held.promise);
  let uploading;
  await act(async () => { uploading = result.current.uploads.upload([new File(["image"], "image.png", { type: "image/png" })], { uploadOrigin: "clipboard" }); });
  expect(api.uploadAttachment).toHaveBeenCalledWith("a", expect.any(File), expect.objectContaining({ uploadOrigin: "clipboard" }));
  const pending = result.current.session.state.pendingUploads[0];
  expect(pending.preview_url).toBe("blob:preview");
  await act(async () => { await result.current.uploads.remove(pending.id); });
  expect(api.deleteAttachment).not.toHaveBeenCalled();
  expect(result.current.session.state.isUploading).toBe(true);
  await act(async () => { await result.current.stream.submit("Wait for removal"); });
  expect(api.streamChat).not.toHaveBeenCalled();
  await act(async () => { held.resolve({ id: "uploaded", status: "ready" }); await uploading; });
  expect(api.deleteAttachment).toHaveBeenCalledWith("a", "uploaded");
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:preview");
  expect(result.current.session.state.isUploading).toBe(false);
});

test.each(["select", "new"])("upload completion cannot leak into another conversation after %s", async (action) => {
  const { result } = await setup();
  const held = deferred();
  api.uploadAttachment.mockReturnValue(held.promise);
  let uploading;
  await act(async () => { uploading = result.current.uploads.upload([new File(["pdf"], "file.pdf", { type: "application/pdf" })]); });
  await act(async () => {
    if (action === "select") await result.current.session.loadConversation("b");
    else await result.current.session.createNewConversation();
  });
  await act(async () => { held.resolve({ id: "uploaded", status: "ready" }); await uploading; });
  expect(result.current.session.state.attachments).toEqual(action === "select" ? detail("b").attachments : []);
  expect(result.current.session.state.pendingUploads).toEqual([]);
  expect(result.current.session.state.isUploading).toBe(false);
});

test("upload validation retains conversation count, declared MIME and empty-file rules", async () => {
  const { result } = await setup();
  for (const files of [
    Array.from({ length: 5 }, () => new File(["pdf"], "file.pdf", { type: "application/pdf" })),
    [new File(["svg"], "fake.png", { type: "image/svg+xml" })],
    [new File([], "empty.pdf", { type: "application/pdf" })],
  ]) {
    await act(async () => { await result.current.uploads.upload(files); });
    expect(api.uploadAttachment).not.toHaveBeenCalled();
    expect(result.current.session.state.error).not.toBe("");
  }
});
