import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import AttachmentTray from "./components/AttachmentTray";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import ChatSidebar, { SidebarOpenButton } from "./components/ChatSidebar";
import BrandMark from "./components/BrandMark";
import EmptyState from "./components/EmptyState";
import SourcesDrawer from "./components/SourcesDrawer";
import StatusIndicator from "./components/StatusIndicator";
import {
  createConversation,
  deleteAttachment,
  EVIDENCE_CONTRACT_FAILURE_MESSAGE,
  deleteConversation,
  getConversation,
  listConversations,
  streamChat,
  updateConversation,
  uploadAttachment,
} from "./services/api";
import { shouldShowDebugTrace } from "./services/debugTrace";
import { persistChatMode, readStoredChatMode } from "./config/chatModes";
import { useChatScroll } from "./hooks/useChatScroll";
import { normalizeUploadFile, validateAttachments } from "./services/attachments";
import "./App.css";

const THEME_STORAGE_KEY = "vn-history-theme";
const ACTIVE_STATUSES = new Set([
  "processing", "retrieval_started", "reranking", "generating", "validating", "validated", "streaming",
  "hybrid_retrieval", "hybrid_answering",
  "three_llm_research", "three_llm_evidence", "three_llm_answering",
  "central_loading", "central_analyzing", "central_tools", "central_answering",
]);
const SHOW_DEBUG_TRACE = shouldShowDebugTrace(import.meta.env);

function getInitialTheme() {
  const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (["dark", "light"].includes(savedTheme)) return savedTheme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function createLocalId(prefix) {
  const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${id}`;
}

function normalizeConversationList(payload) {
  if (Array.isArray(payload)) return payload;
  return payload?.items ?? payload?.conversations ?? [];
}

function normalizeConversationDetail(payload) {
  const conversation = payload?.conversation ?? payload ?? {};
  return {
    conversation,
    messages: payload?.messages ?? conversation.messages ?? [],
    attachments: payload?.attachments ?? conversation.attachments ?? [],
  };
}

function getSources(data) {
  if (Array.isArray(data)) return data;
  return data?.items ?? data?.sources ?? data?.final_context ?? data?.retrieval?.final_context ?? [];
}

function getLatestSources(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.sources?.length) return message.sources;
  }
  return [];
}

function App() {
  const [theme, setTheme] = useState(getInitialTheme);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 840);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeSourceIndex, setActiveSourceIndex] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [attachments, setAttachments] = useState([]);
  const [pendingUploads, setPendingUploads] = useState([]);
  const [sources, setSources] = useState([]);
  const [question, setQuestion] = useState("");
  const [inferenceMode, setInferenceMode] = useState(readStoredChatMode);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [conversationToDelete, setConversationToDelete] = useState(null);
  const [isDeletingConversation, setIsDeletingConversation] = useState(false);

  const abortControllerRef = useRef(null);
  const uploadBusyRef = useRef(false);
  const cancelledUploadsRef = useRef(new Set());
  const previewUrlsRef = useRef(new Map());
  const [uploadBusy, setUploadBusy] = useState(false);
  useEffect(() => {
    const urls = previewUrlsRef.current;
    return () => { for (const url of urls.values()) URL.revokeObjectURL(url); urls.clear(); };
  }, []);
  const releasePreview = (id) => {
    const url = previewUrlsRef.current.get(id);
    if (url) URL.revokeObjectURL(url);
    previewUrlsRef.current.delete(id);
  };
  const clearPreviews = () => {
    for (const id of previewUrlsRef.current.keys()) releasePreview(id);
  };
  const refreshAttachments = (items) => setAttachments((current) => items.map((item) => ({
    ...item, preview_url: current.find((old) => old.id === item.id)?.preview_url,
  })));
  const { scrollerRef, contentRef, onScroll, followLatest } = useChatScroll(messages, status);
  const isRunning = ACTIVE_STATUSES.has(status);
  const isUploading = uploadBusy;
  const readyAttachments = attachments.filter((item) => item.status === "ready");

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations],
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    persistChatMode(inferenceMode);
  }, [inferenceMode]);

  useEffect(() => {
    const controller = new AbortController();

    async function bootstrap() {
      try {
        const payload = await listConversations({ signal: controller.signal });
        const items = normalizeConversationList(payload);
        setConversations(items);

        if (items.length > 0) {
          const detailPayload = await getConversation(items[0].id, { signal: controller.signal });
          const detail = normalizeConversationDetail(detailPayload);
          setActiveConversationId(items[0].id);
          setMessages(detail.messages);
          setAttachments(detail.attachments);
          setSources(getLatestSources(detail.messages));
        }
      } catch (requestError) {
        if (requestError.name !== "AbortError") {
          console.error(requestError);
          setError(requestError.message || "Không thể kết nối tới backend.");
        }
      } finally {
        if (!controller.signal.aborted) setIsLoadingConversations(false);
      }
    }

    bootstrap();
    return () => controller.abort();
  }, []);

  const updateMessage = (messageId, updater) => {
    setMessages((current) => current.map((message) => {
      if (message.id !== messageId) return message;
      return typeof updater === "function" ? updater(message) : { ...message, ...updater };
    }));
  };

  const refreshConversations = async () => {
    const payload = await listConversations();
    const items = normalizeConversationList(payload);
    setConversations(items);
    return items;
  };

  const loadConversation = async (conversationId) => {
    setIsLoadingConversation(true);
    setError("");

    try {
      const payload = await getConversation(conversationId);
      const detail = normalizeConversationDetail(payload);
      clearPreviews();
      followLatest();
      setActiveConversationId(conversationId);
      setMessages(detail.messages);
      setAttachments(detail.attachments);
      setSources(getLatestSources(detail.messages));
      setStatus("idle");
      return detail;
    } finally {
      setIsLoadingConversation(false);
    }
  };

  const createNewConversation = async ({ preserveDraft = false } = {}) => {
    if (isRunning) return null;

    const payload = await createConversation({ title: null });
    const conversation = payload?.conversation ?? payload;
    if (!conversation?.id) throw new Error("Backend không trả về conversation ID.");

    setConversations((current) => [conversation, ...current.filter((item) => item.id !== conversation.id)]);
    setActiveConversationId(conversation.id);
    setMessages([]);
    if (!preserveDraft) clearPreviews();
    setAttachments([]);
    setSources([]);
    if (!preserveDraft) setQuestion("");
    setStatus("idle");
    return conversation;
  };

  const ensureActiveConversation = async (options) => {
    if (activeConversationId) return activeConversationId;
    const conversation = await createNewConversation(options);
    return conversation.id;
  };

  const handleSelectConversation = async (conversationId) => {
    if (isRunning || uploadBusyRef.current || conversationId === activeConversationId) return;

    try {
      await loadConversation(conversationId);
      if (window.matchMedia("(max-width: 839px)").matches) setSidebarOpen(false);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || "Không thể tải cuộc trò chuyện.");
    }
  };

  const handleNewConversation = async () => {
    if (uploadBusyRef.current) return;
    try {
      await createNewConversation();
      if (window.matchMedia("(max-width: 839px)").matches) setSidebarOpen(false);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || "Không thể tạo cuộc trò chuyện mới.");
    }
  };

  const handleRenameConversation = async (conversation, title) => {
    try {
      const updated = await updateConversation(conversation.id, { title });
      setConversations((current) => current.map((item) => item.id === conversation.id ? { ...item, ...updated } : item));
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || "Không thể đổi tên cuộc trò chuyện.");
    }
  };

  const confirmDeleteConversation = async () => {
    if (!conversationToDelete || isRunning || uploadBusyRef.current) return;

    setIsDeletingConversation(true);
    setError("");

    try {
      await deleteConversation(conversationToDelete.id);
      const remaining = conversations.filter((item) => item.id !== conversationToDelete.id);
      setConversations(remaining);

      if (activeConversationId === conversationToDelete.id) {
        if (remaining.length > 0) {
          await loadConversation(remaining[0].id);
        } else {
          clearPreviews();
          setActiveConversationId(null);
          setMessages([]);
          setAttachments([]);
          setSources([]);
        }
      }

      setConversationToDelete(null);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || "Không thể xóa cuộc trò chuyện.");
    } finally {
      setIsDeletingConversation(false);
    }
  };

  const handleSubmit = async (event) => {
    event?.preventDefault();
    const trimmedQuestion = question.trim();
    if ((!trimmedQuestion && !readyAttachments.length) || isRunning || uploadBusyRef.current) return;

    setQuestion("");
    setError("");
    setStatus("processing");

    let conversationId;

    try {
      conversationId = await ensureActiveConversation();
    } catch (requestError) {
      setError(requestError.message || "Không thể tạo cuộc trò chuyện.");
      setStatus("error");
      return;
    }

    const userMessage = {
      id: createLocalId("user"),
      role: "user",
      content: trimmedQuestion,
      sources: readyAttachments.map((item) => ({ chunk_id: `attachment:${item.id}`, attachment_id: item.id,
        title: item.filename, source_kind: "attachment" })),
      status: "done",
      created_at: new Date().toISOString(),
    };
    const assistantMessageId = createLocalId("assistant");
    const assistantMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      sources: [],
      status: "processing",
      created_at: new Date().toISOString(),
    };

    // Creating the first conversation resets idle; keep the pending request visible.
    followLatest();
    setStatus("processing");
    setMessages((current) => [...current, userMessage, assistantMessage]);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let streamFailed = false;

    try {
      await streamChat({
        conversationId,
        question: trimmedQuestion,
        attachmentIds: readyAttachments.map((item) => item.id),
        mode: inferenceMode,
        finalK: 6,
        debug: SHOW_DEBUG_TRACE,
        signal: controller.signal,
        onEvent: ({ event: eventName, data }) => {
          if (eventName === "status") {
            const nextStatus = typeof data === "string" ? data : data?.stage ?? "processing";
            setStatus(nextStatus);
            updateMessage(assistantMessageId, { status: nextStatus, mode: data?.mode ?? inferenceMode });
            return;
          }

          if (eventName === "answer_delta") {
            const delta = typeof data === "string" ? data : data?.delta ?? "";
            setStatus("streaming");
            updateMessage(assistantMessageId, (message) => ({
              ...message,
              content: message.content + delta,
              status: "streaming",
            }));
            return;
          }

          if (eventName === "sources") {
            const nextSources = getSources(data);
            setSources(nextSources);
            updateMessage(assistantMessageId, { sources: nextSources });
            return;
          }

          if (eventName === "debug_trace" || eventName === "debug") {
            updateMessage(assistantMessageId, { debug_trace: data });
            return;
          }

          if (eventName === "error") {
            const message = typeof data === "string" ? data : data?.message ?? "Backend không thể hoàn tất yêu cầu.";
            const assistantErrorMessage = data?.type === "evidence_contract_error"
              ? EVIDENCE_CONTRACT_FAILURE_MESSAGE
              : "Không thể hoàn tất câu trả lời.";
            streamFailed = true;
            setError(message);
            setStatus("error");
            updateMessage(assistantMessageId, (current) => ({
              ...current,
              content: current.content || assistantErrorMessage,
              status: "error",
              debug_trace: data?.debug_trace ?? current.debug_trace,
            }));
            return;
          }

          if (eventName === "done") {
            setStatus(streamFailed ? "error" : "done");
            updateMessage(assistantMessageId, { status: streamFailed ? "error" : "done" });
          }
        },
      });

      if (!streamFailed) {
        setStatus("done");
        const [conversationPayload, detailPayload] = await Promise.all([
          listConversations(),
          getConversation(conversationId),
        ]);
        const detail = normalizeConversationDetail(detailPayload);
        setConversations(normalizeConversationList(conversationPayload));
        setMessages(detail.messages);
        refreshAttachments(detail.attachments);
        setSources(getLatestSources(detail.messages));
      }
    } catch (requestError) {
      if (requestError.name === "AbortError") {
        setStatus("cancelled");
        updateMessage(assistantMessageId, (message) => ({
          ...message,
          content: message.content || "Đã dừng tạo câu trả lời.",
          status: "cancelled",
        }));
      } else {
        console.error(requestError);
        setStatus("error");
        setError(requestError.message || "Không thể kết nối tới backend.");
        updateMessage(assistantMessageId, (message) => ({
          ...message,
          content: message.content || "Không thể hoàn tất câu trả lời.",
          status: "error",
        }));
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  const handleFilesSelected = async (selectedFiles, { uploadOrigin = "file" } = {}) => {
    if (isRunning || uploadBusyRef.current) {
      setError("Vui lòng chờ thao tác hiện tại hoàn tất trước khi thêm ảnh.");
      return;
    }
    const files = selectedFiles.map(normalizeUploadFile);
    const validationError = validateAttachments(files, attachments.length);
    if (validationError) { setError(validationError); return; }
    if (!files.length) return;
    uploadBusyRef.current = true;
    setUploadBusy(true);
    setError("");
    const queuedFiles = files.map((file) => {
      const id = createLocalId("upload");
      const preview_url = file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined;
      if (preview_url) previewUrlsRef.current.set(id, preview_url);
      return { id, name: file.name, type: file.type, size_bytes: file.size, status: "queued", file, preview_url };
    });
    setPendingUploads(queuedFiles);
    try {
      const conversationId = await ensureActiveConversation({ preserveDraft: true });
      for (const queued of queuedFiles) {
        if (cancelledUploadsRef.current.has(queued.id)) continue;
        setPendingUploads((current) => current.map((item) => item.id === queued.id ? { ...item, status: "uploading" } : item));
        try {
          const payload = await uploadAttachment(conversationId, queued.file, { uploadOrigin });
          const attachment = payload?.attachment ?? payload;
          // Let an in-flight upload finish so its server record can also be removed.
          if (cancelledUploadsRef.current.has(queued.id)) {
            await deleteAttachment(conversationId, attachment.id);
          } else {
            if (queued.preview_url) {
              previewUrlsRef.current.delete(queued.id);
              previewUrlsRef.current.set(attachment.id, queued.preview_url);
            }
            setAttachments((current) => [...current, { ...attachment, preview_url: queued.preview_url }]);
            if (attachment.status === "failed") setError(attachment.error || "Không thể đọc ảnh. Hãy thử ảnh rõ hơn.");
          }
        } catch (requestError) {
          releasePreview(queued.id);
          setError(requestError.message || `Không thể xử lý ${queued.name}.`);
          // Failed OCR records remain removable, but are never sent as ready IDs.
          const detail = normalizeConversationDetail(await getConversation(conversationId));
          refreshAttachments(detail.attachments);
        } finally {
          setPendingUploads((current) => current.filter((item) => item.id !== queued.id));
        }
      }
      await refreshConversations();
    } catch (requestError) {
      setError(requestError.message || "Không thể tải tài liệu.");
    } finally {
      for (const queued of queuedFiles) {
        releasePreview(queued.id);
        cancelledUploadsRef.current.delete(queued.id);
      }
      setPendingUploads([]);
      uploadBusyRef.current = false;
      setUploadBusy(false);
    }
  };

  const handleDeleteAttachment = async (attachmentId) => {
    if (isRunning) return;
    if (pendingUploads.some((item) => item.id === attachmentId)) {
      cancelledUploadsRef.current.add(attachmentId);
      releasePreview(attachmentId);
      setPendingUploads((current) => current.filter((item) => item.id !== attachmentId));
      return;
    }
    if (!activeConversationId) return;
    try {
      await deleteAttachment(activeConversationId, attachmentId);
      releasePreview(attachmentId);
      setAttachments((current) => current.filter((item) => item.id !== attachmentId));
      await refreshConversations();
    } catch (requestError) {
      setError(requestError.message || "Không thể xóa tài liệu.");
    }
  };

  const showMessageSources = (message, index = null) => {
    if (!message.sources?.length) return;
    setSources(message.sources);
    setActiveSourceIndex(index);
    setSourcesOpen(true);
  };

  const toggleTheme = () => setTheme((current) => current === "dark" ? "light" : "dark");
  const isEmpty = !isLoadingConversation && messages.length === 0;
  const composer = (
    <div className="composer-content">
      <AttachmentTray attachments={attachments} pendingUploads={pendingUploads} onDelete={handleDeleteAttachment} disabled={isRunning} />
      <ChatInput question={question} onQuestionChange={setQuestion} onSubmit={handleSubmit}
        onStop={() => abortControllerRef.current?.abort()} onFilesSelected={handleFilesSelected}
        mode={inferenceMode} onModeChange={setInferenceMode} isRunning={isRunning} isUploading={isUploading} hasAttachments={readyAttachments.length > 0} />
      <p className="composer-disclaimer">Lịch sử cần được nhìn từ nhiều nguồn. Hãy đối chiếu tư liệu khi cần.</p>
    </div>
  );

  return (
    <div className="app-shell">
      <ChatSidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        isOpen={sidebarOpen}
        isLoading={isLoadingConversations}
        isRunning={isRunning || isUploading}
        theme={theme}
        onClose={() => setSidebarOpen(false)}
        onNewConversation={handleNewConversation}
        onSelectConversation={handleSelectConversation}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={setConversationToDelete}
        onToggleTheme={toggleTheme}
      />

      {sidebarOpen && <button className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-label="Đóng thanh bên" />}

      <section className={`chat-workspace ${isEmpty ? "is-empty" : ""}`}>
        <header className="chat-header">
          <div className="chat-header-main">
            {!sidebarOpen && <SidebarOpenButton onClick={() => setSidebarOpen(true)} />}
            {!sidebarOpen && <BrandMark size={27} className="header-logo" label="Sử Việt AI" />}
            <div className="chat-title">
              <h1>{activeConversation?.title || "Sử Việt AI"}</h1>
              <span>
                {attachments.length > 0 ? (
                  <><Paperclip /> {attachments.length} tài liệu</>
                ) : (
                  "Không gian tìm hiểu lịch sử"
                )}
              </span>
            </div>
          </div>

          <div className="chat-header-actions">
            <button type="button" className="icon-button" onClick={toggleTheme} title="Đổi giao diện" aria-label={`Chuyển sang giao diện ${theme === "dark" ? "sáng" : "tối"}`}>
              {theme === "dark" ? <Sun /> : <Moon />}
            </button>
            <button
              type="button"
              className={`source-toggle ${sourcesOpen ? "is-active" : ""}`}
              onClick={() => { setActiveSourceIndex(null); setSourcesOpen((current) => !current); }}
              title="Nguồn tham khảo"
              aria-expanded={sourcesOpen}
            >
              {sourcesOpen ? <PanelRightClose /> : <PanelRightOpen />}
              <span>Nguồn</span>
              {sources.length > 0 && <b>{sources.length}</b>}
            </button>
          </div>
        </header>

        <main className="thread-scroll" id="conversation" aria-label="Cuộc trò chuyện" ref={scrollerRef} onScroll={onScroll}>
          <div className="thread-content" ref={contentRef}>
            {isLoadingConversation && <div className="thread-loading"><i /><i /><i /></div>}

            {isEmpty && <EmptyState onSuggestion={setQuestion}>{composer}</EmptyState>}

            {!isLoadingConversation && messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                isStreaming={message.role === "assistant" && message.status === "streaming"}
                onShowSources={(index) => showMessageSources(message, index)}
                enableDebugTrace={SHOW_DEBUG_TRACE}
              />
            ))}

            <StatusIndicator status={messages.at(-1)?.role === "assistant" && !messages.at(-1)?.content ? null : status} />
          </div>
        </main>

        {error && (
          <div className="error-toast" role="alert">
            <AlertCircle />
            <span>{error}</span>
            <button type="button" onClick={() => setError("")} aria-label="Đóng thông báo"><X /></button>
          </div>
        )}

        {!isEmpty && <footer className="composer-shell">{composer}</footer>}
      </section>

      <SourcesDrawer isOpen={sourcesOpen} sources={sources} activeIndex={activeSourceIndex} onClose={() => setSourcesOpen(false)} />

      {conversationToDelete && (
        <div className="dialog-backdrop" role="presentation">
          <div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <span className="dialog-icon"><Trash2 /></span>
            <h2 id="delete-title">Xóa cuộc trò chuyện?</h2>
            <p>“{conversationToDelete.title || "Cuộc trò chuyện mới"}” cùng tài liệu tạm thời sẽ bị xóa.</p>
            <div className="dialog-actions">
              <button type="button" onClick={() => setConversationToDelete(null)} disabled={isDeletingConversation}>Hủy</button>
              <button type="button" className="danger-button" onClick={confirmDeleteConversation} disabled={isDeletingConversation}>
                {isDeletingConversation ? "Đang xóa..." : "Xóa"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
