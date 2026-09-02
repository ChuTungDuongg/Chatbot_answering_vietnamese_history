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
import LogoMark from "./components/LogoMark";
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
import "./App.css";

const THEME_STORAGE_KEY = "vn-history-theme";
const ACTIVE_STATUSES = new Set([
  "processing", "retrieval_started", "reranking", "generating", "validating", "validated", "streaming",
  "hybrid_retrieval", "hybrid_answering",
  "three_llm_research", "three_llm_evidence", "three_llm_answering",
  "central_loading", "central_analyzing", "central_tools", "central_answering",
]);
const ALLOWED_MIME_TYPES = new Set(["application/pdf", "image/png", "image/jpeg", "image/webp"]);
const MIME_BY_EXTENSION = { pdf: "application/pdf", png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp" };
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const MAX_FILES_PER_UPLOAD = 5;
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

function normalizeUploadFile(file) {
  if (ALLOWED_MIME_TYPES.has(file.type)) return file;

  const extension = file.name.split(".").pop()?.toLowerCase();
  const inferredType = MIME_BY_EXTENSION[extension];
  if (!inferredType) return file;

  return new File([file], file.name, { type: inferredType, lastModified: file.lastModified });
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
  const { scrollerRef, contentRef, onScroll, followLatest } = useChatScroll(messages, status);
  const isRunning = ACTIVE_STATUSES.has(status);
  const isUploading = pendingUploads.length > 0;

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

  const createNewConversation = async () => {
    if (isRunning) return null;

    const payload = await createConversation({ title: null });
    const conversation = payload?.conversation ?? payload;
    if (!conversation?.id) throw new Error("Backend không trả về conversation ID.");

    setConversations((current) => [conversation, ...current.filter((item) => item.id !== conversation.id)]);
    setActiveConversationId(conversation.id);
    setMessages([]);
    setAttachments([]);
    setSources([]);
    setQuestion("");
    setStatus("idle");
    return conversation;
  };

  const ensureActiveConversation = async () => {
    if (activeConversationId) return activeConversationId;
    const conversation = await createNewConversation();
    return conversation.id;
  };

  const handleSelectConversation = async (conversationId) => {
    if (isRunning || conversationId === activeConversationId) return;

    try {
      await loadConversation(conversationId);
      if (window.matchMedia("(max-width: 839px)").matches) setSidebarOpen(false);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || "Không thể tải cuộc trò chuyện.");
    }
  };

  const handleNewConversation = async () => {
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
    if (!conversationToDelete || isRunning) return;

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
    if (!trimmedQuestion || isRunning) return;

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
      sources: [],
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
        setAttachments(detail.attachments);
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

  const handleFilesSelected = async (selectedFiles) => {
    const files = selectedFiles.slice(0, MAX_FILES_PER_UPLOAD).map(normalizeUploadFile);
    const invalidFile = files.find((file) => !ALLOWED_MIME_TYPES.has(file.type));
    const oversizedFile = files.find((file) => file.size > MAX_FILE_SIZE);

    if (selectedFiles.length > MAX_FILES_PER_UPLOAD) {
      setError(`Mỗi lần chỉ có thể tải tối đa ${MAX_FILES_PER_UPLOAD} file.`);
      return;
    }
    if (invalidFile) {
      setError(`Không hỗ trợ định dạng của ${invalidFile.name}. Chỉ nhận PDF, PNG, JPEG và WebP.`);
      return;
    }
    if (oversizedFile) {
      setError(`${oversizedFile.name} vượt quá giới hạn 20 MB.`);
      return;
    }

    setError("");

    let conversationId;
    try {
      conversationId = await ensureActiveConversation();
    } catch (requestError) {
      setError(requestError.message || "Không thể tạo cuộc trò chuyện.");
      return;
    }

    const queuedFiles = files.map((file) => ({
      id: createLocalId("upload"),
      name: file.name,
      type: file.type,
      size_bytes: file.size,
      status: "queued",
      file,
    }));
    setPendingUploads((current) => [...current, ...queuedFiles]);

    for (const queuedFile of queuedFiles) {
      setPendingUploads((current) => current.map((item) =>
        item.id === queuedFile.id ? { ...item, status: "processing" } : item,
      ));

      try {
        const payload = await uploadAttachment(conversationId, queuedFile.file);
        const attachment = payload?.attachment ?? payload;
        setAttachments((current) => [
          ...current.filter((item) => item.id !== attachment.id),
          attachment,
        ]);
      } catch (requestError) {
        console.error(requestError);
        setError(requestError.message || `Không thể xử lý ${queuedFile.name}.`);
      } finally {
        setPendingUploads((current) => current.filter((item) => item.id !== queuedFile.id));
      }
    }

    try {
      const [conversationPayload, detailPayload] = await Promise.all([
        listConversations(),
        getConversation(conversationId),
      ]);
      const detail = normalizeConversationDetail(detailPayload);
      setConversations(normalizeConversationList(conversationPayload));
      setAttachments(detail.attachments);
    } catch (refreshError) {
      console.warn("Could not refresh attachments", refreshError);
    }
  };

  const handleDeleteAttachment = async (attachmentId) => {
    if (!activeConversationId || isRunning) return;

    try {
      await deleteAttachment(activeConversationId, attachmentId);
      setAttachments((current) => current.filter((item) => item.id !== attachmentId));
      await refreshConversations();
    } catch (requestError) {
      console.error(requestError);
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
        mode={inferenceMode} onModeChange={setInferenceMode} isRunning={isRunning} isUploading={isUploading} />
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
        isRunning={isRunning}
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
            {!sidebarOpen && <LogoMark className="header-logo" />}
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
