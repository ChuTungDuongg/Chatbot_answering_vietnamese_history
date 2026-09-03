import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  BookOpenText,
  Landmark,
  Moon,
  PanelRightClose,
  PanelRightOpen,
  Paperclip,
  ScrollText,
  Sun,
  Swords,
  Trash2,
  X,
} from "lucide-react";
import AttachmentTray from "./components/AttachmentTray";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import ChatSidebar, { SidebarOpenButton } from "./components/ChatSidebar";
import LogoMark from "./components/LogoMark";
import RetrievedChunks from "./components/RetrievedChunks";
import StatusIndicator from "./components/StatusIndicator";
import {
  CONVERSATION_CREATE_NEW_FAILURE_MESSAGE,
  CONVERSATION_DELETE_FAILURE_MESSAGE,
  CONVERSATION_RENAME_FAILURE_MESSAGE,
} from "./config/messages";
import { useAttachments } from "./hooks/useAttachments";
import { useChatMode } from "./hooks/useChatMode";
import { useChatSession } from "./hooks/useChatSession";
import { useChatStream } from "./hooks/useChatStream";
import { useTheme } from "./hooks/useTheme";
import { shouldShowDebugTrace } from "./services/debugTrace";
import "./App.css";

const SHOW_DEBUG_TRACE = shouldShowDebugTrace(import.meta.env);
const SUGGESTIONS = [
  {
    icon: Landmark,
    label: "Một bước ngoặt lịch sử",
    question: "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
  },
  {
    icon: ScrollText,
    label: "Một triều đại",
    question: "Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?",
  },
  {
    icon: Swords,
    label: "So sánh sự kiện",
    question: "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
  },
];

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 840);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [conversationToDelete, setConversationToDelete] = useState(null);
  const [isDeletingConversation, setIsDeletingConversation] = useState(false);
  const bottomRef = useRef(null);

  const { theme, toggleTheme } = useTheme();
  const { mode: inferenceMode, setMode: setInferenceMode } = useChatMode();

  const session = useChatSession();
  const { state, dispatch, isRunning, ensureActiveConversation } = session;
  const {
    conversations,
    activeConversationId,
    messages,
    attachments,
    pendingUploads,
    sources,
    status,
    error,
    isLoadingConversations,
    isLoadingConversation,
  } = state;

  const stream = useChatStream({
    dispatch,
    isRunning,
    mode: inferenceMode,
    showDebugTrace: SHOW_DEBUG_TRACE,
    ensureActiveConversation,
  });

  const uploads = useAttachments({
    dispatch,
    activeConversationId,
    isRunning,
    ensureActiveConversation,
  });

  const isUploading = pendingUploads.length > 0;

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations],
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: status === "streaming" ? "auto" : "smooth" });
  }, [messages, status]);

  const closeSidebarOnMobile = () => {
    if (window.matchMedia("(max-width: 839px)").matches) setSidebarOpen(false);
  };

  const setError = (message) => dispatch({ type: "ERROR_SET", message });

  const handleSelectConversation = async (conversationId) => {
    if (isRunning || conversationId === activeConversationId) return;

    try {
      await session.loadConversation(conversationId);
      closeSidebarOnMobile();
    } catch (requestError) {
      // loadConversation đã dispatch CONVERSATION_LOAD_FAILED, ở đây chỉ ghi log.
      console.error(requestError);
    }
  };

  const handleNewConversation = async () => {
    try {
      await session.createNewConversation();
      setQuestion("");
      closeSidebarOnMobile();
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_CREATE_NEW_FAILURE_MESSAGE);
    }
  };

  const handleRenameConversation = async (conversation, title) => {
    try {
      await session.renameConversation(conversation, title);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_RENAME_FAILURE_MESSAGE);
    }
  };

  const confirmDeleteConversation = async () => {
    if (!conversationToDelete || isRunning) return;

    setIsDeletingConversation(true);
    setError("");

    try {
      const wasActive = activeConversationId === conversationToDelete.id;
      const remaining = await session.removeConversation(conversationToDelete.id);
      if (wasActive && remaining.length > 0) await session.loadConversation(remaining[0]);
      setConversationToDelete(null);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_DELETE_FAILURE_MESSAGE);
    } finally {
      setIsDeletingConversation(false);
    }
  };

  const handleSubmit = async (event) => {
    event?.preventDefault();
    // Chặn TRƯỚC khi xoá ô nhập. Xoá trước rồi mới để hook từ chối sẽ làm mất
    // câu người dùng vừa gõ khi họ bấm gửi lúc đang stream.
    if (!question.trim() || isRunning) return;

    const pending = question;
    setQuestion("");
    await stream.submit(pending);
  };

  const showMessageSources = (message) => {
    if (!message.sources?.length) return;
    dispatch({ type: "SOURCES_SHOWN", sources: message.sources });
    setSourcesOpen(true);
  };
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

      <section className="chat-workspace">
        <header className="chat-header">
          <div className="chat-header-main">
            {!sidebarOpen && <SidebarOpenButton onClick={() => setSidebarOpen(true)} />}
            <div className="chat-title">
              <h1>{activeConversation?.title || "Cuộc trò chuyện mới"}</h1>
              <span>
                {attachments.length > 0 ? (
                  <><Paperclip /> {attachments.length} tài liệu</>
                ) : (
                  "Vietnamese History RAG"
                )}
              </span>
            </div>
          </div>

          <div className="chat-header-actions">
            <button type="button" className="icon-button" onClick={toggleTheme} title="Đổi giao diện">
              {theme === "dark" ? <Sun /> : <Moon />}
            </button>
            <button
              type="button"
              className={`source-toggle ${sourcesOpen ? "is-active" : ""}`}
              onClick={() => setSourcesOpen((current) => !current)}
              title="Nguồn tham khảo"
            >
              {sourcesOpen ? <PanelRightClose /> : <PanelRightOpen />}
              <span>Nguồn</span>
              {sources.length > 0 && <b>{sources.length}</b>}
            </button>
          </div>
        </header>

        <main className="thread-scroll">
          <div className="thread-content">
            {isLoadingConversation && <div className="thread-loading"><i /><i /><i /></div>}

            {!isLoadingConversation && messages.length === 0 && (
              <section className="welcome-state">
                <LogoMark className="welcome-logo" />
                <h2>Hỏi chuyện sử Việt</h2>
                <p>Bạn muốn tìm hiểu nhân vật, sự kiện hay giai đoạn nào?</p>

                <div className="suggestion-grid">
                  {SUGGESTIONS.map(({ icon: Icon, label, question: suggestion }) => (
                    <button type="button" key={label} onClick={() => setQuestion(suggestion)}>
                      <Icon />
                      <span>{label}</span>
                      <small>{suggestion}</small>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {!isLoadingConversation && messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                isStreaming={message.role === "assistant" && message.status === "streaming"}
                onShowSources={() => showMessageSources(message)}
                enableDebugTrace={SHOW_DEBUG_TRACE}
              />
            ))}

            <StatusIndicator status={status} />
            <div ref={bottomRef} />
          </div>
        </main>

        {error && (
          <div className="error-toast" role="alert">
            <AlertCircle />
            <span>{error}</span>
            <button type="button" onClick={() => setError("")} aria-label="Đóng thông báo"><X /></button>
          </div>
        )}

        <footer className="composer-shell">
          <div className="composer-content">
            <AttachmentTray
              attachments={attachments}
              pendingUploads={pendingUploads}
              onDelete={uploads.remove}
              disabled={isRunning}
            />
            <ChatInput
              question={question}
              onQuestionChange={setQuestion}
              onSubmit={handleSubmit}
              onStop={stream.stop}
              onFilesSelected={uploads.upload}
              mode={inferenceMode}
              onModeChange={setInferenceMode}
              isRunning={isRunning}
              isUploading={isUploading}
            />
            <p className="composer-disclaimer">Sử Việt AI có thể mắc lỗi. Hãy đối chiếu phần nguồn khi cần độ chính xác cao.</p>
          </div>
        </footer>
      </section>

      <aside className={`source-drawer ${sourcesOpen ? "is-open" : ""}`} aria-label="Nguồn tham khảo">
        <div className="source-drawer-header">
          <div>
            <span>Bằng chứng RAG</span>
            <h2>Nguồn tham khảo</h2>
          </div>
          <button type="button" className="icon-button" onClick={() => setSourcesOpen(false)} title="Đóng nguồn"><X /></button>
        </div>

        <div className="source-drawer-body">
          {sources.length > 0 ? (
            <RetrievedChunks sources={sources} />
          ) : (
            <div className="source-empty">
              <BookOpenText />
              <strong>Chưa có nguồn được chọn</strong>
              <span>Nguồn của câu trả lời sẽ xuất hiện tại đây.</span>
            </div>
          )}

        </div>
      </aside>

      {sourcesOpen && <button className="source-backdrop" onClick={() => setSourcesOpen(false)} aria-label="Đóng nguồn" />}

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
