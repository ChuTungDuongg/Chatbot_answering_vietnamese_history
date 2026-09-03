import { useMemo, useRef, useState } from "react";
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
  CONVERSATION_CREATE_NEW_FAILURE_MESSAGE,
  CONVERSATION_DELETE_FAILURE_MESSAGE,
  CONVERSATION_RENAME_FAILURE_MESSAGE,
} from "./config/messages";
import { useAttachments } from "./hooks/useAttachments";
import { useChatMode } from "./hooks/useChatMode";
import { useChatSession } from "./hooks/useChatSession";
import { useChatScroll } from "./hooks/useChatScroll";
import { useChatStream } from "./hooks/useChatStream";
import { useTheme } from "./hooks/useTheme";
import { shouldShowDebugTrace } from "./services/debugTrace";
import "./App.css";

const SHOW_DEBUG_TRACE = shouldShowDebugTrace(import.meta.env);

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 840);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [activeSourceIndex, setActiveSourceIndex] = useState(null);
  const [question, setQuestion] = useState("");
  const [conversationToDelete, setConversationToDelete] = useState(null);
  const [isDeletingConversation, setIsDeletingConversation] = useState(false);
  const conversationActionRef = useRef(false);

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
    isCreatingConversation,
    isUploading,
  } = state;

  const stream = useChatStream({
    dispatch,
    activeConversationId,
    attachments,
    isUploading,
    isRunning,
    mode: inferenceMode,
    showDebugTrace: SHOW_DEBUG_TRACE,
    ensureActiveConversation,
  });

  const uploads = useAttachments({
    dispatch,
    activeConversationId,
    attachments,
    isRunning,
    ensureActiveConversation,
  });

  const readyAttachments = attachments.filter((item) => item.status === "ready");
  const { scrollerRef, contentRef, onScroll, followLatest } = useChatScroll(messages, status);
  const isLoading = isLoadingConversations || isLoadingConversation || isCreatingConversation;
  const isBusy = isRunning || isUploading || isLoading || isDeletingConversation;

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations],
  );

  const closeSidebarOnMobile = () => {
    if (window.matchMedia("(max-width: 839px)").matches) setSidebarOpen(false);
  };

  const setError = (message) => dispatch({ type: "ERROR_SET", message });

  const handleSelectConversation = async (conversationId) => {
    if (isRunning || stream.isBusy() || uploads.isBusy() || conversationActionRef.current || isLoadingConversations || isCreatingConversation
      || (conversationId === activeConversationId && !isLoadingConversation)) return;

    try {
      if (await session.loadConversation(conversationId)) {
        followLatest();
        setActiveSourceIndex(null);
        closeSidebarOnMobile();
      }
    } catch (requestError) {
      // loadConversation đã dispatch CONVERSATION_LOAD_FAILED, ở đây chỉ ghi log.
      console.error(requestError);
    }
  };

  const handleNewConversation = async () => {
    if (isRunning || stream.isBusy() || uploads.isBusy() || conversationActionRef.current || isLoadingConversations) return;
    conversationActionRef.current = true;
    try {
      await session.createNewConversation();
      setQuestion("");
      setActiveSourceIndex(null);
      followLatest();
      closeSidebarOnMobile();
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_CREATE_NEW_FAILURE_MESSAGE);
    } finally {
      conversationActionRef.current = false;
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
    if (!conversationToDelete || isBusy || stream.isBusy() || uploads.isBusy() || conversationActionRef.current) return;

    conversationActionRef.current = true;
    setIsDeletingConversation(true);
    setError("");

    try {
      const wasActive = activeConversationId === conversationToDelete.id;
      const remaining = await session.removeConversation(conversationToDelete.id);
      setConversationToDelete(null);
      setActiveSourceIndex(null);
      if (wasActive && remaining.length > 0) {
        try {
          await session.loadConversation(remaining[0]);
          followLatest();
        } catch (requestError) {
          // The session owns errors from loading the replacement conversation.
          console.error(requestError);
        }
      }
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_DELETE_FAILURE_MESSAGE);
    } finally {
      setIsDeletingConversation(false);
      conversationActionRef.current = false;
    }
  };

  const handleSubmit = async (event) => {
    event?.preventDefault();
    if ((!question.trim() && !readyAttachments.length) || isBusy || stream.isBusy() || uploads.isBusy() || conversationActionRef.current) return;

    const pending = question;
    setQuestion("");
    setActiveSourceIndex(null);
    followLatest();
    await stream.submit(pending);
  };

  const showMessageSources = (message, index = null) => {
    if (!message.sources?.length) return;
    dispatch({ type: "SOURCES_SHOWN", sources: message.sources });
    setActiveSourceIndex(index);
    setSourcesOpen(true);
  };

  // Keep the empty composer mounted during lazy creation so native mixed paste completes.
  const isEmpty = !isLoadingConversation && messages.length === 0;
  const composer = (
    <div className="composer-content">
      <AttachmentTray attachments={attachments} pendingUploads={pendingUploads} onDelete={uploads.remove}
        disabled={isRunning || isLoading || isDeletingConversation} />
      <ChatInput question={question} onQuestionChange={setQuestion} onSubmit={handleSubmit}
        onStop={stream.stop} onFilesSelected={(files, options) => {
          if (!isLoading && !isDeletingConversation && !conversationActionRef.current && !stream.isBusy()) uploads.upload(files, options);
        }}
        mode={inferenceMode} onModeChange={setInferenceMode} isRunning={isRunning} isUploading={isUploading}
        disabled={isLoading || isDeletingConversation} hasAttachments={readyAttachments.length > 0} />
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
        isRunning={isRunning || isUploading || isLoadingConversations || isCreatingConversation || isDeletingConversation}
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
              <button type="button" className="danger-button" onClick={confirmDeleteConversation} disabled={isBusy}>
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
