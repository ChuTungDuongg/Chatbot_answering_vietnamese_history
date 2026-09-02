import { useMemo, useState } from "react";
import {
  Check,
  Menu,
  MessageSquare,
  Moon,
  MoreHorizontal,
  PanelLeftClose,
  Pencil,
  Plus,
  Search,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import LogoMark from "./LogoMark";

function ChatSidebar({
  conversations,
  activeConversationId,
  isOpen,
  isLoading,
  isRunning,
  theme,
  onClose,
  onNewConversation,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
  onToggleTheme,
}) {
  const [query, setQuery] = useState("");
  const [menuId, setMenuId] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState("");

  const filteredConversations = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase("vi");
    if (!normalizedQuery) return conversations;

    return conversations.filter((conversation) =>
      (conversation.title || "Cuộc trò chuyện mới").toLocaleLowerCase("vi").includes(normalizedQuery),
    );
  }, [conversations, query]);

  const startRename = (conversation) => {
    setEditingId(conversation.id);
    setEditingTitle(conversation.title || "Cuộc trò chuyện mới");
    setMenuId(null);
  };

  const finishRename = async (conversation) => {
    const title = editingTitle.trim();
    if (title && title !== conversation.title) await onRenameConversation(conversation, title);
    setEditingId(null);
  };

  return (
    <aside className={`chat-sidebar ${isOpen ? "is-open" : ""}`} aria-label="Lịch sử trò chuyện" inert={!isOpen} aria-hidden={!isOpen}>
      <div className="sidebar-brand-row">
        <div className="sidebar-brand">
          <LogoMark />
          <span>
            <strong>Sử Việt AI</strong>
            <small>Một góc nhìn về quá khứ</small>
          </span>
        </div>

        <button type="button" className="icon-button sidebar-close" onClick={onClose} title="Đóng thanh bên" aria-label="Đóng thanh bên">
          <PanelLeftClose className="desktop-close-icon" />
          <X className="mobile-close-icon" />
        </button>
      </div>

      <button
        type="button"
        className="new-chat-button"
        onClick={onNewConversation}
        disabled={isRunning}
      >
        <Plus />
        <span>Cuộc trò chuyện mới</span>
      </button>

      <label className="conversation-search">
        <Search />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Tìm cuộc trò chuyện" />
        <span className="sr-only">Tìm cuộc trò chuyện</span>
      </label>

      <div className="conversation-section-label">
        <span>Gần đây</span>
        <span>{conversations.length}</span>
      </div>

      <div className="conversation-list">
        {isLoading && <div className="sidebar-skeleton"><i /><i /><i /></div>}

        {!isLoading && filteredConversations.length === 0 && (
          <div className="conversation-empty">
            <MessageSquare />
            <span>{query ? "Không tìm thấy cuộc trò chuyện" : "Chưa có cuộc trò chuyện"}</span>
          </div>
        )}

        {filteredConversations.map((conversation) => {
          const isActive = conversation.id === activeConversationId;
          const isEditing = editingId === conversation.id;

          return (
            <div className={`conversation-item ${isActive ? "is-active" : ""}`} key={conversation.id}>
              <MessageSquare className="conversation-icon" />

              {isEditing ? (
                <form
                  className="conversation-rename"
                  onSubmit={(event) => {
                    event.preventDefault();
                    finishRename(conversation);
                  }}
                >
                  <input
                    autoFocus
                    value={editingTitle}
                    onChange={(event) => setEditingTitle(event.target.value)}
                    onBlur={(event) => {
                      if (!event.currentTarget.form?.contains(event.relatedTarget)) finishRename(conversation);
                    }}
                    maxLength={120}
                  />
                  <button type="submit" aria-label="Lưu tên" title="Lưu tên"><Check /></button>
                </form>
              ) : (
                <button
                  type="button"
                  className="conversation-select"
                  onClick={() => onSelectConversation(conversation.id)}
                  disabled={isRunning}
                  title={conversation.title || "Cuộc trò chuyện mới"}
                >
                  <span>{conversation.title || "Cuộc trò chuyện mới"}</span>
                  <small>{conversation.message_count ?? 0} tin nhắn</small>
                </button>
              )}

              {!isEditing && (
                <button
                  type="button"
                  className="conversation-menu-button"
                  onClick={() => setMenuId((current) => current === conversation.id ? null : conversation.id)}
                  aria-label="Tùy chọn cuộc trò chuyện"
                  title="Tùy chọn"
                >
                  <MoreHorizontal />
                </button>
              )}

              {menuId === conversation.id && (
                <div className="conversation-menu">
                  <button type="button" onClick={() => startRename(conversation)}>
                    <Pencil />
                    Đổi tên
                  </button>
                  <button
                    type="button"
                    className="danger-menu-item"
                    onClick={() => {
                      setMenuId(null);
                      onDeleteConversation(conversation);
                    }}
                  >
                    <Trash2 />
                    Xóa
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="sidebar-footer">
        <button type="button" onClick={onToggleTheme}>
          {theme === "dark" ? <Sun /> : <Moon />}
          <span>{theme === "dark" ? "Giao diện sáng" : "Giao diện tối"}</span>
        </button>
      </div>
    </aside>
  );
}

export function SidebarOpenButton({ onClick }) {
  return (
    <button type="button" className="icon-button sidebar-open-button" onClick={onClick} title="Mở thanh bên" aria-label="Mở thanh bên">
      <Menu />
    </button>
  );
}

export default ChatSidebar;
