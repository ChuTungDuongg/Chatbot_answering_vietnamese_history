function ChatMessage({ role, children, isStreaming = false }) {
  const isUser = role === "user";

  return (
    <div className={`message ${isUser ? "user-message" : "assistant-message"}`}>
      <div className="message-role">{isUser ? "Bạn" : "Sử Việt AI"}</div>

      <div className="message-content">
        {children}
        {isStreaming && <span className="cursor">|</span>}
      </div>
    </div>
  );
}

export default ChatMessage;
