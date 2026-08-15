function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="m5 12 14-7-4.5 14-2.6-5.1L5 12Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="m11.9 13.9 3.2-5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect
        x="7"
        y="7"
        width="10"
        height="10"
        rx="2"
        fill="currentColor"
      />
    </svg>
  );
}

function ChatInput({
  question,
  onQuestionChange,
  onSubmit,
  onStop,
  isRunning,
}) {
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();

      if (!isRunning && question.trim()) {
        onSubmit(event);
      }
    }
  };

  return (
    <form className="chat-form" onSubmit={onSubmit}>
      <textarea
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Hỏi về một sự kiện, nhân vật hoặc triều đại..."
        rows={3}
        disabled={isRunning}
      />

      <div className="form-actions">
        {isRunning ? (
          <button
            type="button"
            className="composer-button stop-button"
            onClick={onStop}
            aria-label="Dừng tạo câu trả lời"
            title="Dừng"
          >
            <StopIcon />
            <span className="sr-only">Dừng</span>
          </button>
        ) : (
          <button
            type="submit"
            className="composer-button send-button"
            disabled={!question.trim()}
            aria-label="Gửi câu hỏi"
            title="Gửi"
          >
            <SendIcon />
            <span className="sr-only">Gửi</span>
          </button>
        )}
      </div>
    </form>
  );
}

export default ChatInput;
