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
        placeholder="Hỏi một câu về lịch sử Việt Nam..."
        rows={3}
        disabled={isRunning}
      />

      <div className="form-actions">
        {isRunning ? (
          <button
            type="button"
            className="stop-button"
            onClick={onStop}
          >
            Dừng
          </button>
        ) : (
          <button
            type="submit"
            disabled={!question.trim()}
          >
            Gửi
          </button>
        )}
      </div>
    </form>
  );
}

export default ChatInput;