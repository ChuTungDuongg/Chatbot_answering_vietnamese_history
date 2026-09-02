import { useEffect, useRef, useState } from "react";
import { ArrowUp, LoaderCircle, Paperclip, Square } from "lucide-react";
import ModeSelector from "./ModeSelector";

const ACCEPTED_FILES = ".pdf,image/png,image/jpeg,image/webp";

function ChatInput({
  question,
  onQuestionChange,
  onSubmit,
  onStop,
  onFilesSelected,
  mode,
  onModeChange,
  isRunning,
  isUploading,
}) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [question]);

  const handleKeyDown = (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;

    event.preventDefault();
    if (!isRunning && question.trim()) onSubmit(event);
  };

  const handleFileInput = (event) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length > 0) onFilesSelected(files);
  };

  const handleDragOver = (event) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsDragging(true);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    if (isRunning || isUploading) return;
    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) onFilesSelected(files);
  };

  return (
    <form
      className={`chat-composer ${isDragging ? "is-dragging" : ""}`}
      onSubmit={onSubmit}
      onDragEnter={handleDragOver}
      onDragOver={handleDragOver}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        hidden
        type="file"
        accept={ACCEPTED_FILES}
        multiple
        onChange={handleFileInput}
      />

      <textarea
        ref={textareaRef}
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Hỏi về lịch sử Việt Nam..."
        aria-label="Nội dung câu hỏi"
        rows={2}
      />

      <div className="composer-toolbar">
      <div className="composer-leading-actions">
        <ModeSelector mode={mode} onModeChange={onModeChange} disabled={isRunning} />
        <button
          type="button"
          className="icon-button composer-attach"
          onClick={() => fileInputRef.current?.click()}
          disabled={isRunning || isUploading}
          aria-label="Tải PDF hoặc hình ảnh"
          title="Tải PDF hoặc hình ảnh"
        >
          {isUploading ? <LoaderCircle className="spin" /> : <Paperclip />}
        </button>
      </div>

      {isRunning ? (
        <button
          type="button"
          className="icon-button composer-submit stop-button"
          onClick={onStop}
          aria-label="Dừng tạo câu trả lời"
          title="Dừng tạo câu trả lời"
        >
          <Square fill="currentColor" />
        </button>
      ) : (
        <button
          type="submit"
          className="icon-button composer-submit send-button"
          disabled={!question.trim()}
          aria-label="Gửi câu hỏi"
          title="Gửi câu hỏi"
        >
          <ArrowUp />
        </button>
      )}
      </div>

      {isDragging && <div className="composer-drop-label">Thả tài liệu để cùng tìm hiểu</div>}
    </form>
  );
}

export default ChatInput;
