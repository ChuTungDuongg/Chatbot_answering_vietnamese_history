import { useState } from "react";
import { BookOpenText, Check, Copy, UserRound } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import LogoMark from "./LogoMark";
import DeveloperTrace from "./DeveloperTrace";

function ChatMessage({ message, isStreaming = false, onShowSources, enableDebugTrace = false }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const sourceCount = message.sources?.length ?? 0;
  const modeLabel = message.mode === "hybrid_rag"
    ? "Hybrid RAG — Fast"
    : message.mode === "agentic_rag"
      ? "Agentic RAG — Deep Research"
      : "";

  const copyMessage = async () => {
    if (!message.content) return;

    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <article className={`message-row ${isUser ? "user-message" : "assistant-message"}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? <UserRound /> : <LogoMark />}
      </div>

      <div className="message-body">
        <div className="message-author">
          <span>{isUser ? "Bạn" : "Sử Việt AI"}</span>
          {!isUser && modeLabel && <small>{modeLabel}</small>}
        </div>

        <div className="message-content">
          {isUser ? (
            message.content
          ) : message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          ) : (
            <span className="thinking-dots" aria-label="Đang tạo câu trả lời">
              <i />
              <i />
              <i />
            </span>
          )}
          {isStreaming && message.content && <span className="stream-cursor" />}
        </div>

        {!isUser && message.content && (
          <div className="message-actions">
            <button type="button" className="message-action" onClick={copyMessage} title="Sao chép">
              {copied ? <Check /> : <Copy />}
              <span>{copied ? "Đã chép" : "Sao chép"}</span>
            </button>

            {sourceCount > 0 && (
              <button type="button" className="message-action" onClick={onShowSources} title="Xem nguồn">
                <BookOpenText />
                <span>{sourceCount} nguồn</span>
              </button>
            )}
          </div>
        )}

        {enableDebugTrace && !isUser && message.debug_trace && (
          <DeveloperTrace trace={message.debug_trace} />
        )}
      </div>
    </article>
  );
}

export default ChatMessage;
