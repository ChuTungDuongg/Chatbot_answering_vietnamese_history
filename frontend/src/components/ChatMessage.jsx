import { createContext, useContext, useState } from "react";
import { BookOpenText, Check, Copy, Info } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import LogoMark from "./LogoMark";
import DeveloperTrace from "./DeveloperTrace";
import { CHAT_MODES } from "../config/chatModes";
import { displayAnswer, indexedSources, remarkSourceCitations } from "../services/citations";
import { progressLabel } from "../services/progressLabels";

const CitationContext = createContext(null);

function MessageLink({ href, children }) {
  const { sourcesByIndex, onShowSources } = useContext(CitationContext);
  const index = Number(href?.match(/^#source-(\d+)$/)?.[1]);
  const source = sourcesByIndex.get(index);
  return source ? <button type="button" className="citation" onClick={() => onShowSources?.(index)}
    aria-label={`Nguồn ${index}: ${source.title ?? "Tư liệu tham khảo"}`} title={source.title}>{children}</button>
    : <a href={href} target="_blank" rel="noreferrer">{children}</a>;
}

const markdownComponents = {
  a: MessageLink,
  table: ({ children }) => <div className="table-scroll" role="region" aria-label="Bảng so sánh" tabIndex={0}><table>{children}</table></div>,
};

function ChatMessage({ message, isStreaming = false, onShowSources, enableDebugTrace = false }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const sourceCount = message.sources?.length ?? 0;
  const content = isUser ? message.content : displayAnswer(message.content, message.sources);
  const sourcesByIndex = indexedSources(message.sources);
  const insufficient = !isUser && (message.status === "insufficient_evidence" || message.answer_status === "insufficient_evidence"
    || /^Mình chưa tìm thấy đủ bằng chứng đáng tin cậy/.test(message.content ?? ""));
  const validationFailed = !isUser && (message.status === "answer_validation_failed"
    || /^Đã tìm thấy tư liệu phù hợp/.test(message.content ?? ""));
  const canonicalMode = message.mode === "hybrid_rag"
    ? "hybrid"
    : message.mode === "agentic_rag"
      ? "three_llm"
      : message.mode === "fast"
        ? "hybrid"
        : message.mode === "agent"
          ? "central"
      : message.mode;
  const modeLabel = CHAT_MODES.find((item) => item.value === canonicalMode)?.label ?? "";

  const copyMessage = async () => {
    if (!message.content) return;

    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <article className={`message-row ${isUser ? "user-message" : "assistant-message"}`}>
      <div className="message-body">
        <div className="message-author">
          {!isUser && <LogoMark />}
          <span>{isUser ? "Bạn" : "Sử Việt AI"}</span>
          {!isUser && modeLabel && <small>{modeLabel}</small>}
        </div>

        <div className={`message-content ${insufficient || validationFailed ? "insufficient-panel" : ""}`}>
          {insufficient && <div className="insufficient-title"><Info aria-hidden="true" /><strong>Chưa đủ tư liệu để trả lời chắc chắn.</strong></div>}
          {validationFailed && <div className="insufficient-title"><Info aria-hidden="true" /><strong>Câu trả lời chưa vượt qua kiểm tra.</strong></div>}
          {isUser ? (
            message.content
          ) : message.content ? (
            <CitationContext.Provider value={{ sourcesByIndex, onShowSources }}>
              <ReactMarkdown remarkPlugins={[remarkGfm, [remarkSourceCitations, { sources: message.sources }]]}
                components={markdownComponents}>{content}</ReactMarkdown>
            </CitationContext.Provider>
          ) : (
            <span className="thinking-state" role="status">
              <span className="thinking-dots" aria-hidden="true"><i /><i /><i /></span>
              {progressLabel(message.status)}
            </span>
          )}
          {isStreaming && message.content && <span className="stream-cursor" aria-label="Đang trả lời" />}
        </div>

        {!isUser && message.content && (
          <div className="message-actions">
            <button type="button" className="message-action copy-action" onClick={copyMessage} title="Sao chép" aria-label="Sao chép câu trả lời">
              {copied ? <Check /> : <Copy />}
              <span aria-live="polite">{copied ? "Đã sao chép" : "Sao chép"}</span>
            </button>

            {sourceCount > 0 && (
              <button type="button" className="message-action sources-disclosure" onClick={() => onShowSources?.()} title="Xem nguồn">
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
