import { useEffect, useRef, useState } from "react";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import RetrievedChunks from "./components/RetrievedChunks";
import StatusIndicator from "./components/StatusIndicator";
import { streamChat } from "./services/api";
import "./App.css";

const THEME_STORAGE_KEY = "vn-history-theme";

function getInitialTheme() {
  if (typeof window === "undefined") {
    return "light";
  }

  const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);

  if (savedTheme === "dark" || savedTheme === "light") {
    return savedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function LogoMark({ className = "" }) {
  return (
    <svg
      className={`logo-mark ${className}`}
      viewBox="0 0 64 64"
      role="img"
      aria-label="Logo Su Viet AI"
    >
      <defs>
        <linearGradient id="logo-gradient" x1="10" x2="54" y1="8" y2="56">
          <stop offset="0%" stopColor="#ef4444" />
          <stop offset="48%" stopColor="#d97706" />
          <stop offset="100%" stopColor="#0f766e" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill="url(#logo-gradient)" />
      <path
        d="M14 28.5 32 15l18 13.5"
        fill="none"
        stroke="white"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="4.2"
      />
      <path
        d="M20 30h24M23 35h18M26 40h12"
        fill="none"
        stroke="white"
        strokeLinecap="round"
        strokeWidth="3.4"
      />
      <path
        d="M43.5 15.5 45.8 20l5 .7-3.6 3.5.9 5-4.6-2.4-4.5 2.4.8-5-3.6-3.5 5-.7 2.3-4.5Z"
        fill="#fef3c7"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M20.2 14.8A8.6 8.6 0 0 1 9.2 3.8 8.8 8.8 0 1 0 20.2 14.8Z"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 4V2m0 20v-2M4 12H2m20 0h-2M5 5l-1.4-1.4M20.4 20.4 19 19M19 5l1.4-1.4M3.6 20.4 5 19"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
      />
      <circle
        cx="12"
        cy="12"
        r="4"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      />
    </svg>
  );
}

function getDeltaText(data) {
  if (typeof data === "string") {
    return data;
  }

  return data?.delta ?? data?.text ?? data?.content ?? "";
}

function getStatusText(event, data) {
  if (typeof data === "string") {
    return data;
  }

  return data?.status ?? data?.stage ?? data?.message ?? event;
}

function getSources(data) {
  if (Array.isArray(data)) {
    return data;
  }

  return (
    data?.items ??
    data?.sources ??
    data?.chunks ??
    data?.contexts ??
    data?.final_context ??
    data?.retrieval?.final_context ??
    []
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [debugData, setDebugData] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [theme, setTheme] = useState(getInitialTheme);

  const abortControllerRef = useRef(null);
  const bottomRef = useRef(null);

  const isRunning = !["idle", "done", "error", "cancelled"].includes(status);
  const isDarkTheme = theme === "dark";

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [answer, status]);

  const toggleTheme = () => {
    setTheme((currentTheme) => (currentTheme === "dark" ? "light" : "dark"));
  };

  const handleSubmit = async (event) => {
    event?.preventDefault();

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || isRunning) {
      return;
    }

    setSubmittedQuestion(trimmedQuestion);
    setQuestion("");
    setAnswer("");
    setSources([]);
    setDebugData(null);
    setError("");
    setStatus("processing");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamChat({
        question: trimmedQuestion,
        finalK: 6,
        debug: true,
        signal: controller.signal,
        onEvent: ({ event: eventName, data }) => {
          console.log("SSE:", eventName, data);

          switch (eventName) {
            case "status":
            case "retrieval_started":
            case "reranking":
            case "generating":
            case "validating":
            case "validated":
              setStatus(getStatusText(eventName, data));
              break;

            case "answer_delta":
              setAnswer((current) => current + getDeltaText(data));
              setStatus("streaming");
              break;

            case "sources": {
              const retrievedSources = getSources(data);

              if (retrievedSources.length > 0) {
                setSources(retrievedSources);
              }

              break;
            }

            case "debug": {
              setDebugData(data);

              const retrievedSources = getSources(data);

              if (retrievedSources.length > 0) {
                setSources(retrievedSources);
              }

              break;
            }

            case "error":
              setError(
                typeof data === "string"
                  ? data
                  : data?.message ?? "Backend returned an unknown error.",
              );
              setStatus("error");
              break;

            case "done":
              setStatus("done");
              break;

            default:
              break;
          }
        },
      });

      setStatus((current) => {
        if (current === "error" || current === "cancelled") {
          return current;
        }

        return "done";
      });
    } catch (err) {
      if (err.name === "AbortError") {
        setStatus("cancelled");
      } else {
        console.error(err);
        setError(err.message || "Unable to connect to the backend.");
        setStatus("error");
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    abortControllerRef.current?.abort();
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <LogoMark />

          <div className="brand-copy">
            <h1>Sử Việt AI</h1>
            <p>Trợ lý hỏi đáp lịch sử Việt Nam</p>
          </div>
        </div>

        <div className="topbar-actions">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={
              isDarkTheme ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"
            }
            aria-pressed={isDarkTheme}
            title={
              isDarkTheme ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"
            }
          >
            {isDarkTheme ? <SunIcon /> : <MoonIcon />}
            <span className="sr-only">
              {isDarkTheme ? "Light mode" : "Dark mode"}
            </span>
          </button>

          <div className="system-badge">
            <span className="system-dot" />
            RAG Online
          </div>
        </div>
      </header>

      <main className="workspace">
        <section className="conversation-panel">
          <div className="conversation-scroll">
            {!submittedQuestion && (
              <section className="welcome">
                <LogoMark className="welcome-logo" />

                <h2>Khám phá lịch sử Việt Nam</h2>

                <p>
                  Những câu chuyện, triều đại và bước ngoặt lịch sử được đặt
                  trong một không gian trò chuyện rõ ràng, có dẫn chứng.
                </p>

                <div className="example-grid">
                  <button
                    type="button"
                    onClick={() =>
                      setQuestion(
                        "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?",
                      )
                    }
                  >
                    Ý nghĩa chiến thắng Bạch Đằng năm 938
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setQuestion(
                        "Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?",
                      )
                    }
                  >
                    Vì sao nhà Trần suy yếu?
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setQuestion(
                        "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ.",
                      )
                    }
                  >
                    So sánh hai sự kiện lịch sử
                  </button>
                </div>
              </section>
            )}

            {submittedQuestion && (
              <>
                <ChatMessage role="user">{submittedQuestion}</ChatMessage>

                {(answer || isRunning) && (
                  <ChatMessage
                    role="assistant"
                    isStreaming={status === "streaming"}
                  >
                    {answer || "Đang chuẩn bị câu trả lời..."}
                  </ChatMessage>
                )}

                <StatusIndicator status={status} />

                {error && (
                  <div className="error-box">
                    <strong>Không thể hoàn tất yêu cầu</strong>
                    <p>{error}</p>
                  </div>
                )}

                {debugData && (
                  <details className="debug-panel">
                    <summary>Debug information</summary>
                    <pre>{JSON.stringify(debugData, null, 2)}</pre>
                  </details>
                )}
              </>
            )}

            <div ref={bottomRef} />
          </div>

          <div className="input-area">
            <ChatInput
              question={question}
              onQuestionChange={setQuestion}
              onSubmit={handleSubmit}
              onStop={handleStop}
              isRunning={isRunning}
            />

            <p className="input-note">
              Câu trả lời được tạo dựa trên các tài liệu được hệ thống truy xuất.
            </p>
          </div>
        </section>

        <aside className="evidence-panel">
          <div className="evidence-panel-header">
            <div>
              <span className="eyebrow">RAG TRACE</span>
              <h2>Retrieved Evidence</h2>
            </div>

            {sources.length > 0 && (
              <span className="chunk-count">{sources.length}</span>
            )}
          </div>

          {sources.length > 0 ? (
            <RetrievedChunks sources={sources} />
          ) : (
            <div className="empty-evidence">
              <LogoMark className="empty-evidence-logo" />
              <h3>Chưa có tài liệu</h3>
              <p>
                Các chunk được Hybrid RAG lựa chọn sẽ xuất hiện tại đây sau khi
                bạn gửi câu hỏi.
              </p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}

export default App;
