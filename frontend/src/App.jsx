import { useEffect, useRef, useState } from "react";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import RetrievedChunks from "./components/RetrievedChunks";
import StatusIndicator from "./components/StatusIndicator";
import { streamChat } from "./services/api";
import "./App.css";

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

  const abortControllerRef = useRef(null);
  const bottomRef = useRef(null);

  const isRunning = !["idle", "done", "error", "cancelled"].includes(status);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [answer, status]);

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
          <div className="brand-icon">V</div>

          <div>
            <h1>Vietnamese History AI</h1>
            <p>Hybrid RAG · Qwen2.5 · Grounded Generation</p>
          </div>
        </div>

        <div className="system-badge">
          <span className="system-dot" />
          RAG Online
        </div>
      </header>

      <main className="workspace">
        <section className="conversation-panel">
          <div className="conversation-scroll">
            {!submittedQuestion && (
              <section className="welcome">
                <div className="welcome-icon">🇻🇳</div>

                <h2>Khám phá lịch sử Việt Nam</h2>

                <p>
                  Đặt câu hỏi về nhân vật, sự kiện, triều đại hoặc các giai đoạn
                  lịch sử Việt Nam. Hệ thống sẽ truy xuất bằng chứng trước khi tạo
                  câu trả lời.
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
              <div className="empty-evidence-icon">⌕</div>
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