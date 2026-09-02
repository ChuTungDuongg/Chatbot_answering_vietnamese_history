import { BookOpen, ChevronDown, FileImage, FileText } from "lucide-react";

function sourceLabel(source) {
  if (source.source_kind === "attachment") return "Tài liệu của bạn";
  if (source.source_kind === "wikipedia") return "Wikipedia";
  if (source.source_kind === "web") return "Nguồn web";
  return "Kho sử liệu";
}

function RetrievedChunks({ sources }) {
  if (!sources?.length) return null;

  return (
    <div className="retrieved-list">
      {sources.map((source, index) => {
        const chunkId = source.chunk_id ?? source.id ?? `chunk-${index}`;
        const displayIndex = source.display_index ?? index + 1;
        const title = source.title ?? "Nguồn chưa có tiêu đề";
        const text = source.text ?? source.content ?? "";
        const score = source.final_retrieval_score;
        const rerankerScore = source.reranker_score;
        const sourceUrl = /^https?:\/\//i.test(source.url ?? "") ? source.url : null;
        const isAttachment = source.source_kind === "attachment" || String(chunkId).startsWith("temp:");
        const isImage = /\.(png|jpe?g|webp)$/i.test(title);
        const SourceIcon = isAttachment ? (isImage ? FileImage : FileText) : BookOpen;

        return (
          <details className="retrieved-card" key={chunkId}>
            <summary className="retrieved-summary">
              <SourceIcon className="retrieved-type-icon" />
              <span className="retrieved-summary-main">
                <span className="retrieved-source-type">{sourceLabel(source)}</span>
                <strong>[{displayIndex}] {title}</strong>
                <span className="retrieved-meta">
                  {source.page_number ? `Trang ${source.page_number}` : `Nguồn ${displayIndex}`}
                  {source.cited ? " · Được trích dẫn" : ""}
                </span>
              </span>
              <ChevronDown className="expand-icon" />
            </summary>

            <div className="retrieved-details">
              {sourceUrl && (
                <a href={sourceUrl} target="_blank" rel="noreferrer">Mở nguồn</a>
              )}

              {(typeof score === "number" || typeof rerankerScore === "number") && (
                <div className="score-row">
                  {typeof score === "number" && <span>Final {score.toFixed(3)}</span>}
                  {typeof rerankerScore === "number" && <span>Rerank {rerankerScore.toFixed(3)}</span>}
                </div>
              )}

              {text ? (
                <p className="retrieved-full-text">{text}</p>
              ) : (
                <p className="retrieved-empty-text">Nội dung chi tiết không có trong message đã lưu.</p>
              )}
            </div>
          </details>
        );
      })}
    </div>
  );
}

export default RetrievedChunks;
