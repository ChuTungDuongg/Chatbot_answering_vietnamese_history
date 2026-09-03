import { useEffect, useRef } from "react";
import { ArrowUpRight, ChevronDown } from "lucide-react";

function sourceLabel(source) {
  if (source.source_kind === "attachment") return "Tài liệu của bạn";
  if (source.source_kind === "wikipedia") return "Wikipedia";
  if (source.source_kind === "web") return "Nguồn web";
  return "Kho sử liệu";
}

function RetrievedChunks({ sources, activeIndex }) {
  const rootRef = useRef(null);
  useEffect(() => {
    const selected = rootRef.current?.querySelector(`[data-source-index="${activeIndex}"]`);
    selected?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);
  if (!sources?.length) return null;

  return (
    <div className="retrieved-list" ref={rootRef}>
      {sources.map((source, index) => {
        const chunkId = source.chunk_id ?? source.id ?? `chunk-${index}`;
        const displayIndex = source.display_index ?? index + 1;
        const title = source.title ?? "Nguồn chưa có tiêu đề";
        const text = source.text ?? source.content ?? "";
        const sourceUrl = /^https?:\/\//i.test(source.url ?? "") ? source.url : null;

        return (
          <details className={`retrieved-card ${activeIndex === displayIndex ? "is-highlighted" : ""}`} key={chunkId} data-source-index={displayIndex} open={activeIndex === displayIndex || undefined}>
            <summary className="retrieved-summary">
              <span className="source-index" aria-hidden="true">{displayIndex}</span>
              <span className="retrieved-summary-main">
                <span className="retrieved-source-type">{sourceLabel(source)}</span>
                <strong><span className="sr-only">[{displayIndex}] </span>{title}</strong>
                <span className="retrieved-meta">
                  {source.page_number ? `Trang ${source.page_number}` : `Nguồn ${displayIndex}`}
                  {source.cited ? " · Được trích dẫn" : ""}
                </span>
              </span>
              <ChevronDown className="expand-icon" />
            </summary>

            <div className="retrieved-details">
              {sourceUrl && (
                <a href={sourceUrl} target="_blank" rel="noreferrer">Đọc nguồn gốc <ArrowUpRight aria-hidden="true" /></a>
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
