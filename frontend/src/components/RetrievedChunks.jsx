function ExpandIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="m6 9 6 6 6-6"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

function RetrievedChunks({ sources }) {
  if (!sources?.length) {
    return null;
  }

  return (
    <section className="retrieved-chunks">
      <div className="retrieved-list">
        {sources.map((source, index) => {
          const chunkId = source.chunk_id ?? source.id ?? `chunk-${index}`;
          const title = source.title ?? "Unknown source";
          const text = source.text ?? source.content ?? "";
          const score = source.final_retrieval_score;
          const rerankScore = source.rerank_score;

          return (
            <details className="retrieved-card" key={chunkId}>
              <summary className="retrieved-summary">
                <div className="retrieved-summary-main">
                  <strong>
                    {index + 1}. {title}
                  </strong>

                  <code className="chunk-id">{chunkId}</code>
                </div>

                <span className="expand-icon">
                  <ExpandIcon />
                </span>
              </summary>

              <div className="retrieved-details">
                {typeof score === "number" && (
                  <div className="retrieved-score">
                    Final score: {score.toFixed(4)}
                  </div>
                )}

                {typeof rerankScore === "number" && (
                  <div className="retrieved-score">
                    Rerank score: {rerankScore.toFixed(4)}
                  </div>
                )}

                {text ? (
                  <p className="retrieved-full-text">{text}</p>
                ) : (
                  <p className="retrieved-empty-text">
                    Nội dung chunk không được backend gửi về.
                  </p>
                )}
              </div>
            </details>
          );
        })}
      </div>
    </section>
  );
}

export default RetrievedChunks;
