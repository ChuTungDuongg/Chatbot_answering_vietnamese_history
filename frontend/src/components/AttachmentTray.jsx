import { CheckCircle2, FileImage, FileText, LoaderCircle, TriangleAlert, X } from "lucide-react";

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function AttachmentIcon({ filename = "", mimeType = "" }) {
  const isImage = mimeType.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(filename);
  return isImage ? <FileImage /> : <FileText />;
}

function AttachmentTray({ attachments, pendingUploads, onDelete, disabled }) {
  const items = [
    ...pendingUploads.map((item) => ({ ...item, pending: true })),
    ...attachments.map((item) => ({ ...item, pending: false })),
  ];

  if (items.length === 0) return null;

  return (
    <div className="attachment-tray" aria-label="Tài liệu của cuộc trò chuyện">
      {items.map((item) => {
        const filename = item.filename ?? item.name ?? "Tài liệu";
        const status = item.pending ? item.status : item.status ?? "ready";
        const isProcessing = ["queued", "processing", "uploading"].includes(status);
        const isFailed = status === "failed";

        return (
          <div className={`attachment-item attachment-${status}`} key={item.id} title={item.error || filename}>
            <span className="attachment-icon">
              <AttachmentIcon filename={filename} mimeType={item.mime_type ?? item.type ?? ""} />
            </span>

            <span className="attachment-copy">
              <strong>{filename}</strong>
              <small>
                {isProcessing && "Đang đọc tài liệu..."}
                {isFailed && (item.error || "Xử lý thất bại")}
                {!isProcessing && !isFailed && (
                  <>
                    Sẵn sàng để tra cứu
                    {item.size_bytes ? ` · ${formatBytes(item.size_bytes)}` : ""}
                  </>
                )}
              </small>
            </span>

            <span className="attachment-status" aria-hidden="true">
              {isProcessing && <LoaderCircle className="spin" />}
              {isFailed && <TriangleAlert />}
              {!isProcessing && !isFailed && <CheckCircle2 />}
            </span>

            {!item.pending && (
              <button
                type="button"
                className="attachment-remove"
                onClick={() => onDelete(item.id)}
                disabled={disabled}
                aria-label={`Xóa ${filename}`}
                title="Xóa tài liệu"
              >
                <X />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default AttachmentTray;
