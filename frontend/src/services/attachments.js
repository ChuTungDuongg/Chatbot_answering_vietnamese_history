export const MAX_ATTACHMENTS = 5;
export const MAX_FILE_SIZE = 20 * 1024 * 1024;
const MIME_BY_EXTENSION = { pdf: "application/pdf", png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp" };
const ALLOWED_TYPES = new Set(Object.values(MIME_BY_EXTENSION));
const IMAGE_EXTENSIONS = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp" };

export function normalizeUploadFile(file) {
  // Only infer missing MIME. A declared unsupported type must not become valid by renaming it.
  if (file.type) return file;
  const type = MIME_BY_EXTENSION[file.name.split(".").pop()?.toLowerCase()];
  return type ? new File([file], file.name, { type, lastModified: file.lastModified }) : file;
}

export function validateAttachments(files, existingCount = 0) {
  if (files.length + existingCount > MAX_ATTACHMENTS) return `Mỗi cuộc trò chuyện chỉ nhận tối đa ${MAX_ATTACHMENTS} tài liệu.`;
  for (const file of files) {
    if (!ALLOWED_TYPES.has(file.type)) return `Không hỗ trợ định dạng của ${file.name}. Chỉ nhận PDF, PNG, JPEG và WebP.`;
    if (file.size > MAX_FILE_SIZE) return `${file.name} vượt quá giới hạn 20 MB.`;
    if (!file.size) return `${file.name} không có nội dung.`;
  }
  return "";
}

export function clipboardImages(clipboardData, sequence) {
  return Array.from(clipboardData?.items ?? [])
    .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
    .map((item) => item.getAsFile()).filter(Boolean)
    .map((file, index) => new File([file], `clipboard-image-${sequence + index}.${IMAGE_EXTENSIONS[file.type] ?? "unsupported"}`,
      { type: file.type, lastModified: file.lastModified }));
}
