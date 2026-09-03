import { createLocalId } from "./ids.js";
import { normalizeUploadFile, validateAttachments } from "../services/attachments.js";

export { normalizeUploadFile } from "../services/attachments.js";

export function validateUploadSelection(selectedFiles, existingCount = 0) {
  const files = Array.from(selectedFiles).map(normalizeUploadFile);
  const error = validateAttachments(files, existingCount);
  return { error: error || null, files: error ? [] : files };
}

export function createUploadItems(files) {
  return files.map((file) => ({
    id: createLocalId("upload"),
    name: file.name,
    type: file.type,
    size_bytes: file.size,
    status: "queued",
    file,
  }));
}
