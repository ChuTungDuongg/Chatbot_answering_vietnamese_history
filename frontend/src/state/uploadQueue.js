import { createLocalId } from "./ids.js";

export const ALLOWED_MIME_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/webp",
]);

export const MIME_BY_EXTENSION = {
  pdf: "application/pdf",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
};

export const MAX_FILE_SIZE = 20 * 1024 * 1024;
export const MAX_FILES_PER_UPLOAD = 5;

export function normalizeUploadFile(file) {
  if (ALLOWED_MIME_TYPES.has(file.type)) return file;

  const extension = file.name.split(".").pop()?.toLowerCase();
  const inferredType = MIME_BY_EXTENSION[extension];
  if (!inferredType) return file;

  return new File([file], file.name, { type: inferredType, lastModified: file.lastModified });
}

export function validateUploadSelection(selectedFiles) {
  const files = selectedFiles.slice(0, MAX_FILES_PER_UPLOAD).map(normalizeUploadFile);

  if (selectedFiles.length > MAX_FILES_PER_UPLOAD) {
    return { error: `Mỗi lần chỉ có thể tải tối đa ${MAX_FILES_PER_UPLOAD} file.`, files: [] };
  }

  const invalidFile = files.find((file) => !ALLOWED_MIME_TYPES.has(file.type));
  if (invalidFile) {
    return {
      error: `Không hỗ trợ định dạng của ${invalidFile.name}. Chỉ nhận PDF, PNG, JPEG và WebP.`,
      files: [],
    };
  }

  const oversizedFile = files.find((file) => file.size > MAX_FILE_SIZE);
  if (oversizedFile) {
    return { error: `${oversizedFile.name} vượt quá giới hạn 20 MB.`, files: [] };
  }

  return { error: null, files };
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
