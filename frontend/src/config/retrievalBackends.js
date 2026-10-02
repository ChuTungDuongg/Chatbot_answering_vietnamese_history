export const RETRIEVAL_BACKENDS = [
  { value: "faiss", label: "FAISS", description: "Local vector index" },
  { value: "qdrant", label: "Qdrant", description: "Vector database" },
];
export const RETRIEVAL_STORAGE_KEY = "vn-history-retrieval-backend-v1";

export function normalizeRetrievalBackend(value, capabilities) {
  const available = capabilities.available_backends ?? [];
  return available.includes(value) ? value
    : available.includes(capabilities.default_backend) ? capabilities.default_backend : null;
}

export function readRetrievalPreference() {
  try { return window.localStorage.getItem(RETRIEVAL_STORAGE_KEY); } catch { return null; }
}
