import { useEffect, useState } from "react";
import { getRetrievalCapabilities } from "../services/api.js";
import { normalizeRetrievalBackend, readRetrievalPreference, RETRIEVAL_STORAGE_KEY } from "../config/retrievalBackends.js";

export function useRetrievalBackend() {
  const [capabilities, setCapabilities] = useState({ available_backends: [] });
  const [selected, setSelected] = useState(readRetrievalPreference);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    getRetrievalCapabilities({ signal: controller.signal }).then((result) => {
      if (!controller.signal.aborted) {
        const available = (result.available_backends ?? []).filter((name) => ["faiss", "qdrant"].includes(name));
        setCapabilities({ ...result, available_backends: available });
      }
    }).catch(() => {
      if (!controller.signal.aborted) setError("Không xác nhận được nguồn truy xuất. Hãy tải lại trang.");
    });
    return () => controller.abort();
  }, []);
  const backend = normalizeRetrievalBackend(selected, capabilities);
  useEffect(() => {
    if (backend) {
      try { window.localStorage.setItem(RETRIEVAL_STORAGE_KEY, backend); } catch { /* Storage may be blocked. */ }
    }
  }, [backend]);
  return { backend, capabilities, available: capabilities.available_backends, ready: !!backend, error,
    setBackend: (value) => { if (capabilities.available_backends.includes(value)) setSelected(value); } };
}
