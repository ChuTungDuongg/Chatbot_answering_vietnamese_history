// Studio uses the owner session on its own origin. The chatbot's optional
// remote inference URL must never receive evaluation data or downloads.
const base = "/api/v1/eval";
export async function api(path, options = {}) {
  const response = await fetch(`${base}${path}`, { ...options, headers: { "Content-Type": "application/json", ...options.headers } });
  if (!response.ok) {
    if (response.status === 401) window.location.assign("/login");
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || "Không thể hoàn tất yêu cầu"));
  }
  return response.status === 204 ? null : response.json();
}
export const exportUrl = (format, smoke, approved, baseOnly) => `${base}/export?format=${format}&smoke=${smoke}&approved=${approved}&base_only=${baseOnly}`;
export const reviewExportUrl = `${base}/export/reviews`;
