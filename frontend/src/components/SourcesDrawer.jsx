import { useEffect, useRef } from "react";
import { BookOpenText, X } from "lucide-react";
import RetrievedChunks from "./RetrievedChunks";

function SourcesDrawer({ isOpen, sources, activeIndex, onClose }) {
  const panelRef = useRef(null);
  useEffect(() => {
    if (!isOpen) return undefined;
    const trigger = document.activeElement;
    const frame = requestAnimationFrame(() => panelRef.current?.querySelector("button")?.focus());
    return () => {
      cancelAnimationFrame(frame);
      if (trigger?.isConnected) trigger.focus();
    };
  }, [isOpen]);
  const handleKeyDown = (event) => {
    if (event.key === "Escape") onClose();
    if (event.key !== "Tab") return;
    const controls = [...panelRef.current.querySelectorAll("button, summary, a[href]")].filter((node) => node.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  };
  return (
    <>
      {isOpen && <button className="source-backdrop" onClick={onClose} aria-label="Đóng nguồn" tabIndex={-1} />}
      <aside ref={panelRef} className={`source-drawer ${isOpen ? "is-open" : ""}`} role="dialog" aria-modal={isOpen || undefined}
        aria-labelledby="sources-title" aria-hidden={!isOpen} inert={!isOpen} onKeyDown={handleKeyDown}>
        <div className="source-drawer-header">
          <div><span>TƯ LIỆU THAM KHẢO</span><h2 id="sources-title">Nguồn của câu trả lời</h2></div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Đóng nguồn"><X /></button>
        </div>
        <div className="source-drawer-body">
          {sources.length ? <RetrievedChunks sources={sources} activeIndex={activeIndex} /> : (
            <div className="source-empty"><BookOpenText /><strong>Tư liệu sẽ xuất hiện ở đây</strong><span>Mở nguồn để đọc thêm và đối chiếu câu trả lời.</span></div>
          )}
        </div>
        <div className="source-drawer-footer">Đọc nguồn, hiểu thêm bối cảnh.</div>
      </aside>
    </>
  );
}

export default SourcesDrawer;
