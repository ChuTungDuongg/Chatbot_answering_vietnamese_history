import { useEffect, useRef } from "react";
import EvidenceInspector from "./EvidenceInspector";

export default function EvidenceDrawer({ runId, onClose }) {
  const container = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    const node = container.current;
    const focusable = () => [...node.querySelectorAll('button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]')];
    (focusable()[0] || node).focus();
    function onKey(event) {
      if (event.key === "Escape") { event.preventDefault(); onClose(); }
      if (event.key !== "Tab") return;
      const elements = focusable(), first = elements[0] || node, last = elements.at(-1) || node;
      if (event.shiftKey && (document.activeElement === first || !node.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !node.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
    }
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); if (previous?.isConnected) previous.focus(); };
  }, [onClose]);
  return <aside className="drawer-backdrop" ref={container} role="dialog" aria-modal="true" aria-label="Chi tiết run" tabIndex={-1} onClick={event => { if (event.target === event.currentTarget) onClose(); }}><EvidenceInspector runId={runId} onClose={onClose}/></aside>;
}
