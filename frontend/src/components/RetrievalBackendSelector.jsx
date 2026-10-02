import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { RETRIEVAL_BACKENDS } from "../config/retrievalBackends.js";

export default function RetrievalBackendSelector({ backend, available = [], onChange, disabled }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  const trigger = useRef(null);
  const options = useRef([]);
  const id = useId();
  useLayoutEffect(() => {
    if (open) options.current[RETRIEVAL_BACKENDS.findIndex((item) => available.includes(item.value))]?.focus();
  }, [open, available]);
  useEffect(() => {
    if (!open) return;
    const outside = (event) => { if (!root.current?.contains(event.target)) setOpen(false); };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  const label = RETRIEVAL_BACKENDS.find((item) => item.value === backend)?.label ?? "Truy xuất";
  const enabledIndexes = RETRIEVAL_BACKENDS.flatMap((item, index) => available.includes(item.value) ? [index] : []);
  const focus = (index) => options.current[index]?.focus();
  return <div className="composer-mode-selector retrieval-backend-selector" ref={root}>
    <button type="button" className="composer-mode-trigger" ref={trigger} disabled={disabled || !backend}
      aria-label={`Chọn nguồn truy xuất. Hiện tại: ${label}`} aria-haspopup="listbox"
      aria-expanded={open} aria-controls={open ? id : undefined}
      onClick={() => setOpen(!open)} onKeyDown={(event) => {
        if (["ArrowDown", "ArrowUp"].includes(event.key)) {
          event.preventDefault(); setOpen(true);
        }
      }}><span>{label}</span><ChevronDown aria-hidden="true" /></button>
    {open && <div className="composer-mode-menu" id={id} role="listbox" aria-label="Nguồn truy xuất">
      {RETRIEVAL_BACKENDS.map((item, index) => <button type="button" key={item.value} role="option"
        ref={(node) => { options.current[index] = node; }} disabled={!available.includes(item.value)}
        aria-selected={backend === item.value} onClick={() => {
          onChange(item.value); setOpen(false); trigger.current?.focus();
        }} onKeyDown={(event) => {
          if (event.key === "Escape") { event.preventDefault(); setOpen(false); trigger.current?.focus(); }
          else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
            event.preventDefault();
            const position = enabledIndexes.indexOf(index);
            const next = event.key === "Home" ? 0 : event.key === "End" ? enabledIndexes.length - 1
              : (position + (event.key === "ArrowDown" ? 1 : -1) + enabledIndexes.length) % enabledIndexes.length;
            focus(enabledIndexes[next]);
          }
        }}><span className="mode-option-copy"><strong>{item.label}</strong><small>{item.description}</small></span>
        {backend === item.value && <Check aria-hidden="true" />}
      </button>)}
    </div>}
  </div>;
}
