import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { CHAT_MODES } from "../config/chatModes";

function ModeSelector({ mode, onModeChange, disabled = false }) {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef(null);
  const triggerRef = useRef(null);
  const optionRefs = useRef([]);
  const listboxId = useId();
  const selected = CHAT_MODES.find((item) => item.value === mode) ?? CHAT_MODES[0];

  useEffect(() => {
    if (!isOpen) return undefined;
    const closeOnOutsidePointer = (event) => {
      if (!rootRef.current?.contains(event.target)) setIsOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, [isOpen]);

  const focusOption = (index) => {
    const normalized = (index + CHAT_MODES.length) % CHAT_MODES.length;
    optionRefs.current[normalized]?.focus();
  };

  const openAndFocus = (index) => {
    setIsOpen(true);
    requestAnimationFrame(() => focusOption(index));
  };

  const handleTriggerKeyDown = (event) => {
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    const selectedIndex = CHAT_MODES.findIndex((item) => item.value === selected.value);
    openAndFocus(event.key === "ArrowDown" ? selectedIndex : selectedIndex - 1);
  };

  const handleOptionKeyDown = (event, index) => {
    if (event.key === "Escape") {
      event.preventDefault();
      setIsOpen(false);
      triggerRef.current?.focus();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      focusOption(index + (event.key === "ArrowDown" ? 1 : -1));
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      focusOption(event.key === "Home" ? 0 : CHAT_MODES.length - 1);
    }
  };

  const choose = (value) => {
    onModeChange(value);
    setIsOpen(false);
    triggerRef.current?.focus();
  };

  return (
    <div ref={rootRef} className="composer-mode-selector">
      <button
        ref={triggerRef}
        type="button"
        className="composer-mode-trigger"
        onClick={() => setIsOpen((open) => !open)}
        onKeyDown={handleTriggerKeyDown}
        disabled={disabled}
        aria-label={`Chọn chế độ trả lời. Hiện tại: ${selected.label}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={isOpen ? listboxId : undefined}
        title={`${selected.label}: ${selected.description}`}
      >
        <span>{selected.label}</span>
        <ChevronDown aria-hidden="true" />
      </button>

      {isOpen && (
        <div id={listboxId} className="composer-mode-menu" role="listbox" aria-label="Chế độ trả lời">
          {CHAT_MODES.map((item, index) => (
            <button
              key={item.value}
              ref={(node) => { optionRefs.current[index] = node; }}
              type="button"
              role="option"
              aria-selected={item.value === selected.value}
              className={item.value === selected.value ? "is-selected" : ""}
              onClick={() => choose(item.value)}
              onKeyDown={(event) => handleOptionKeyDown(event, index)}
            >
              <span className="mode-option-copy">
                <strong>{item.label}</strong>
                <small>{item.description}</small>
              </span>
              {item.value === selected.value && <Check aria-hidden="true" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default ModeSelector;
