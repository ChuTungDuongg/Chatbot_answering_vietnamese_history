import { useEffect, useRef, useState } from "react";
import { Settings2 } from "lucide-react";

export default function ToolSteering({ control, disabled }) {
  const [open, setOpen] = useState(false);
  const root = useRef(null);
  useEffect(() => {
    if (!open) return;
    const onClick = (event) => { if (!root.current?.contains(event.target)) setOpen(false); };
    const onKey = (event) => { if (event.key === "Escape") { setOpen(false); root.current?.querySelector("button")?.focus(); } };
    document.addEventListener("pointerdown", onClick);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("pointerdown", onClick); document.removeEventListener("keydown", onKey); };
  }, [open]);
  if (!control) return null;
  return <div className="tool-steering" ref={root}>
    <button type="button" className="icon-button" disabled={disabled} aria-label="Công cụ Central"
      aria-expanded={open} aria-controls="central-tools" onClick={() => setOpen(!open)}><Settings2 /></button>
    {open && <div id="central-tools" className="tool-steering-panel" role="group" aria-label="Công cụ Central Agent">
      <strong>Công cụ</strong>
      {!control.tools.length && <p>Chưa có thông tin công cụ.</p>}
      {control.tools.map((tool) => <label key={tool.id}><input type="checkbox" disabled={disabled}
        checked={control.isAllowed(tool.id)} onChange={() => control.toggleTool(tool.id)} />{tool.label}</label>)}
      {!!control.servers.length && <strong>MCP</strong>}
      {control.servers.map((server) => {
        const enabled = control.steering.allowed_mcp_servers.includes(server.id);
        return <fieldset key={server.id}><label><input type="checkbox" checked={enabled}
          disabled={disabled || !server.available} onChange={() => control.toggleServer(server.id)} />{server.label}
          {!server.available && <small>Không khả dụng</small>}</label>
          {enabled && (server.tools ?? []).map((tool) => <label className="mcp-tool-option" key={tool.id}>
            <input type="checkbox" checked={control.isAllowed(tool.id)} disabled={disabled}
              onChange={() => control.toggleTool(tool.id)} />{tool.label}</label>)}</fieldset>;
      })}
      {control.overBudget && <p role="alert">Chọn tối đa {control.limit} công cụ MCP.</p>}
      <small>Chỉ áp dụng cho request Central. Nội dung câu hỏi có thể hạn chế thêm công cụ.</small>
    </div>}
  </div>;
}
