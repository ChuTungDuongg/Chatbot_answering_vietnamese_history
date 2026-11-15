import { useState } from "react";

const arrays = { required_tools: [100, 100], forbidden_tools: [100, 100], allowed_tools: [100, 100], expected_evidence_ids: [100, 200], evidence_terms: [100, 1000], expected_exit_reasons: [30, 100], required_json_keys: [100, 200] };
const optional = ["max_tool_calls", "max_repeated_calls", "output_format", "require_citations", "answerable", "in_domain", "any_tool_groups"];
function validate(value) {
  if (!value || Array.isArray(value) || typeof value !== "object") return "Expected behavior cần một JSON object.";
  const stringList = (list, max, length) => Array.isArray(list) && list.length <= max && list.every(item => typeof item === "string" && item.trim().length > 0 && item.trim().length <= length);
  for (const [key, content] of Object.entries(value)) {
    if (!Object.hasOwn(arrays, key) && !optional.includes(key)) return `Trường không được hỗ trợ: ${key}. Không áp đặt trình tự gọi tool.`;
    if (Object.hasOwn(arrays, key) && !(key === "allowed_tools" && content === null) && !stringList(content, ...arrays[key])) return `${key} cần một mảng chuỗi không rỗng, trong giới hạn cho phép.`;
    if (["max_tool_calls", "max_repeated_calls"].includes(key) && content !== null && (!Number.isInteger(content) || content < 0 || content > 10000)) return `${key} cần số nguyên từ 0 đến 10000 hoặc null.`;
    if (["require_citations", "answerable", "in_domain"].includes(key) && content !== null && typeof content !== "boolean") return `${key} cần true, false hoặc null.`;
    if (key === "output_format" && content !== null && !["text", "json", "markdown"].includes(content)) return "output_format cần text, json, markdown hoặc null.";
    if (key === "any_tool_groups" && (!Array.isArray(content) || content.length > 30 || content.some(group => !stringList(group, 30, 100) || !group.length))) return "any_tool_groups cần mảng nhóm tool không rỗng.";
  }
  return "";
}
export default function ExpectedBehaviorEditor({ value, onChange }) {
  const [text, setText] = useState(value ? JSON.stringify(value, null, 2) : ""), [error, setError] = useState("");
  function edit(next) {
    setText(next);
    if (!next.trim()) { setError(""); onChange(null); return; }
    let parsed;
    try { parsed = JSON.parse(next); } catch { setError("JSON không hợp lệ. Nội dung này chưa được lưu."); return; }
    const message = validate(parsed); setError(message);
    if (!message) onChange(parsed);
  }
  return <div className="stack"><label className="eval-field"><span>Expected behavior (JSON)</span><small>Tùy chọn. Kiểm tra tập tool, giới hạn, bằng chứng và định dạng; không yêu cầu chuỗi thao tác cứng.</small><textarea aria-label="Expected behavior (JSON)" rows={9} value={text} onChange={e => edit(e.target.value)} aria-invalid={!!error} aria-describedby="expected-behavior-help" spellCheck={false} placeholder={'{"required_tools":["search_local"],"max_tool_calls":3,"require_citations":true}'}/></label>{error && <p className="inline-error" role="alert">{error}</p>}<p className="helper" id="expected-behavior-help">Trường hỗ trợ: {[...Object.keys(arrays), ...optional].join(", ")}. Chỉ JSON hợp lệ mới được tự động lưu.</p></div>;
}
