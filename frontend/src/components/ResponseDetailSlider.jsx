import { RESPONSE_MODES, RESPONSE_MODE_LABELS } from "../config/responseModes.js";

export default function ResponseDetailSlider({ value, onChange, disabled }) {
  const position = RESPONSE_MODES.indexOf(value);
  return (
    <div className="response-detail-control">
      <label htmlFor="response-detail">Độ chi tiết: <strong>{RESPONSE_MODE_LABELS[position]}</strong></label>
      <input id="response-detail" type="range" min="0" max="2" step="1"
        value={position} disabled={disabled} aria-valuetext={RESPONSE_MODE_LABELS[position]}
        onChange={(event) => onChange(RESPONSE_MODES[Number(event.target.value)])} />
      <div className="response-detail-ticks" aria-hidden="true">
        {RESPONSE_MODE_LABELS.map((label) => <span key={label}>{label}</span>)}
      </div>
    </div>
  );
}
