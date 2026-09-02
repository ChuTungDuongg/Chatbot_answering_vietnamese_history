import { useRef } from "react";
import { ArrowUpRight, Compass, Landmark, ScrollText, Swords } from "lucide-react";
import LogoMark from "./LogoMark";

const SUGGESTIONS = [
  { icon: Compass, label: "Tìm hiểu nguyên nhân", question: "Vì sao Cách mạng Tháng Tám thành công?" },
  { icon: ScrollText, label: "Đặt cạnh nhau", question: "So sánh Hiệp định Genève và Paris" },
  { icon: Swords, label: "Nhân vật & dấu mốc", question: "Ngô Quyền và chiến thắng Bạch Đằng" },
  { icon: Landmark, label: "Nhìn lại một triều đại", question: "Nhà Trần suy yếu vì những nguyên nhân nào?" },
];

function EmptyState({ children, onSuggestion }) {
  const rootRef = useRef(null);
  return (
    <section className="welcome-state" ref={rootRef} aria-labelledby="welcome-title">
      <div className="welcome-identity"><LogoMark /><span>Sử Việt AI</span></div>
      <h2 id="welcome-title">Lịch sử bắt đầu từ<br /><span>một câu hỏi.</span></h2>
      <p>Khám phá lịch sử Việt Nam qua nguồn tư liệu và phân tích.</p>
      <div className="welcome-composer">{children}</div>
      <div className="suggestions-heading">MỘT VÀI ĐIỀU ĐỂ BẮT ĐẦU</div>
      <div className="suggestion-grid">
        {SUGGESTIONS.map(({ icon: Icon, label, question }) => (
          <button type="button" key={label} onClick={() => {
            onSuggestion(question);
            rootRef.current?.querySelector("textarea")?.focus();
          }}>
            <span className="suggestion-label"><Icon aria-hidden="true" />{label}</span>
            <span className="suggestion-question">{question}<ArrowUpRight aria-hidden="true" /></span>
          </button>
        ))}
      </div>
    </section>
  );
}

export default EmptyState;
