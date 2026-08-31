# 💬 React Frontend

[Về README gốc](../README.md)

`frontend/` là giao diện React 19 + Vite 8 cho chatbot. UI có nhiều cửa sổ chat, persisted
memory ở backend, upload PDF/hình ảnh, validated SSE, Markdown/GFM, evidence drawer, responsive
sidebar và dark/light mode.

## Chạy development

Luồng chuẩn chỉ cần một lệnh tại repository root:

```powershell
npm run dev
```

Root script dùng `concurrently` để chạy Vite và `modal serve modal_app.py`. Không cần
`cd frontend` hoặc mở terminal backend riêng.

Chỉ chạy frontend khi backend đã hoạt động sẵn:

```powershell
npm run frontend
```

## Cấu hình backend URL

Tạo `frontend/.env` từ file mẫu:

```powershell
Copy-Item frontend/.env.example frontend/.env
```

```dotenv
VITE_API_BASE_URL=https://your-modal-api.modal.run
```

Không thêm dấu `/` ở cuối URL. Vite đọc biến này khi process khởi động và nhúng nó vào
production build, vì vậy phải restart Vite sau khi thay `.env`.

Ở lần chạy đầu, có thể chạy `npm run dev`, lấy URL `https://...modal.run` từ log
`[BACKEND]`, cập nhật `frontend/.env`, rồi chạy lại `npm run dev`.

## Bản đồ source

| File | Trách nhiệm | Sửa khi |
|---|---|---|
| [`src/App.jsx`](src/App.jsx) | Bootstrap, active conversation, message/upload/SSE state, theme | Thay workflow chính |
| [`src/services/api.js`](src/services/api.js) | Base URL, client header, REST calls, SSE parser | Backend contract thay đổi |
| [`src/components/ChatSidebar.jsx`](src/components/ChatSidebar.jsx) | List/search/create/rename/delete và mobile navigation | Thay quản lý cửa sổ chat |
| [`src/components/ChatInput.jsx`](src/components/ChatInput.jsx) | Composer, mode selector, submit/stop, picker, drag/drop | Thay input/upload interaction |
| [`src/components/ModeSelector.jsx`](src/components/ModeSelector.jsx) | Compact Fast/Hybrid/Agent dropdown ở trái composer | Thay mode interaction |
| [`src/config/chatModes.js`](src/config/chatModes.js) | Canonical mode values, labels và localStorage helper | Thay mode contract |
| [`src/components/AttachmentTray.jsx`](src/components/AttachmentTray.jsx) | Attachment queue/status/delete | Thay document status UI |
| [`src/components/ChatMessage.jsx`](src/components/ChatMessage.jsx) | Markdown message, copy và source action | Thay answer rendering |
| [`src/components/RetrievedChunks.jsx`](src/components/RetrievedChunks.jsx) | Drawer hiển thị global/temp evidence | Thay source inspection |
| [`src/components/StatusIndicator.jsx`](src/components/StatusIndicator.jsx) | Chuyển SSE state thành nhãn UI | Backend thêm trạng thái |
| [`src/components/LogoMark.jsx`](src/components/LogoMark.jsx) | Logo Sử Việt AI | Thay brand mark |
| [`src/App.css`](src/App.css) | Layout và component styles responsive | Thay bố cục |
| [`src/index.css`](src/index.css) | Reset, font, light/dark design tokens | Thay theme |
| [`src/main.jsx`](src/main.jsx) | React entry point | Thay bootstrap |

## Identity và state

Frontend lưu ba giá trị vào `localStorage`:

| Key | Giá trị |
|---|---|
| `vn-history-client-id` | UUID gửi bằng header `X-Client-ID` |
| `vn-history-theme` | `light` hoặc `dark` |
| `vn-history-chat-mode` | `fast`, `hybrid` hoặc `agent` |

Messages, sources, attachments và temporary chunks nằm trong backend SQLite; frontend không
lưu lịch sử chat trong trình duyệt. Khi xóa browser storage, client nhận UUID mới và không còn
liệt kê các conversation thuộc UUID cũ.

`X-Client-ID` chỉ phân vùng dữ liệu cho anonymous demo. Nó không phải đăng nhập hoặc cơ chế
bảo vệ dữ liệu production.

## Conversation flow

```text
App bootstrap
  -> lấy/tạo X-Client-ID
  -> list conversations
  -> chọn hoặc tạo conversation
  -> load messages + attachments
  -> chat, rename, delete hoặc upload
```

Backend tự đổi title mặc định theo câu hỏi user đầu tiên. UI đồng bộ lại conversation sau các
thao tác làm thay đổi dữ liệu server.

## Validated SSE

```text
streamChat(conversation_id, question)
  -> status/ping cập nhật processing UI
  -> answer_delta ghép answer cuối
  -> sources cập nhật evidence drawer
  -> debug khi được yêu cầu
  -> done đồng bộ message IDs và trạng thái
```

`answer_delta` không phải token stream trực tiếp từ model. Backend chỉ phát answer sau khi
generation, guards và repair đã hoàn tất, rồi chia kết quả theo từng đoạn từ để tạo hiệu ứng
streaming an toàn.

Nút stop dùng `AbortController` để hủy request phía client. UI phải luôn xử lý được cả
`done`, `error`, network abort và conversation reload.

## Upload document

- Hỗ trợ PDF, PNG, JPEG và WebP.
- Giới hạn phía API là 20 MB mỗi file.
- File picker nhận tối đa 5 file trong một lần chọn.
- Frontend xử lý queue tuần tự, không upload cả 5 file song song.
- File luôn gắn với active conversation.
- Chỉ upload sau khi conversation đã được tạo ở backend.

Backend chịu trách nhiệm validate MIME/magic bytes, giới hạn trang, text extraction, OCR,
chunking và embedding. UI validation chỉ giúp báo lỗi sớm.

PDF có text được đọc bằng PyMuPDF. Trang scan/ít text và ảnh được OCR bằng Tesseract
`vie+eng`. Temporary corpus chỉ được retrieval trong conversation chứa file đó.

## Render answer và source

- Answer dùng `react-markdown` và `remark-gfm`.
- HTML thô từ model không được render.
- Heading, list, bảng và code block Markdown được hiển thị trực tiếp.
- Source drawer phân biệt history corpus và attachment corpus, đồng thời hiển thị page khi có.
- Structured answer như “Câu trả lời”, “Lý do và bằng chứng”, “Góc nhìn khác”, “Kết luận” chỉ
  cần backend prompt sinh Markdown đúng; không cần sửa frontend renderer.

## Theme và responsive layout

- Theme khởi tạo theo system preference nếu chưa có lựa chọn lưu.
- Light/dark tokens nằm trong `src/index.css`.
- Application layout và breakpoint nằm trong `src/App.css`.
- Sidebar chuyển sang mobile layout ở khoảng 840 px.
- Lucide được dùng cho icon thao tác; `LogoMark` là SVG nhận diện riêng.

Khi chỉnh UI, kiểm tra text dài, source title, tên file và conversation title không làm tràn
toolbar, composer, drawer hoặc sidebar.

## Kiểm tra

Chạy tại repository root:

```powershell
npm run frontend:lint
npm run frontend:build
```

Chạy automated contract tests cho mode selector và API payload:

```powershell
cd frontend
npm test
```

Ngoài automated tests, sau thay đổi workflow cần smoke test:

- create/select/search/rename/delete conversation;
- refresh trang và xác nhận memory còn trong SQLite;
- chat liên tiếp có câu hỏi phụ thuộc ngữ cảnh;
- upload text PDF, scanned PDF và image;
- xóa attachment rồi xác nhận source tạm không còn;
- stop stream và xử lý lỗi mạng;
- source drawer trên desktop/mobile;
- cả light và dark mode.
