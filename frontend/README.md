# React frontend

[Về README gốc](../README.md)

Thư mục `frontend/` chứa giao diện chatbot React 19 + Vite 8. UI hỗ trợ nhiều conversation, memory được lưu ở backend, upload PDF/hình ảnh, validated SSE, Markdown/GFM, evidence drawer và dark/light mode.

## Chạy ứng dụng

Luồng development chuẩn là chạy một lệnh tại repository root:

```powershell
npm run dev
```

Lệnh root khởi động đồng thời Vite và `modal serve modal_app.py`. Không cần mở terminal backend riêng.

Chỉ chạy frontend khi backend đã hoạt động sẵn:

```powershell
npm run frontend
```

## Cấu hình API

Tạo `frontend/.env` từ file mẫu và đặt URL backend không có dấu `/` cuối:

```dotenv
VITE_API_BASE_URL=https://your-modal-api.modal.run
```

Vite đọc biến này khi khởi động/build. Sau khi sửa `.env`, cần restart Vite.

## Bản đồ file

| File | Trách nhiệm | Bắt đầu ở đây khi |
|---|---|---|
| [`src/App.jsx`](src/App.jsx) | Conversation state, active chat, uploads, SSE orchestration và theme | Thay workflow chính |
| [`src/services/api.js`](src/services/api.js) | API base URL, `X-Client-ID`, CRUD, upload và SSE parser | Backend contract thay đổi |
| [`src/components/ChatSidebar.jsx`](src/components/ChatSidebar.jsx) | Danh sách/search/rename/delete conversation và mobile sidebar | Thay navigation |
| [`src/components/ChatInput.jsx`](src/components/ChatInput.jsx) | Composer, submit/stop, file picker và drag/drop | Thay input behavior |
| [`src/components/AttachmentTray.jsx`](src/components/AttachmentTray.jsx) | Upload progress, attachment state và delete | Thay document UI |
| [`src/components/ChatMessage.jsx`](src/components/ChatMessage.jsx) | User/assistant bubbles, Markdown/GFM, copy và source action | Thay message rendering |
| [`src/components/RetrievedChunks.jsx`](src/components/RetrievedChunks.jsx) | Evidence drawer cho global/temp sources | Thay evidence display |
| [`src/components/StatusIndicator.jsx`](src/components/StatusIndicator.jsx) | Ánh xạ SSE processing state sang label | Backend thêm status mới |
| [`src/components/LogoMark.jsx`](src/components/LogoMark.jsx) | Logo Sử Việt AI dùng tại header/sidebar/messages | Thay brand mark |
| [`src/App.css`](src/App.css) | Layout và component styles responsive | Thay bố cục/UI |
| [`src/index.css`](src/index.css) | Reset, font và dark/light design tokens | Thay palette/theme |
| [`src/main.jsx`](src/main.jsx) | React entry point | Thay bootstrap cấp ứng dụng |

## State và ownership

Frontend lưu hai giá trị trong `localStorage`:

| Key | Vai trò |
|---|---|
| `vn-history-client-id` | UUID gửi qua `X-Client-ID` để backend phân vùng conversation |
| `vn-history-theme` | `light` hoặc `dark` |

`X-Client-ID` là anonymous demo identity, không phải authentication. Nếu local storage bị xóa, trình duyệt nhận ID mới và không còn liệt kê các chat thuộc ID cũ.

Messages, attachments và temporary corpus nằm ở backend SQLite. Frontend không lưu bản sao lịch sử chat trong local storage.

## Luồng chat

```text
App chọn/tạo conversation
  -> streamChat(conversation_id, question)
  -> status/ping events cập nhật processing UI
  -> answer_delta ghép answer đã validated
  -> sources cập nhật evidence drawer
  -> done đồng bộ message IDs và trạng thái cuối
  -> reload conversation từ backend khi cần
```

Đây là validated streaming: backend hoàn tất generation, guards và repair trước khi phát `answer_delta`. UI không nên giả định đây là token stream trực tiếp từ model.

## Luồng upload

- Định dạng: PDF, PNG, JPEG và WebP.
- Tối đa 20 MB/file và 5 file trong một lượt chọn.
- Upload luôn gắn với active conversation.
- `processing`, `ready` và `failed` là attachment states từ backend.
- Chỉ gửi file sau khi conversation đã được tạo thành công.

MIME, số trang, OCR, chunking và embedding vẫn phải được backend xác thực; kiểm tra phía frontend chỉ nhằm phản hồi sớm cho người dùng.

## Styling

- Dark/light colors nằm trong CSS variables ở `src/index.css`.
- Layout desktop/mobile và component selectors nằm trong `src/App.css`.
- Dùng Lucide cho icon thao tác và `LogoMark` cho nhận diện riêng.
- Message content được render bằng `react-markdown` + `remark-gfm`; không render HTML thô từ model.
- Giữ trạng thái động không làm toolbar, composer hoặc sidebar thay đổi kích thước ngoài ý muốn.

## Kiểm tra sau khi sửa

Chạy tại root:

```powershell
npm run frontend:lint
npm run frontend:build
```

Sau đó kiểm tra thủ công desktop/mobile, cả hai theme, conversation CRUD, refresh memory, upload text PDF, upload scanned PDF/image, cancel stream và evidence drawer.
