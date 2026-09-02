# Refactor frontend: tách App.jsx thành reducer + hooks, dựng nền test

**Ngày:** 2026-09-02
**Phạm vi:** `frontend/`
**Trạng thái:** đã duyệt, chờ lập implementation plan

## 1. Vấn đề

`frontend/src/App.jsx` dài 705 dòng và giữ 14 `useState` rời rạc. Ba luồng nghiệp vụ khác nhau — CRUD hội thoại, streaming SSE, upload tài liệu — cùng ghi vào một tập state chung:

| State | Được ghi bởi |
|---|---|
| `conversations` | bootstrap, create, rename, delete, kết thúc stream, kết thúc upload |
| `activeConversationId` | bootstrap, select, create, delete |
| `messages` | bootstrap, load, create, delete, submit, mỗi `answer_delta` |
| `attachments` | bootstrap, load, create, delete, upload, xoá attachment |
| `sources` | bootstrap, load, create, delete, sự kiện `sources`, click xem nguồn |
| `status` / `error` | tất cả các luồng trên |

Hệ quả cụ thể:

- Logic dễ sai nhất nằm trong closure của `handleSubmit`: biến `streamFailed` quyết định sự kiện `done` set `status` thành `"done"` hay `"error"`. Không có cách nào kiểm thử nó mà không render cả ứng dụng.
- `updateMessage` là hàm cập nhật tại chỗ, mọi nhánh xử lý sự kiện SSE đều gọi nó với hình dạng payload khác nhau.
- Test hiện chỉ phủ `config/chatModes.js`, `services/trace.js`, `services/debugTrace.js` — không có test nào chạm tới streaming, upload hay quản lý hội thoại.

## 2. Mục tiêu và phi mục tiêu

**Mục tiêu**

1. Đưa logic dễ sai (gộp delta, abort, error contract, đồng bộ sau stream) thành hàm thuần kiểm thử được không cần DOM.
2. `App.jsx` chỉ còn dựng cây component và nối dây, khoảng 200 dòng.
3. Dựng hạ tầng test cho component/hook mà repo chưa có.
4. Không thay đổi giao diện và không thay đổi contract với backend.

**Phi mục tiêu**

- Không đụng `App.css` (2144 dòng).
- Không chuyển sang TypeScript.
- Không thêm tính năng mới, không đổi UI.
- Không đưa vào Context hay store toàn cục — cây component hiện chỉ sâu một cấp, chưa có prop drilling để giải quyết.

## 3. Kiến trúc

### 3.1 File mới

```
src/config/messages.js            hằng số thông báo, thuần, không chạm import.meta.env
src/state/chatSessionReducer.js   thuần, không import React
src/state/uploadQueue.js          thuần: validate file, chuẩn hoá MIME, dựng queue
src/hooks/useChatSession.js       useReducer + action async cho CRUD hội thoại
src/hooks/useChatStream.js        SSE -> dispatch
src/hooks/useAttachments.js       upload -> dispatch
src/hooks/useTheme.js             độc lập
src/hooks/useChatMode.js          độc lập
```

**Vì sao cần `config/messages.js`.** `services/api.js` đọc `import.meta.env.VITE_API_BASE_URL` ngay ở top-level module:

```js
const API_BASE_URL = String(import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/+$/, "");
```

Dưới `node --test` (không có Vite) `import.meta.env` là `undefined`, nên chỉ cần `import` module này là ném TypeError. Reducer cần `EVIDENCE_CONTRACT_FAILURE_MESSAGE`, hằng số hiện đang nằm trong `api.js`. Nếu reducer import từ đó thì mọi test thuần chết ngay lúc import.

Cách giải: chuyển các hằng số thông báo sang `src/config/messages.js` (thuần, không chạm env). `api.js` re-export lại `EVIDENCE_CONTRACT_FAILURE_MESSAGE` để không phá code đang import từ nó.

Đây đúng là pattern repo đã dùng sẵn: `services/debugTrace.js` nhận `env` qua tham số thay vì đọc thẳng `import.meta.env`, và `config/chatModes.js` cũng thuần — cả hai vì thế mới test được bằng `node:test`.

### 3.2 Hình dạng state của reducer

```js
{
  conversations: [],
  activeConversationId: null,
  messages: [],
  attachments: [],
  pendingUploads: [],
  sources: [],
  status: "idle",
  error: "",
  streamFailed: false,
  isLoadingConversations: true,
  isLoadingConversation: false,
}
```

### 3.3 Danh sách action

Tất cả action đều thuần và đồng bộ. Gọi API là việc của hook, reducer không biết tới mạng.

| Action | Payload | Ý nghĩa |
|---|---|---|
| `BOOTSTRAPPED` | `{ conversations, detail? }` | Kết thúc lần tải đầu |
| `CONVERSATION_LOADING` | — | Bật `isLoadingConversation` |
| `CONVERSATION_LOADED` | `{ conversationId, detail }` | Đổi hội thoại đang xem, tắt `isLoadingConversation` |
| `CONVERSATION_LOAD_FAILED` | `{ message }` | Tắt `isLoadingConversation` và đặt lỗi; thiếu action này thì spinner kẹt vĩnh viễn khi API hỏng |
| `CONVERSATION_CREATED` | `{ conversation }` | Thêm vào đầu danh sách, dọn màn hình |
| `CONVERSATION_RENAMED` | `{ conversationId, patch }` | Cập nhật một mục trong danh sách |
| `CONVERSATION_DELETED` | `{ conversationId }` | Bỏ khỏi danh sách; nếu là hội thoại đang active thì dọn state |
| `MESSAGES_APPENDED` | `{ messages }` | Thêm cặp user + assistant placeholder |
| `STREAM_STATUS` | `{ messageId, status, mode }` | Sự kiện `status` |
| `STREAM_DELTA` | `{ messageId, delta }` | Nối chuỗi vào `content` |
| `STREAM_SOURCES` | `{ messageId, sources }` | Sự kiện `sources` |
| `STREAM_DEBUG` | `{ messageId, trace }` | Sự kiện `debug_trace` |
| `STREAM_ERROR` | `{ messageId, message, kind, trace }` | Sự kiện `error`; bật `streamFailed` |
| `STREAM_DONE` | `{ messageId }` | Chỉ set `"done"` khi `streamFailed` còn false |
| `STREAM_ABORTED` | `{ messageId }` | Giữ nguyên text đã stream |
| `SYNCED` | `{ conversations, detail }` | Đồng bộ lại sau khi stream hoặc upload xong |
| `UPLOAD_QUEUED` | `{ items }` | Đưa file vào `pendingUploads` |
| `UPLOAD_PROGRESS` | `{ id, status }` | Đổi trạng thái một file trong queue |
| `UPLOAD_SETTLED` | `{ id, attachment?, error? }` | Bỏ khỏi queue, ghi vào `attachments` nếu thành công |
| `ATTACHMENT_REMOVED` | `{ attachmentId }` | Xoá tài liệu |
| `ERROR_SET` | `{ message }` | Đặt hoặc xoá thông báo lỗi |

Điểm thiết kế quan trọng: cờ `streamFailed` hiện là biến cục bộ trong closure của `handleSubmit`. Nó chuyển thành một field trong state của reducer, nhờ đó quan hệ "`done` đến sau `error` thì không được ghi đè trạng thái lỗi" trở thành thứ kiểm thử được. Cờ này được reset khi bắt đầu lượt stream mới (`MESSAGES_APPENDED`).

### 3.4 Trách nhiệm của hook

| Hook | Nhận | Trả về | Không làm |
|---|---|---|---|
| `useChatSession` | — | `state`, `dispatch`, `loadConversation`, `createConversation`, `renameConversation`, `deleteConversation`, `ensureActiveConversation` | Không biết gì về SSE hay upload |
| `useChatStream` | `dispatch`, `conversationId`, `mode`, `ensureActiveConversation` | `submit(question)`, `stop()`, `isRunning` | Không giữ state riêng |
| `useAttachments` | `dispatch`, `conversationId`, `ensureActiveConversation`, `isRunning` | `upload(files)`, `remove(id)` | Không validate inline, gọi `uploadQueue.js` |
| `useTheme` | — | `theme`, `toggleTheme` | — |
| `useChatMode` | — | `mode`, `setMode` | — |

### 3.5 State ở lại trong App.jsx

`sidebarOpen`, `sourcesOpen`, `question`, `conversationToDelete`, `isDeletingConversation` là state thuần UI, cục bộ, không luồng nào khác đọc. Giữ nguyên `useState` trong `App.jsx`. Đưa chúng vào reducer chỉ làm reducer phình ra mà không được lợi gì.

## 4. Chiến lược test

### 4.1 Test thuần cho reducer và upload queue

Sau khi đổi `npm test` sang Vitest thì mọi test đều chạy qua một runner. Nhưng test cho `chatSessionReducer.js` và `uploadQueue.js` vẫn viết bằng API `node:test` + `node:assert/strict`, và hai module đó **không được import bất cứ thứ gì chạm `import.meta.env`** — nhờ vậy chúng chạy được bằng cả `vitest run` lẫn `node --test` trực tiếp, không cần DOM, không cần Vite. Đây là ràng buộc kiến trúc chứ không phải sở thích: nó ép logic nghiệp vụ tách khỏi tầng vận chuyển.

Các case cần phủ:

- Nhiều `STREAM_DELTA` liên tiếp nối đúng thứ tự vào đúng message, không đụng message khác.
- `STREAM_ERROR` với `kind === "evidence_contract_error"` cho ra `EVIDENCE_CONTRACT_FAILURE_MESSAGE`; loại khác cho ra thông báo mặc định.
- `STREAM_DONE` đến sau `STREAM_ERROR` giữ nguyên `status: "error"`.
- `STREAM_ABORTED` giữ lại phần text đã stream; nếu chưa có text thì đặt thông báo đã dừng.
- `CONVERSATION_DELETED` trên hội thoại đang active dọn `messages`/`attachments`/`sources`; trên hội thoại khác thì không.
- `SYNCED` không nuốt mất `sources` mới nhất.
- `uploadQueue`: quá 5 file, sai định dạng, quá 20 MB, suy ra MIME từ đuôi file khi trình duyệt không cung cấp.

### 4.2 Vitest + Testing Library cho phần nối dây

Mock `src/services/api.js`:

- `useChatStream.submit()` gọi `streamChat` với đúng `conversationId`, `mode`, `finalK`, `debug`.
- `stop()` gọi `abort` trên đúng controller và dẫn tới `STREAM_ABORTED`.
- `useAttachments.upload()` từ chối file sai định dạng **trước khi** gọi mạng.
- Một smoke test render `<App />` với API mock: hiện danh sách hội thoại, gõ và gửi câu hỏi, thấy text stream ra.

### 4.3 Thay đổi package.json

Thêm `vitest`, `jsdom`, `@testing-library/react`, `@testing-library/jest-dom` vào `devDependencies`. `npm test` đổi từ `node --test tests/*.test.js` sang `vitest run`.

Ba file test hiện có viết bằng `node:assert/strict` và `node:test`; Vitest chạy được cả ba không cần đổi runner API. Mốc đối chiếu hiện tại: `npm test` xanh 9/9.

**Một test sẽ đỏ, và đó là test cần thay.** `tests/chatModes.test.js` có case `"frontend sends the selected canonical mode to the streaming API"` không kiểm tra hành vi mà đọc `App.jsx` như văn bản rồi so khớp chuỗi:

```js
const appSource = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
assert.match(appSource, /mode:\s*inferenceMode/);
```

Refactor chuyển lời gọi `streamChat` sang `useChatStream.js` và đổi tên biến, nên assertion này đỏ dù hành vi không đổi một ly. Nó gắn chặt vào cách viết code chứ không vào contract.

Thay bằng test hành vi trong Vitest, mock `services/api.js`, gửi câu hỏi qua `useChatStream` rồi khẳng định `streamChat` nhận đúng `mode` đã chọn. Ý định gốc của test được giữ nguyên và kiểm chứng chặt hơn: test cũ vẫn xanh nếu ai đó viết `mode: inferenceMode` rồi truyền đi giá trị khác, test mới thì không.

Case còn lại trong cùng file (`"compact mode dropdown is inside the composer's left action group"`) cũng đọc văn bản nguồn, nhưng đọc `ChatInput.jsx` — file refactor này không đụng tới. Giữ nguyên, không nằm trong phạm vi.

## 5. Bảo toàn hành vi

Đây là refactor: không đổi giao diện, không đổi contract backend.

Quy trình theo TDD:

1. Viết test đặc tả hành vi **hiện tại** trước, chạy trên code cũ để xác nhận chúng xanh.
2. Chuyển logic sang reducer và hook.
3. Test phải xanh lại mà không sửa kỳ vọng. Bất kỳ test nào phải sửa kỳ vọng đều là dấu hiệu hành vi đã đổi, phải dừng lại xem xét.

Ngoại lệ duy nhất đã biết trước là case so khớp văn bản nguồn `App.jsx` nêu ở mục 4.3. Nó đỏ vì bám vào cách viết code, không phải vì hành vi đổi. Ngoài case đó, quy tắc ở bước 3 áp dụng nghiêm ngặt — không được nới thêm ngoại lệ nào giữa chừng mà không dừng lại xem xét.

Kiểm chứng cuối: `npm run frontend:lint` và `npm run frontend:build` sạch, `npm test` xanh, và số test không giảm so với mốc 9 test hiện tại.

## 6. Sửa kèm: `ensureActiveConversation` có thể trả về null

`createNewConversation()` trả về `null` khi `isRunning`, nhưng `ensureActiveConversation()` đọc thẳng `conversation.id` từ giá trị trả về:

```js
const ensureActiveConversation = async () => {
  if (activeConversationId) return activeConversationId;
  const conversation = await createNewConversation();
  return conversation.id;          // TypeError nếu createNewConversation trả null
};
```

`handleSubmit` chặn `isRunning` từ đầu nên an toàn. `handleFilesSelected` thì **không** chặn, nên về nguyên tắc đường dẫn này chạm tới được khi người dùng thả file trong lúc đang stream và chưa có hội thoại nào. Trên thực tế rất khó chạm vì `handleSubmit` luôn tạo hội thoại trước khi stream bắt đầu, nhưng đây là mìn chờ.

Sửa: `ensureActiveConversation` ném lỗi có thông báo rõ ràng khi không tạo được hội thoại, và `useAttachments.upload` chặn khi `isRunning` giống các đường dẫn khác. Kèm test.

Sửa trong cùng nhánh vì refactor đi ngang qua đúng đoạn code này.

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Đổi `npm test` ảnh hưởng người khác trong repo và CI | Ba file test cũ chạy được dưới Vitest không đổi runner API; mốc đối chiếu 9/9 xanh |
| Repo có thói quen viết test so khớp văn bản nguồn, loại test này đỏ theo mọi refactor | Case đã biết được thay bằng test hành vi (mục 4.3); test mới viết trong nhánh này không được so khớp văn bản nguồn |
| Refactor lớn làm lọt bug hành vi | Test đặc tả hành vi cũ được viết trước khi chuyển code |
| Diff lớn khó review | Chia commit theo từng đơn vị: hạ tầng test, hook độc lập, reducer, hook stream, hook upload, dọn App.jsx |
