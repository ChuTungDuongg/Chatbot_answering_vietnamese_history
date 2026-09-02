# Frontend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tách `frontend/src/App.jsx` (705 dòng, 14 `useState`) thành một reducer thuần cùng các hook mỏng, và dựng nền test cho phần logic hiện chưa có test nào.

**Architecture:** Toàn bộ state nghiệp vụ dồn vào `chatSessionReducer` — hàm thuần, không import React, không chạm `import.meta.env`. Các hook (`useChatSession`, `useChatStream`, `useAttachments`) chỉ gọi API rồi `dispatch`, không giữ state riêng. `App.jsx` còn lại phần render và nối dây. State thuần UI (`sidebarOpen`, `sourcesOpen`, `question`, `conversationToDelete`, `isDeletingConversation`) ở nguyên trong `App.jsx`.

**Tech Stack:** React 19, Vite 8, `node:test` cho module thuần, Vitest 4 + Testing Library cho hook và component.

**Spec:** [`docs/superpowers/specs/2026-09-02-frontend-refactor-design.md`](../specs/2026-09-02-frontend-refactor-design.md)

## Global Constraints

- Mọi lệnh chạy trong thư mục `frontend/`.
- **Không đổi giao diện, không đổi contract với backend.** Đây là refactor.
- Module trong `src/state/` và `src/config/` **không được import** trực tiếp hay gián tiếp bất cứ thứ gì chạm `import.meta.env` (cụ thể là `src/services/api.js`). Test của chúng chạy bằng `node` trần nên vi phạm sẽ vỡ ngay lúc import.
- Test module thuần: đặt tại `tests/*.test.js`, viết bằng `node:test` + `node:assert/strict`.
- Test hook/component: đặt tại `tests/ui/*.test.jsx`, viết bằng API của Vitest.
- Vitest **không** chạy được test viết bằng `node:test` (đã kiểm chứng: đếm 0 test, fail suite). Hai runner tách nhau bằng đường dẫn, không được trộn.
- Không đụng `src/App.css`. Không chuyển sang TypeScript. Không thêm tính năng.
- Không viết test mới kiểu so khớp văn bản nguồn (`readFile` rồi `assert.match`).
- Mốc đối chiếu: `npm test` hiện xanh 9/9. Sau mỗi task số test không được giảm.
- Commit message tiếng Việt, không kèm dòng co-author.

---

### Task 1: Dựng hạ tầng Vitest

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.js`
- Create: `frontend/tests/ui/setup.js`
- Test: `frontend/tests/ui/smoke.test.jsx`

**Interfaces:**
- Consumes: không có.
- Produces: lệnh `npm test` chạy `node --test tests/*.test.js && vitest run`; thư mục `tests/ui/` là nơi đặt mọi test cần DOM.

- [ ] **Step 1: Ghi lại mốc đối chiếu**

Run: `npm test`
Expected: PASS, `ℹ pass 9` và `ℹ fail 0`. Ghi con số 9 lại, các task sau đối chiếu với nó.

- [ ] **Step 2: Cài dependency**

```bash
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom
```

- [ ] **Step 3: Tạo `vitest.config.js`**

`include` bắt buộc phải giới hạn ở `tests/ui/`. Thiếu nó, Vitest sẽ vơ luôn ba file `tests/*.test.js` viết bằng `node:test` rồi fail với `No test suite found in file`.

```js
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    include: ["tests/ui/**/*.test.{js,jsx}"],
    environment: "jsdom",
    setupFiles: ["tests/ui/setup.js"],
    globals: false,
    restoreMocks: true,
  },
});
```

- [ ] **Step 4: Tạo `tests/ui/setup.js`**

```js
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
```

- [ ] **Step 5: Viết test smoke để chứng minh hạ tầng chạy**

Create `frontend/tests/ui/smoke.test.jsx`:

```jsx
import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";

function Hello({ name }) {
  return <p>Xin chào {name}</p>;
}

test("hạ tầng Vitest render được component React", () => {
  render(<Hello name="Sử Việt" />);
  expect(screen.getByText("Xin chào Sử Việt")).toBeInTheDocument();
});
```

- [ ] **Step 6: Chạy Vitest trực tiếp để kiểm tra `include`**

Run: `npx vitest run`
Expected: PASS 1 test. Nếu báo `No test suite found in file` ở `tests/chatModes.test.js` thì `include` ở Step 3 bị sai — Vitest đang vơ nhầm file `node:test`. Sửa trước khi đi tiếp.

- [ ] **Step 7: Đổi script `test` trong `package.json`**

```json
"test": "node --test tests/*.test.js && vitest run"
```

- [ ] **Step 8: Chạy cả hai runner**

Run: `npm test`
Expected: `node --test` báo `pass 9 / fail 0`, rồi Vitest báo `Test Files 1 passed`, `Tests 1 passed`.

- [ ] **Step 9: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.js frontend/tests/ui/
git commit -m "test: dựng hạ tầng Vitest + Testing Library cho frontend"
```

---

### Task 2: Tách hằng số thông báo sang `config/messages.js`

**Files:**
- Create: `frontend/src/config/messages.js`
- Modify: `frontend/src/services/api.js:5`
- Test: `frontend/tests/messages.test.js`

**Interfaces:**
- Consumes: không có.
- Produces: `src/config/messages.js` export các hằng số chuỗi dưới đây. `src/services/api.js` re-export `EVIDENCE_CONTRACT_FAILURE_MESSAGE` nên mọi import cũ vẫn chạy.

- [ ] **Step 1: Viết test chứng minh module thuần import được ngoài Vite**

Create `frontend/tests/messages.test.js`:

```js
import assert from "node:assert/strict";
import test from "node:test";

import {
  ANSWER_FAILURE_MESSAGE,
  ANSWER_STOPPED_MESSAGE,
  EVIDENCE_CONTRACT_FAILURE_MESSAGE,
} from "../src/config/messages.js";

test("hằng số thông báo import được bằng node trần, không cần Vite", () => {
  assert.equal(
    EVIDENCE_CONTRACT_FAILURE_MESSAGE,
    "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
  );
  assert.equal(ANSWER_FAILURE_MESSAGE, "Không thể hoàn tất câu trả lời.");
  assert.equal(ANSWER_STOPPED_MESSAGE, "Đã dừng tạo câu trả lời.");
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `node --test tests/messages.test.js`
Expected: FAIL, `Cannot find module` trỏ tới `src/config/messages.js`.

- [ ] **Step 3: Tạo `src/config/messages.js`**

Chuỗi phải sao chép **nguyên văn** từ `App.jsx` và `api.js` hiện tại, kể cả dấu câu.

```js
export const EVIDENCE_CONTRACT_FAILURE_MESSAGE =
  "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.";
export const ANSWER_FAILURE_MESSAGE = "Không thể hoàn tất câu trả lời.";
export const ANSWER_STOPPED_MESSAGE = "Đã dừng tạo câu trả lời.";
export const STREAM_FAILURE_MESSAGE = "Backend không thể hoàn tất yêu cầu.";
export const BACKEND_UNREACHABLE_MESSAGE = "Không thể kết nối tới backend.";
export const CONVERSATION_CREATE_FAILURE_MESSAGE = "Không thể tạo cuộc trò chuyện.";
export const CONVERSATION_CREATE_NEW_FAILURE_MESSAGE = "Không thể tạo cuộc trò chuyện mới.";
export const CONVERSATION_LOAD_FAILURE_MESSAGE = "Không thể tải cuộc trò chuyện.";
export const CONVERSATION_RENAME_FAILURE_MESSAGE = "Không thể đổi tên cuộc trò chuyện.";
export const CONVERSATION_DELETE_FAILURE_MESSAGE = "Không thể xóa cuộc trò chuyện.";
export const ATTACHMENT_DELETE_FAILURE_MESSAGE = "Không thể xóa tài liệu.";
export const MISSING_CONVERSATION_ID_MESSAGE = "Backend không trả về conversation ID.";
```

- [ ] **Step 4: Đổi `api.js` thành re-export**

Thay dòng 5 của `src/services/api.js`:

```js
export const EVIDENCE_CONTRACT_FAILURE_MESSAGE = "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.";
```

bằng:

```js
export { EVIDENCE_CONTRACT_FAILURE_MESSAGE } from "../config/messages.js";
```

- [ ] **Step 5: Chạy test**

Run: `npm test`
Expected: `node --test` báo `pass 10` (9 cũ + 1 mới), `fail 0`. Vitest pass 1.

- [ ] **Step 6: Xác nhận build không vỡ**

Run: `npm run build`
Expected: build thành công, không có lỗi resolve import.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/config/messages.js frontend/src/services/api.js frontend/tests/messages.test.js
git commit -m "refactor: tách hằng số thông báo ra khỏi tầng API"
```

---

### Task 3: Trích các helper thuần sang `state/ids.js` và `state/normalizers.js`

**Files:**
- Create: `frontend/src/state/ids.js`
- Create: `frontend/src/state/normalizers.js`
- Test: `frontend/tests/normalizers.test.js`

**Interfaces:**
- Consumes: không có.
- Produces:
  - `createLocalId(prefix: string): string`
  - `normalizeConversationList(payload): Array`
  - `normalizeConversationDetail(payload): { conversation, messages, attachments }`
  - `getSources(data): Array`
  - `getLatestSources(messages): Array`

- [ ] **Step 1: Viết test**

Create `frontend/tests/normalizers.test.js`:

```js
import assert from "node:assert/strict";
import test from "node:test";

import { createLocalId } from "../src/state/ids.js";
import {
  getLatestSources,
  getSources,
  normalizeConversationDetail,
  normalizeConversationList,
} from "../src/state/normalizers.js";

test("createLocalId gắn prefix và không trùng nhau", () => {
  const first = createLocalId("user");
  const second = createLocalId("user");
  assert.ok(first.startsWith("user-"));
  assert.notEqual(first, second);
});

test("normalizeConversationList chấp nhận mảng trần và các hình dạng bọc", () => {
  assert.deepEqual(normalizeConversationList([{ id: "a" }]), [{ id: "a" }]);
  assert.deepEqual(normalizeConversationList({ items: [{ id: "b" }] }), [{ id: "b" }]);
  assert.deepEqual(normalizeConversationList({ conversations: [{ id: "c" }] }), [{ id: "c" }]);
  assert.deepEqual(normalizeConversationList(null), []);
});

test("normalizeConversationDetail lấy messages và attachments từ mọi hình dạng", () => {
  const wrapped = normalizeConversationDetail({
    conversation: { id: "a" },
    messages: [{ id: "m1" }],
    attachments: [{ id: "f1" }],
  });
  assert.deepEqual(wrapped, {
    conversation: { id: "a" },
    messages: [{ id: "m1" }],
    attachments: [{ id: "f1" }],
  });

  const nested = normalizeConversationDetail({
    conversation: { id: "b", messages: [{ id: "m2" }], attachments: [] },
  });
  assert.deepEqual(nested.messages, [{ id: "m2" }]);

  const empty = normalizeConversationDetail(null);
  assert.deepEqual(empty, { conversation: {}, messages: [], attachments: [] });
});

test("getSources đọc được các khoá mà backend dùng", () => {
  assert.deepEqual(getSources([{ id: 1 }]), [{ id: 1 }]);
  assert.deepEqual(getSources({ items: [{ id: 2 }] }), [{ id: 2 }]);
  assert.deepEqual(getSources({ sources: [{ id: 3 }] }), [{ id: 3 }]);
  assert.deepEqual(getSources({ final_context: [{ id: 4 }] }), [{ id: 4 }]);
  assert.deepEqual(getSources({ retrieval: { final_context: [{ id: 5 }] } }), [{ id: 5 }]);
  assert.deepEqual(getSources(undefined), []);
});

test("getLatestSources lấy nguồn của câu trả lời gần nhất có nguồn", () => {
  const messages = [
    { role: "assistant", sources: [{ id: "cũ" }] },
    { role: "user", sources: [] },
    { role: "assistant", sources: [{ id: "mới" }] },
    { role: "assistant", sources: [] },
  ];
  assert.deepEqual(getLatestSources(messages), [{ id: "mới" }]);
  assert.deepEqual(getLatestSources([]), []);
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `node --test tests/normalizers.test.js`
Expected: FAIL, `Cannot find module` trỏ tới `src/state/ids.js`.

- [ ] **Step 3: Tạo `src/state/ids.js`**

Chép nguyên hàm từ `App.jsx:74-77`.

```js
export function createLocalId(prefix) {
  const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${id}`;
}
```

- [ ] **Step 4: Tạo `src/state/normalizers.js`**

Chép nguyên bốn hàm từ `App.jsx:79-102`.

```js
export function normalizeConversationList(payload) {
  if (Array.isArray(payload)) return payload;
  return payload?.items ?? payload?.conversations ?? [];
}

export function normalizeConversationDetail(payload) {
  const conversation = payload?.conversation ?? payload ?? {};
  return {
    conversation,
    messages: payload?.messages ?? conversation.messages ?? [],
    attachments: payload?.attachments ?? conversation.attachments ?? [],
  };
}

export function getSources(data) {
  if (Array.isArray(data)) return data;
  return data?.items ?? data?.sources ?? data?.final_context ?? data?.retrieval?.final_context ?? [];
}

export function getLatestSources(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.sources?.length) return message.sources;
  }
  return [];
}
```

- [ ] **Step 5: Chạy test**

Run: `npm test`
Expected: `node --test` fail 0. Vitest pass 1.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/state/ frontend/tests/normalizers.test.js
git commit -m "refactor: trích helper chuẩn hoá payload và sinh id ra module thuần"
```

---

### Task 4: `state/uploadQueue.js` — validate file thuần

**Files:**
- Create: `frontend/src/state/uploadQueue.js`
- Test: `frontend/tests/uploadQueue.test.js`

**Interfaces:**
- Consumes: `createLocalId` từ `src/state/ids.js`.
- Produces:
  - `MAX_FILE_SIZE: number`, `MAX_FILES_PER_UPLOAD: number`, `ALLOWED_MIME_TYPES: Set<string>`
  - `normalizeUploadFile(file: File): File`
  - `validateUploadSelection(selectedFiles: File[]): { error: string | null, files: File[] }`
  - `createUploadItems(files: File[]): Array<{ id, name, type, size_bytes, status, file }>`

Thứ tự kiểm tra phải giữ **đúng như hiện tại**: quá số lượng trước, rồi sai định dạng, rồi quá dung lượng. Đổi thứ tự là đổi thông báo mà người dùng nhìn thấy.

- [ ] **Step 1: Viết test**

Create `frontend/tests/uploadQueue.test.js`:

```js
import assert from "node:assert/strict";
import test from "node:test";

import {
  createUploadItems,
  normalizeUploadFile,
  validateUploadSelection,
} from "../src/state/uploadQueue.js";

function makeFile(name, type, size = 1024) {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

test("suy ra MIME từ đuôi file khi trình duyệt không cung cấp", () => {
  const normalized = normalizeUploadFile(makeFile("tulieu.pdf", ""));
  assert.equal(normalized.type, "application/pdf");
  assert.equal(normalized.name, "tulieu.pdf");
});

test("giữ nguyên file khi MIME đã hợp lệ", () => {
  const original = makeFile("anh.png", "image/png");
  assert.equal(normalizeUploadFile(original), original);
});

test("từ chối khi chọn quá 5 file, và báo lỗi số lượng trước mọi lỗi khác", () => {
  const files = Array.from({ length: 6 }, (_, index) => makeFile(`f${index}.exe`, "application/x-msdownload"));
  const result = validateUploadSelection(files);
  assert.equal(result.error, "Mỗi lần chỉ có thể tải tối đa 5 file.");
  assert.deepEqual(result.files, []);
});

test("từ chối định dạng không hỗ trợ và nêu đúng tên file", () => {
  const result = validateUploadSelection([makeFile("virus.exe", "application/x-msdownload")]);
  assert.equal(
    result.error,
    "Không hỗ trợ định dạng của virus.exe. Chỉ nhận PDF, PNG, JPEG và WebP.",
  );
});

test("từ chối file quá 20 MB", () => {
  const result = validateUploadSelection([makeFile("to.pdf", "application/pdf", 21 * 1024 * 1024)]);
  assert.equal(result.error, "to.pdf vượt quá giới hạn 20 MB.");
});

test("chấp nhận tuyển chọn hợp lệ", () => {
  const result = validateUploadSelection([makeFile("ok.pdf", "application/pdf")]);
  assert.equal(result.error, null);
  assert.equal(result.files.length, 1);
});

test("createUploadItems dựng hàng đợi có id riêng và trạng thái queued", () => {
  const items = createUploadItems([makeFile("a.pdf", "application/pdf", 2048)]);
  assert.equal(items.length, 1);
  assert.equal(items[0].status, "queued");
  assert.equal(items[0].name, "a.pdf");
  assert.equal(items[0].size_bytes, 2048);
  assert.ok(items[0].id.startsWith("upload-"));
  assert.ok(items[0].file instanceof File);
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `node --test tests/uploadQueue.test.js`
Expected: FAIL, `Cannot find module`.

- [ ] **Step 3: Tạo `src/state/uploadQueue.js`**

```js
import { createLocalId } from "./ids.js";

export const ALLOWED_MIME_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/webp",
]);

export const MIME_BY_EXTENSION = {
  pdf: "application/pdf",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
};

export const MAX_FILE_SIZE = 20 * 1024 * 1024;
export const MAX_FILES_PER_UPLOAD = 5;

export function normalizeUploadFile(file) {
  if (ALLOWED_MIME_TYPES.has(file.type)) return file;

  const extension = file.name.split(".").pop()?.toLowerCase();
  const inferredType = MIME_BY_EXTENSION[extension];
  if (!inferredType) return file;

  return new File([file], file.name, { type: inferredType, lastModified: file.lastModified });
}

export function validateUploadSelection(selectedFiles) {
  const files = selectedFiles.slice(0, MAX_FILES_PER_UPLOAD).map(normalizeUploadFile);

  if (selectedFiles.length > MAX_FILES_PER_UPLOAD) {
    return { error: `Mỗi lần chỉ có thể tải tối đa ${MAX_FILES_PER_UPLOAD} file.`, files: [] };
  }

  const invalidFile = files.find((file) => !ALLOWED_MIME_TYPES.has(file.type));
  if (invalidFile) {
    return {
      error: `Không hỗ trợ định dạng của ${invalidFile.name}. Chỉ nhận PDF, PNG, JPEG và WebP.`,
      files: [],
    };
  }

  const oversizedFile = files.find((file) => file.size > MAX_FILE_SIZE);
  if (oversizedFile) {
    return { error: `${oversizedFile.name} vượt quá giới hạn 20 MB.`, files: [] };
  }

  return { error: null, files };
}

export function createUploadItems(files) {
  return files.map((file) => ({
    id: createLocalId("upload"),
    name: file.name,
    type: file.type,
    size_bytes: file.size,
    status: "queued",
    file,
  }));
}
```

- [ ] **Step 4: Chạy test**

Run: `npm test`
Expected: fail 0 ở cả hai runner.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/state/uploadQueue.js frontend/tests/uploadQueue.test.js
git commit -m "refactor: trích logic validate upload ra module thuần"
```

---

### Task 5: Reducer — khung và nhóm action hội thoại

**Files:**
- Create: `frontend/src/state/chatSessionReducer.js`
- Test: `frontend/tests/chatSessionReducer.conversations.test.js`

**Interfaces:**
- Consumes: `getLatestSources` từ `src/state/normalizers.js`.
- Produces:
  - `initialChatSessionState` — object đúng hình dạng nêu trong spec mục 3.2
  - `chatSessionReducer(state, action)`
  - `ACTIVE_STATUSES: Set<string>`, `isRunningStatus(status: string): boolean`
  - Xử lý các action: `BOOTSTRAPPED`, `BOOTSTRAP_FAILED`, `CONVERSATION_LOADING`, `CONVERSATION_LOADED`, `CONVERSATION_LOAD_FAILED`, `CONVERSATION_CREATED`, `CONVERSATION_RENAMED`, `CONVERSATION_DELETED`, `CONVERSATIONS_SET`, `SOURCES_SHOWN`, `ERROR_SET`

**Ghi chú lệch so với spec:** spec không liệt kê `BOOTSTRAP_FAILED`, `CONVERSATIONS_SET` và `SOURCES_SHOWN`. Cả ba đều cần: `BOOTSTRAP_FAILED` để spinner không kẹt khi lần tải đầu hỏng, `CONVERSATIONS_SET` cho lần làm mới danh sách sau khi xoá tài liệu (`App.jsx:530`), `SOURCES_SHOWN` cho thao tác bấm xem nguồn của một tin nhắn cũ (`App.jsx:540-544`).

- [ ] **Step 1: Viết test**

Create `frontend/tests/chatSessionReducer.conversations.test.js`:

```js
import assert from "node:assert/strict";
import test from "node:test";

import {
  chatSessionReducer,
  initialChatSessionState,
  isRunningStatus,
} from "../src/state/chatSessionReducer.js";

function reduce(actions, state = initialChatSessionState) {
  return actions.reduce(chatSessionReducer, state);
}

test("BOOTSTRAPPED tắt cờ đang tải và nạp hội thoại đầu tiên", () => {
  const state = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "c1" }, { id: "c2" }],
    activeConversationId: "c1",
    detail: {
      messages: [{ id: "m1", role: "assistant", sources: [{ id: "s1" }] }],
      attachments: [{ id: "f1" }],
    },
  }]);

  assert.equal(state.isLoadingConversations, false);
  assert.equal(state.activeConversationId, "c1");
  assert.deepEqual(state.attachments, [{ id: "f1" }]);
  assert.deepEqual(state.sources, [{ id: "s1" }]);
});

test("BOOTSTRAPPED khi chưa có hội thoại nào thì không đụng tới thread", () => {
  const state = reduce([{ type: "BOOTSTRAPPED", conversations: [] }]);
  assert.equal(state.isLoadingConversations, false);
  assert.equal(state.activeConversationId, null);
  assert.deepEqual(state.messages, []);
});

test("BOOTSTRAP_FAILED vẫn phải tắt cờ đang tải, nếu không spinner kẹt vĩnh viễn", () => {
  const state = reduce([{ type: "BOOTSTRAP_FAILED", message: "Sập mạng" }]);
  assert.equal(state.isLoadingConversations, false);
  assert.equal(state.error, "Sập mạng");
});

test("CONVERSATION_LOAD_FAILED tắt cờ đang tải và đặt lỗi", () => {
  const state = reduce([
    { type: "CONVERSATION_LOADING" },
    { type: "CONVERSATION_LOAD_FAILED", message: "Hỏng rồi" },
  ]);
  assert.equal(state.isLoadingConversation, false);
  assert.equal(state.error, "Hỏng rồi");
});

test("CONVERSATION_LOADED đổi thread và đặt lại trạng thái stream", () => {
  const state = reduce([
    { type: "STREAM_ERROR", messageId: "x", message: "lỗi cũ" },
    {
      type: "CONVERSATION_LOADED",
      conversationId: "c9",
      detail: { messages: [{ id: "m", role: "assistant", sources: [] }], attachments: [] },
    },
  ]);

  assert.equal(state.activeConversationId, "c9");
  assert.equal(state.status, "idle");
  assert.equal(state.streamFailed, false);
  assert.equal(state.isLoadingConversation, false);
});

test("CONVERSATION_CREATED đưa hội thoại lên đầu và dọn sạch màn hình", () => {
  const state = reduce([
    { type: "BOOTSTRAPPED", conversations: [{ id: "cũ" }] },
    { type: "CONVERSATION_CREATED", conversation: { id: "mới" } },
  ]);

  assert.deepEqual(state.conversations.map((item) => item.id), ["mới", "cũ"]);
  assert.equal(state.activeConversationId, "mới");
  assert.deepEqual(state.messages, []);
  assert.deepEqual(state.attachments, []);
  assert.deepEqual(state.sources, []);
});

test("CONVERSATION_CREATED không nhân đôi khi backend trả lại id đã có", () => {
  const state = reduce([
    { type: "BOOTSTRAPPED", conversations: [{ id: "a" }, { id: "b" }] },
    { type: "CONVERSATION_CREATED", conversation: { id: "b", title: "Đổi tên" } },
  ]);
  assert.deepEqual(state.conversations.map((item) => item.id), ["b", "a"]);
});

test("CONVERSATION_RENAMED chỉ vá đúng một mục", () => {
  const state = reduce([
    { type: "BOOTSTRAPPED", conversations: [{ id: "a", title: "A" }, { id: "b", title: "B" }] },
    { type: "CONVERSATION_RENAMED", conversationId: "b", patch: { title: "B mới" } },
  ]);
  assert.deepEqual(state.conversations.map((item) => item.title), ["A", "B mới"]);
});

test("xoá hội thoại đang mở thì dọn thread", () => {
  const start = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "a" }, { id: "b" }],
    activeConversationId: "a",
    detail: { messages: [{ id: "m", role: "assistant", sources: [{ id: "s" }] }], attachments: [{ id: "f" }] },
  }]);

  const state = chatSessionReducer(start, { type: "CONVERSATION_DELETED", conversationId: "a" });

  assert.deepEqual(state.conversations.map((item) => item.id), ["b"]);
  assert.equal(state.activeConversationId, null);
  assert.deepEqual(state.messages, []);
  assert.deepEqual(state.attachments, []);
  assert.deepEqual(state.sources, []);
});

test("xoá hội thoại khác thì thread đang mở không suy suyển", () => {
  const start = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "a" }, { id: "b" }],
    activeConversationId: "a",
    detail: { messages: [{ id: "m", role: "assistant", sources: [] }], attachments: [] },
  }]);

  const state = chatSessionReducer(start, { type: "CONVERSATION_DELETED", conversationId: "b" });

  assert.equal(state.activeConversationId, "a");
  assert.equal(state.messages.length, 1);
});

test("SOURCES_SHOWN chỉ đổi nguồn đang xem, không đụng tin nhắn", () => {
  const start = reduce([{
    type: "BOOTSTRAPPED",
    conversations: [{ id: "a" }],
    activeConversationId: "a",
    detail: { messages: [{ id: "m", role: "assistant", sources: [{ id: "mới" }] }], attachments: [] },
  }]);

  const state = chatSessionReducer(start, { type: "SOURCES_SHOWN", sources: [{ id: "cũ" }] });

  assert.deepEqual(state.sources, [{ id: "cũ" }]);
  assert.deepEqual(state.messages[0].sources, [{ id: "mới" }]);
});

test("isRunningStatus nhận diện đúng các trạng thái đang chạy", () => {
  assert.equal(isRunningStatus("streaming"), true);
  assert.equal(isRunningStatus("central_tools"), true);
  assert.equal(isRunningStatus("idle"), false);
  assert.equal(isRunningStatus("done"), false);
  assert.equal(isRunningStatus("error"), false);
});

test("action lạ không làm thay đổi state", () => {
  const state = chatSessionReducer(initialChatSessionState, { type: "KHÔNG_TỒN_TẠI" });
  assert.equal(state, initialChatSessionState);
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `node --test tests/chatSessionReducer.conversations.test.js`
Expected: FAIL, `Cannot find module`.

- [ ] **Step 3: Tạo `src/state/chatSessionReducer.js`**

`ACTIVE_STATUSES` chép nguyên từ `App.jsx:39-45`.

```js
import { getLatestSources } from "./normalizers.js";

export const ACTIVE_STATUSES = new Set([
  "processing", "retrieval_started", "reranking", "generating", "validating", "validated", "streaming",
  "hybrid_retrieval", "hybrid_answering",
  "three_llm_research", "three_llm_evidence", "three_llm_answering",
  "central_analyzing", "central_tools", "central_answering",
]);

export function isRunningStatus(status) {
  return ACTIVE_STATUSES.has(status);
}

export const initialChatSessionState = {
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
};

const CLEARED_THREAD = { messages: [], attachments: [], sources: [] };

export function chatSessionReducer(state, action) {
  switch (action.type) {
    case "BOOTSTRAPPED": {
      const base = { ...state, conversations: action.conversations, isLoadingConversations: false };
      if (!action.detail) return base;
      return {
        ...base,
        activeConversationId: action.activeConversationId ?? null,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
      };
    }

    case "BOOTSTRAP_FAILED":
      return { ...state, isLoadingConversations: false, error: action.message };

    case "CONVERSATION_LOADING":
      return { ...state, isLoadingConversation: true, error: "" };

    case "CONVERSATION_LOADED":
      return {
        ...state,
        activeConversationId: action.conversationId,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
        status: "idle",
        streamFailed: false,
        isLoadingConversation: false,
      };

    case "CONVERSATION_LOAD_FAILED":
      return { ...state, isLoadingConversation: false, error: action.message };

    case "CONVERSATION_CREATED":
      return {
        ...state,
        ...CLEARED_THREAD,
        conversations: [
          action.conversation,
          ...state.conversations.filter((item) => item.id !== action.conversation.id),
        ],
        activeConversationId: action.conversation.id,
        status: "idle",
        streamFailed: false,
      };

    case "CONVERSATION_RENAMED":
      return {
        ...state,
        conversations: state.conversations.map((item) =>
          item.id === action.conversationId ? { ...item, ...action.patch } : item),
      };

    case "CONVERSATION_DELETED": {
      const conversations = state.conversations.filter((item) => item.id !== action.conversationId);
      if (state.activeConversationId !== action.conversationId) return { ...state, conversations };
      return { ...state, ...CLEARED_THREAD, conversations, activeConversationId: null };
    }

    case "CONVERSATIONS_SET":
      return { ...state, conversations: action.conversations };

    case "SOURCES_SHOWN":
      return { ...state, sources: action.sources };

    case "ERROR_SET":
      return { ...state, error: action.message };

    default:
      return state;
  }
}
```

- [ ] **Step 4: Chạy test, xác nhận xanh trừ một test cố ý dùng action của task sau**

Run: `node --test tests/chatSessionReducer.conversations.test.js`
Expected: PASS toàn bộ. Test `CONVERSATION_LOADED` có dispatch `STREAM_ERROR` — action đó chưa xử lý nên rơi vào `default` và trả về state cũ, test vẫn xanh vì nó chỉ kiểm tra kết quả sau `CONVERSATION_LOADED`.

- [ ] **Step 5: Chạy toàn bộ**

Run: `npm test`
Expected: fail 0 ở cả hai runner.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/state/chatSessionReducer.js frontend/tests/chatSessionReducer.conversations.test.js
git commit -m "refactor: thêm reducer phiên chat với nhóm action hội thoại"
```

---

### Task 6: Reducer — nhóm action streaming

**Files:**
- Modify: `frontend/src/state/chatSessionReducer.js`
- Test: `frontend/tests/chatSessionReducer.stream.test.js`

**Interfaces:**
- Consumes: `chatSessionReducer`, `initialChatSessionState` từ Task 5.
- Produces: xử lý `MESSAGES_APPENDED`, `STREAM_STATUS`, `STREAM_DELTA`, `STREAM_SOURCES`, `STREAM_DEBUG`, `STREAM_ERROR`, `STREAM_DONE`, `STREAM_ABORTED`, `STREAM_SYNCED`.

**Ghi chú lệch so với spec:** spec liệt kê một action `SYNCED` duy nhất. Khi viết plan mới thấy hai chỗ gọi đồng bộ có hành vi **khác nhau**: sau stream thì ghi đè cả `messages`/`attachments`/`sources`/`status`, còn sau upload thì chỉ ghi `conversations` và `attachments` (xem `App.jsx:522-535`). Gộp làm một sẽ đổi hành vi upload. Nên tách thành `STREAM_SYNCED` (task này) và `ATTACHMENTS_SYNCED` (Task 7).

- [ ] **Step 1: Viết test**

Create `frontend/tests/chatSessionReducer.stream.test.js`:

```js
import assert from "node:assert/strict";
import test from "node:test";

import { chatSessionReducer, initialChatSessionState } from "../src/state/chatSessionReducer.js";

const ASSISTANT_ID = "assistant-1";

function startedStream() {
  return chatSessionReducer(initialChatSessionState, {
    type: "MESSAGES_APPENDED",
    messages: [
      { id: "user-1", role: "user", content: "Hỏi gì đó", sources: [], status: "done" },
      { id: ASSISTANT_ID, role: "assistant", content: "", sources: [], status: "processing" },
    ],
  });
}

function reduce(state, actions) {
  return actions.reduce(chatSessionReducer, state);
}

test("MESSAGES_APPENDED thêm cặp tin nhắn, xoá lỗi cũ và bật trạng thái processing", () => {
  const state = reduce(initialChatSessionState, [
    { type: "ERROR_SET", message: "lỗi lượt trước" },
    { type: "MESSAGES_APPENDED", messages: [{ id: "u", role: "user", content: "x" }] },
  ]);

  assert.equal(state.messages.length, 1);
  assert.equal(state.error, "");
  assert.equal(state.status, "processing");
  assert.equal(state.streamFailed, false);
});

test("nhiều STREAM_DELTA nối đúng thứ tự vào đúng tin nhắn", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Chiến thắng " },
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Bạch Đằng " },
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "năm 938." },
  ]);

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.equal(assistant.content, "Chiến thắng Bạch Đằng năm 938.");
  assert.equal(assistant.status, "streaming");
  assert.equal(state.status, "streaming");
  assert.equal(state.messages.find((message) => message.id === "user-1").content, "Hỏi gì đó");
});

test("STREAM_SOURCES ghi nguồn vào cả tin nhắn lẫn drawer", () => {
  const sources = [{ id: "s1", text: "trích đoạn" }];
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_SOURCES", messageId: ASSISTANT_ID, sources,
  });

  assert.deepEqual(state.sources, sources);
  assert.deepEqual(state.messages.find((message) => message.id === ASSISTANT_ID).sources, sources);
});

test("STREAM_ERROR loại evidence contract dùng đúng thông báo riêng", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_ERROR",
    messageId: ASSISTANT_ID,
    message: "Evidence critic từ chối",
    kind: "evidence_contract_error",
  });

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.equal(
    assistant.content,
    "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
  );
  assert.equal(state.error, "Evidence critic từ chối");
  assert.equal(state.streamFailed, true);
});

test("STREAM_ERROR loại khác dùng thông báo mặc định", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Backend sập",
  });

  assert.equal(
    state.messages.find((message) => message.id === ASSISTANT_ID).content,
    "Không thể hoàn tất câu trả lời.",
  );
});

test("STREAM_ERROR không nuốt mất phần text đã stream được", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Nửa câu trả lời" },
    { type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Đứt giữa chừng" },
  ]);

  assert.equal(
    state.messages.find((message) => message.id === ASSISTANT_ID).content,
    "Nửa câu trả lời",
  );
});

test("STREAM_DONE đến sau STREAM_ERROR không được ghi đè trạng thái lỗi", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Hỏng" },
    { type: "STREAM_DONE", messageId: ASSISTANT_ID },
  ]);

  assert.equal(state.status, "error");
  assert.equal(state.messages.find((message) => message.id === ASSISTANT_ID).status, "error");
});

test("STREAM_DONE của lượt sạch cho ra trạng thái done", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Xong" },
    { type: "STREAM_DONE", messageId: ASSISTANT_ID },
  ]);

  assert.equal(state.status, "done");
});

test("lượt mới đặt lại cờ lỗi của lượt trước", () => {
  const failed = reduce(startedStream(), [
    { type: "STREAM_ERROR", messageId: ASSISTANT_ID, message: "Hỏng" },
  ]);
  const fresh = chatSessionReducer(failed, {
    type: "MESSAGES_APPENDED",
    messages: [{ id: "assistant-2", role: "assistant", content: "", status: "processing" }],
  });
  const state = chatSessionReducer(fresh, { type: "STREAM_DONE", messageId: "assistant-2" });

  assert.equal(state.status, "done");
});

test("STREAM_ABORTED giữ lại text đã stream", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Đang viết dở" },
    { type: "STREAM_ABORTED", messageId: ASSISTANT_ID },
  ]);

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.equal(assistant.content, "Đang viết dở");
  assert.equal(assistant.status, "cancelled");
  assert.equal(state.status, "cancelled");
});

test("STREAM_ABORTED khi chưa kịp stream chữ nào thì đặt thông báo đã dừng", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_ABORTED", messageId: ASSISTANT_ID,
  });

  assert.equal(
    state.messages.find((message) => message.id === ASSISTANT_ID).content,
    "Đã dừng tạo câu trả lời.",
  );
});

test("STREAM_DEBUG gắn trace vào tin nhắn mà không đụng nội dung", () => {
  const state = reduce(startedStream(), [
    { type: "STREAM_DELTA", messageId: ASSISTANT_ID, delta: "Nội dung" },
    { type: "STREAM_DEBUG", messageId: ASSISTANT_ID, trace: { steps: 3 } },
  ]);

  const assistant = state.messages.find((message) => message.id === ASSISTANT_ID);
  assert.deepEqual(assistant.debug_trace, { steps: 3 });
  assert.equal(assistant.content, "Nội dung");
});

test("STREAM_SYNCED thay thread bằng bản chính chủ từ backend", () => {
  const state = chatSessionReducer(startedStream(), {
    type: "STREAM_SYNCED",
    conversations: [{ id: "c1" }],
    detail: {
      messages: [{ id: "m1", role: "assistant", sources: [{ id: "s9" }] }],
      attachments: [{ id: "f1" }],
    },
  });

  assert.deepEqual(state.conversations, [{ id: "c1" }]);
  assert.equal(state.messages.length, 1);
  assert.deepEqual(state.sources, [{ id: "s9" }]);
  assert.deepEqual(state.attachments, [{ id: "f1" }]);
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `node --test tests/chatSessionReducer.stream.test.js`
Expected: FAIL nhiều test — các action stream đang rơi vào `default` nên state không đổi.

- [ ] **Step 3: Thêm helper và các case stream vào reducer**

Thêm import ở đầu `src/state/chatSessionReducer.js`:

```js
import {
  ANSWER_FAILURE_MESSAGE,
  ANSWER_STOPPED_MESSAGE,
  EVIDENCE_CONTRACT_FAILURE_MESSAGE,
} from "../config/messages.js";
```

Thêm helper ngay trước `export function chatSessionReducer`:

```js
function patchMessage(state, messageId, patch) {
  return state.messages.map((message) => {
    if (message.id !== messageId) return message;
    return typeof patch === "function" ? patch(message) : { ...message, ...patch };
  });
}
```

Thêm các case sau vào `switch`, đặt trước `default`:

```js
    case "MESSAGES_APPENDED":
      return {
        ...state,
        messages: [...state.messages, ...action.messages],
        status: "processing",
        error: "",
        streamFailed: false,
      };

    case "STREAM_STATUS":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, { status: action.status, mode: action.mode }),
        status: action.status,
      };

    case "STREAM_DELTA":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content + action.delta,
          status: "streaming",
        })),
        status: "streaming",
      };

    case "STREAM_SOURCES":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, { sources: action.sources }),
        sources: action.sources,
      };

    case "STREAM_DEBUG":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, { debug_trace: action.trace }),
      };

    case "STREAM_ERROR": {
      const fallback = action.kind === "evidence_contract_error"
        ? EVIDENCE_CONTRACT_FAILURE_MESSAGE
        : ANSWER_FAILURE_MESSAGE;

      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content || fallback,
          status: "error",
          debug_trace: action.trace ?? message.debug_trace,
        })),
        status: "error",
        error: action.message,
        streamFailed: true,
      };
    }

    case "STREAM_DONE": {
      const status = state.streamFailed ? "error" : "done";
      return { ...state, messages: patchMessage(state, action.messageId, { status }), status };
    }

    case "STREAM_ABORTED":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content || ANSWER_STOPPED_MESSAGE,
          status: "cancelled",
        })),
        status: "cancelled",
      };

    case "STREAM_SYNCED":
      return {
        ...state,
        conversations: action.conversations,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
      };
```

- [ ] **Step 4: Chạy test**

Run: `node --test tests/chatSessionReducer.stream.test.js`
Expected: PASS toàn bộ.

- [ ] **Step 5: Chạy toàn bộ**

Run: `npm test`
Expected: fail 0 ở cả hai runner.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/state/chatSessionReducer.js frontend/tests/chatSessionReducer.stream.test.js
git commit -m "refactor: đưa logic streaming SSE vào reducer thuần"
```

---

### Task 7: Reducer — nhóm action upload và tài liệu

**Files:**
- Modify: `frontend/src/state/chatSessionReducer.js`
- Test: `frontend/tests/chatSessionReducer.uploads.test.js`

**Interfaces:**
- Consumes: reducer từ Task 5 và 6.
- Produces: xử lý `UPLOAD_QUEUED`, `UPLOAD_PROGRESS`, `UPLOAD_SETTLED`, `ATTACHMENT_REMOVED`, `ATTACHMENTS_SYNCED`.

- [ ] **Step 1: Viết test**

Create `frontend/tests/chatSessionReducer.uploads.test.js`:

```js
import assert from "node:assert/strict";
import test from "node:test";

import { chatSessionReducer, initialChatSessionState } from "../src/state/chatSessionReducer.js";

function reduce(actions, state = initialChatSessionState) {
  return actions.reduce(chatSessionReducer, state);
}

const QUEUED = [
  { id: "upload-1", name: "a.pdf", status: "queued" },
  { id: "upload-2", name: "b.pdf", status: "queued" },
];

test("UPLOAD_QUEUED đưa file vào hàng đợi và xoá lỗi cũ", () => {
  const state = reduce([
    { type: "ERROR_SET", message: "lỗi cũ" },
    { type: "UPLOAD_QUEUED", items: QUEUED },
  ]);

  assert.equal(state.pendingUploads.length, 2);
  assert.equal(state.error, "");
});

test("UPLOAD_PROGRESS chỉ đổi trạng thái đúng một file", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_PROGRESS", id: "upload-2", status: "processing" },
  ]);

  assert.deepEqual(state.pendingUploads.map((item) => item.status), ["queued", "processing"]);
});

test("UPLOAD_SETTLED thành công: rời hàng đợi và vào danh sách tài liệu", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_SETTLED", id: "upload-1", attachment: { id: "att-1", name: "a.pdf" } },
  ]);

  assert.deepEqual(state.pendingUploads.map((item) => item.id), ["upload-2"]);
  assert.deepEqual(state.attachments, [{ id: "att-1", name: "a.pdf" }]);
});

test("UPLOAD_SETTLED không nhân đôi tài liệu khi backend trả lại cùng id", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_SETTLED", id: "upload-1", attachment: { id: "att-1", name: "cũ" } },
    { type: "UPLOAD_SETTLED", id: "upload-2", attachment: { id: "att-1", name: "mới" } },
  ]);

  assert.equal(state.attachments.length, 1);
  assert.equal(state.attachments[0].name, "mới");
});

test("UPLOAD_SETTLED thất bại: rời hàng đợi và ghi lỗi, không thêm tài liệu", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: QUEUED },
    { type: "UPLOAD_SETTLED", id: "upload-1", error: "OCR hỏng" },
  ]);

  assert.deepEqual(state.pendingUploads.map((item) => item.id), ["upload-2"]);
  assert.deepEqual(state.attachments, []);
  assert.equal(state.error, "OCR hỏng");
});

test("ATTACHMENT_REMOVED bỏ đúng tài liệu", () => {
  const state = reduce([
    { type: "UPLOAD_QUEUED", items: [] },
    { type: "UPLOAD_SETTLED", id: "x", attachment: { id: "att-1" } },
    { type: "UPLOAD_SETTLED", id: "y", attachment: { id: "att-2" } },
    { type: "ATTACHMENT_REMOVED", attachmentId: "att-1" },
  ]);

  assert.deepEqual(state.attachments.map((item) => item.id), ["att-2"]);
});

test("ATTACHMENTS_SYNCED chỉ chạm conversations và attachments, không đụng thread", () => {
  const withThread = reduce([{
    type: "MESSAGES_APPENDED",
    messages: [{ id: "m1", role: "assistant", content: "giữ nguyên", sources: [{ id: "s1" }] }],
  }]);

  const state = chatSessionReducer(withThread, {
    type: "ATTACHMENTS_SYNCED",
    conversations: [{ id: "c1" }],
    attachments: [{ id: "att-9" }],
  });

  assert.deepEqual(state.conversations, [{ id: "c1" }]);
  assert.deepEqual(state.attachments, [{ id: "att-9" }]);
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].content, "giữ nguyên");
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `node --test tests/chatSessionReducer.uploads.test.js`
Expected: FAIL nhiều test.

- [ ] **Step 3: Thêm các case upload vào reducer**

Đặt trước `default`:

```js
    case "UPLOAD_QUEUED":
      return { ...state, pendingUploads: [...state.pendingUploads, ...action.items], error: "" };

    case "UPLOAD_PROGRESS":
      return {
        ...state,
        pendingUploads: state.pendingUploads.map((item) =>
          item.id === action.id ? { ...item, status: action.status } : item),
      };

    case "UPLOAD_SETTLED": {
      const pendingUploads = state.pendingUploads.filter((item) => item.id !== action.id);
      if (!action.attachment) {
        return { ...state, pendingUploads, error: action.error ?? state.error };
      }
      return {
        ...state,
        pendingUploads,
        attachments: [
          ...state.attachments.filter((item) => item.id !== action.attachment.id),
          action.attachment,
        ],
      };
    }

    case "ATTACHMENT_REMOVED":
      return {
        ...state,
        attachments: state.attachments.filter((item) => item.id !== action.attachmentId),
      };

    case "ATTACHMENTS_SYNCED":
      return { ...state, conversations: action.conversations, attachments: action.attachments };
```

- [ ] **Step 4: Chạy test**

Run: `node --test tests/chatSessionReducer.uploads.test.js`
Expected: PASS toàn bộ.

- [ ] **Step 5: Chạy toàn bộ**

Run: `npm test`
Expected: fail 0.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/state/chatSessionReducer.js frontend/tests/chatSessionReducer.uploads.test.js
git commit -m "refactor: đưa logic hàng đợi upload vào reducer thuần"
```

---

### Task 8: Hook độc lập `useTheme` và `useChatMode`

**Files:**
- Create: `frontend/src/hooks/useTheme.js`
- Create: `frontend/src/hooks/useChatMode.js`
- Test: `frontend/tests/ui/useTheme.test.jsx`

**Interfaces:**
- Consumes: `persistChatMode`, `readStoredChatMode` từ `src/config/chatModes.js`.
- Produces:
  - `useTheme(): { theme: "dark" | "light", toggleTheme: () => void }`
  - `useChatMode(): { mode: string, setMode: (mode: string) => void }`
  - `THEME_STORAGE_KEY` export từ `useTheme.js`

- [ ] **Step 1: Viết test**

Create `frontend/tests/ui/useTheme.test.jsx`:

```jsx
import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test } from "vitest";

import { THEME_STORAGE_KEY, useTheme } from "../../src/hooks/useTheme.js";
import { useChatMode } from "../../src/hooks/useChatMode.js";

beforeEach(() => {
  window.localStorage.clear();
  window.matchMedia = (query) => ({
    matches: false,
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  });
});

test("useTheme đọc lựa chọn đã lưu", () => {
  window.localStorage.setItem(THEME_STORAGE_KEY, "dark");
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("dark");
});

test("useTheme lật giữa dark và light rồi ghi vào localStorage và thẻ html", () => {
  const { result } = renderHook(() => useTheme());
  const first = result.current.theme;

  act(() => result.current.toggleTheme());

  expect(result.current.theme).not.toBe(first);
  expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe(result.current.theme);
  expect(document.documentElement.dataset.theme).toBe(result.current.theme);
});

test("useChatMode mặc định hybrid và lưu lại lựa chọn mới", () => {
  const { result } = renderHook(() => useChatMode());
  expect(result.current.mode).toBe("hybrid");

  act(() => result.current.setMode("central"));

  expect(result.current.mode).toBe("central");
  expect(window.localStorage.getItem("vn-history-chat-mode-v2")).toBe("central");
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `npx vitest run tests/ui/useTheme.test.jsx`
Expected: FAIL, không resolve được `src/hooks/useTheme.js`.

- [ ] **Step 3: Tạo `src/hooks/useTheme.js`**

Chép logic từ `App.jsx:37`, `App.jsx:68-72`, `App.jsx:145-148`.

```js
import { useEffect, useState } from "react";

export const THEME_STORAGE_KEY = "vn-history-theme";

function getInitialTheme() {
  const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (["dark", "light"].includes(savedTheme)) return savedTheme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useTheme() {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = () => setTheme((current) => (current === "dark" ? "light" : "dark"));

  return { theme, toggleTheme };
}
```

- [ ] **Step 4: Tạo `src/hooks/useChatMode.js`**

```js
import { useEffect, useState } from "react";

import { persistChatMode, readStoredChatMode } from "../config/chatModes.js";

export function useChatMode() {
  const [mode, setMode] = useState(readStoredChatMode);

  useEffect(() => {
    persistChatMode(mode);
  }, [mode]);

  return { mode, setMode };
}
```

- [ ] **Step 5: Chạy test**

Run: `npx vitest run tests/ui/useTheme.test.jsx`
Expected: PASS 3 test.

- [ ] **Step 6: Chạy toàn bộ**

Run: `npm test`
Expected: fail 0.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/ frontend/tests/ui/useTheme.test.jsx
git commit -m "refactor: tách useTheme và useChatMode ra hook riêng"
```

---

### Task 9: `useChatSession` — reducer cùng các action bất đồng bộ

**Files:**
- Create: `frontend/src/hooks/useChatSession.js`
- Test: `frontend/tests/ui/useChatSession.test.jsx`

**Interfaces:**
- Consumes: `chatSessionReducer`, `initialChatSessionState` (Task 5-7); `normalizeConversationList`, `normalizeConversationDetail` (Task 3); hằng số thông báo (Task 2); `listConversations`, `getConversation`, `createConversation`, `updateConversation`, `deleteConversation` từ `src/services/api.js`.
- Produces:

```js
useChatSession(): {
  state,                                          // hình dạng của initialChatSessionState
  dispatch,                                       // (action) => void
  isRunning: boolean,                             // isRunningStatus(state.status)
  loadConversation(conversationId): Promise<void>,
  createNewConversation(): Promise<{ id: string }>,
  renameConversation(conversation, title): Promise<void>,
  removeConversation(conversationId): Promise<string[]>,  // trả về id còn lại
  ensureActiveConversation(): Promise<string>,
}
```

**Hook không nhận `isRunning` từ ngoài.** `isRunning` suy ra từ `state.status`, mà `state` nằm trong chính hook này — truyền từ ngoài vào sẽ tạo phụ thuộc vòng, buộc `App.jsx` phải giữ một bản sao `status` và đồng bộ bằng `useEffect`, tức là state đúp và lệch một nhịp render. Hook tự tính lấy và trả ra ngoài cho các hook khác dùng.

**Sửa bug kèm theo (spec mục 6):** `ensureActiveConversation` hiện đọc `conversation.id` từ giá trị có thể là `null`. Sau task này nó **ném lỗi có thông báo rõ ràng** thay vì `TypeError`.

- [ ] **Step 1: Viết test**

Create `frontend/tests/ui/useChatSession.test.jsx`:

```jsx
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  createConversation: vi.fn(),
  updateConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));

const api = await import("../../src/services/api.js");
const { useChatSession } = await import("../../src/hooks/useChatSession.js");

beforeEach(() => {
  api.listConversations.mockResolvedValue([]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
  api.createConversation.mockResolvedValue({ conversation: { id: "c-mới" } });
  api.updateConversation.mockResolvedValue({ title: "Tên mới" });
  api.deleteConversation.mockResolvedValue({});
});

test("bootstrap nạp danh sách và mở hội thoại đầu tiên", async () => {
  api.listConversations.mockResolvedValue([{ id: "c1" }, { id: "c2" }]);
  api.getConversation.mockResolvedValue({
    messages: [{ id: "m1", role: "assistant", sources: [] }],
    attachments: [],
  });

  const { result } = renderHook(() => useChatSession());

  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));
  expect(result.current.state.activeConversationId).toBe("c1");
  expect(result.current.state.messages).toHaveLength(1);
});

test("bootstrap hỏng vẫn tắt spinner và hiện lỗi", async () => {
  api.listConversations.mockRejectedValue(new Error("Mất mạng"));

  const { result } = renderHook(() => useChatSession());

  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));
  expect(result.current.state.error).toBe("Mất mạng");
});

test("ensureActiveConversation tái sử dụng hội thoại đang mở, không gọi tạo mới", async () => {
  api.listConversations.mockResolvedValue([{ id: "c1" }]);

  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.activeConversationId).toBe("c1"));

  const id = await result.current.ensureActiveConversation();

  expect(id).toBe("c1");
  expect(api.createConversation).not.toHaveBeenCalled();
});

test("ensureActiveConversation tạo hội thoại mới khi chưa có", async () => {
  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  const id = await result.current.ensureActiveConversation();

  expect(id).toBe("c-mới");
});

test("ensureActiveConversation ném lỗi rõ ràng khi đang stream, không ném TypeError", async () => {
  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  act(() => {
    result.current.dispatch({ type: "STREAM_STATUS", messageId: "m", status: "streaming" });
  });
  expect(result.current.isRunning).toBe(true);

  await expect(result.current.ensureActiveConversation()).rejects.toThrow(
    "Không thể tạo cuộc trò chuyện mới.",
  );
  expect(api.createConversation).not.toHaveBeenCalled();
});

test("ensureActiveConversation ném lỗi khi backend không trả về id", async () => {
  api.createConversation.mockResolvedValue({});

  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  await expect(result.current.ensureActiveConversation()).rejects.toThrow(
    "Backend không trả về conversation ID.",
  );
});

test("loadConversation hỏng vẫn tắt cờ đang tải và ghi lỗi", async () => {
  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.isLoadingConversations).toBe(false));

  api.getConversation.mockRejectedValue(new Error("Không tải nổi"));

  await act(async () => {
    await expect(result.current.loadConversation("c9")).rejects.toThrow("Không tải nổi");
  });

  expect(result.current.state.isLoadingConversation).toBe(false);
  expect(result.current.state.error).toBe("Không tải nổi");
});

test("removeConversation trả về danh sách id còn lại", async () => {
  api.listConversations.mockResolvedValue([{ id: "c1" }, { id: "c2" }]);

  const { result } = renderHook(() => useChatSession());
  await waitFor(() => expect(result.current.state.conversations).toHaveLength(2));

  let remaining;
  await act(async () => {
    remaining = await result.current.removeConversation("c1");
  });

  expect(remaining).toEqual(["c2"]);
  expect(api.deleteConversation).toHaveBeenCalledWith("c1");
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `npx vitest run tests/ui/useChatSession.test.jsx`
Expected: FAIL, không resolve được `src/hooks/useChatSession.js`.

- [ ] **Step 3: Tạo `src/hooks/useChatSession.js`**

```js
import { useEffect, useReducer } from "react";

import {
  BACKEND_UNREACHABLE_MESSAGE,
  CONVERSATION_CREATE_NEW_FAILURE_MESSAGE,
  CONVERSATION_LOAD_FAILURE_MESSAGE,
  MISSING_CONVERSATION_ID_MESSAGE,
} from "../config/messages.js";
import {
  chatSessionReducer,
  initialChatSessionState,
  isRunningStatus,
} from "../state/chatSessionReducer.js";
import { normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  updateConversation,
} from "../services/api.js";

export function useChatSession() {
  const [state, dispatch] = useReducer(chatSessionReducer, initialChatSessionState);
  const isRunning = isRunningStatus(state.status);

  useEffect(() => {
    const controller = new AbortController();

    async function bootstrap() {
      try {
        const payload = await listConversations({ signal: controller.signal });
        const items = normalizeConversationList(payload);

        if (items.length === 0) {
          dispatch({ type: "BOOTSTRAPPED", conversations: items });
          return;
        }

        const detailPayload = await getConversation(items[0].id, { signal: controller.signal });
        dispatch({
          type: "BOOTSTRAPPED",
          conversations: items,
          activeConversationId: items[0].id,
          detail: normalizeConversationDetail(detailPayload),
        });
      } catch (requestError) {
        if (requestError.name === "AbortError") return;
        console.error(requestError);
        dispatch({
          type: "BOOTSTRAP_FAILED",
          message: requestError.message || BACKEND_UNREACHABLE_MESSAGE,
        });
      }
    }

    bootstrap();
    return () => controller.abort();
  }, []);

  // Hook tự chịu trách nhiệm tắt cờ đang tải trong CẢ hai nhánh. Nếu để việc đó
  // cho nơi gọi thì chỉ cần một nơi quên là spinner kẹt vĩnh viễn — mà hàm này
  // được gọi từ hai chỗ khác nhau.
  const loadConversation = async (conversationId) => {
    dispatch({ type: "CONVERSATION_LOADING" });
    try {
      const payload = await getConversation(conversationId);
      dispatch({
        type: "CONVERSATION_LOADED",
        conversationId,
        detail: normalizeConversationDetail(payload),
      });
    } catch (requestError) {
      dispatch({
        type: "CONVERSATION_LOAD_FAILED",
        message: requestError.message || CONVERSATION_LOAD_FAILURE_MESSAGE,
      });
      throw requestError;
    }
  };

  const createNewConversation = async () => {
    if (isRunning) throw new Error(CONVERSATION_CREATE_NEW_FAILURE_MESSAGE);

    const payload = await createConversation({ title: null });
    const conversation = payload?.conversation ?? payload;
    if (!conversation?.id) throw new Error(MISSING_CONVERSATION_ID_MESSAGE);

    dispatch({ type: "CONVERSATION_CREATED", conversation });
    return conversation;
  };

  const renameConversation = async (conversation, title) => {
    const updated = await updateConversation(conversation.id, { title });
    dispatch({ type: "CONVERSATION_RENAMED", conversationId: conversation.id, patch: updated });
  };

  const removeConversation = async (conversationId) => {
    await deleteConversation(conversationId);
    dispatch({ type: "CONVERSATION_DELETED", conversationId });
    return state.conversations
      .filter((item) => item.id !== conversationId)
      .map((item) => item.id);
  };

  const ensureActiveConversation = async () => {
    if (state.activeConversationId) return state.activeConversationId;
    const conversation = await createNewConversation();
    return conversation.id;
  };

  return {
    state,
    dispatch,
    isRunning,
    loadConversation,
    createNewConversation,
    renameConversation,
    removeConversation,
    ensureActiveConversation,
  };
}
```

- [ ] **Step 4: Chạy test**

Run: `npx vitest run tests/ui/useChatSession.test.jsx`
Expected: PASS 8 test.

- [ ] **Step 5: Chạy toàn bộ**

Run: `npm test`
Expected: fail 0.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useChatSession.js frontend/tests/ui/useChatSession.test.jsx
git commit -m "refactor: thêm useChatSession và sửa lỗi ensureActiveConversation trả null"
```

---

### Task 10: `useChatStream` và thay test so khớp văn bản nguồn

**Files:**
- Create: `frontend/src/hooks/useChatStream.js`
- Modify: `frontend/tests/chatModes.test.js:72-79`
- Test: `frontend/tests/ui/useChatStream.test.jsx`

**Interfaces:**
- Consumes: `dispatch`, `isRunning` và `ensureActiveConversation` từ `useChatSession` (Task 9); `streamChat`, `listConversations`, `getConversation` từ `src/services/api.js`; `createLocalId` (Task 3); `getSources`, `normalizeConversationList`, `normalizeConversationDetail` (Task 3).
- Produces:

```js
useChatStream({ dispatch, isRunning, mode, showDebugTrace, ensureActiveConversation }): {
  submit(question: string): Promise<void>,
  stop(): void,
}
```

**Ghi chú lệch so với spec:** spec mục 3.4 ghi hook này trả về cả `isRunning`. Bỏ đi vì `useChatSession` đã trả `isRunning` rồi — trả lần nữa ở đây là hai nguồn sự thật cho cùng một giá trị.

- [ ] **Step 1: Xoá test so khớp văn bản nguồn khỏi `tests/chatModes.test.js`**

Xoá nguyên test case này (dòng 72-79) — nó đọc `App.jsx` như văn bản nên sẽ đỏ sau refactor dù hành vi không đổi:

```js
test("frontend sends the selected canonical mode to the streaming API", async () => {
  const appSource = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const apiSource = await readFile(new URL("../src/services/api.js", import.meta.url), "utf8");

  assert.match(appSource, /mode:\s*inferenceMode/);
  assert.match(apiSource, /JSON\.stringify\([\s\S]*?mode,/);
});
```

Giữ nguyên test `"compact mode dropdown is inside the composer's left action group"` — nó đọc `ChatInput.jsx`, file mà refactor này không đụng tới.

- [ ] **Step 2: Viết test thay thế, kiểm tra hành vi thay vì văn bản**

Create `frontend/tests/ui/useChatStream.test.jsx`:

```jsx
import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  streamChat: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  EVIDENCE_CONTRACT_FAILURE_MESSAGE: "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
}));

const api = await import("../../src/services/api.js");
const { useChatStream } = await import("../../src/hooks/useChatStream.js");

function setup({ isRunning = false, mode = "central" } = {}) {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockResolvedValue("c1");
  const { result } = renderHook(() => useChatStream({
    dispatch,
    isRunning,
    mode,
    showDebugTrace: false,
    ensureActiveConversation,
  }));
  return { dispatch, ensureActiveConversation, result };
}

beforeEach(() => {
  api.streamChat.mockResolvedValue(undefined);
  // Hook gọi hai hàm này để đồng bộ lại sau khi stream xong. Thiếu chúng thì
  // mọi test "thành công" sẽ ngã vào nhánh catch và xanh vì lý do sai.
  api.listConversations.mockResolvedValue([{ id: "c1" }]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
});

test("gửi đúng chế độ đang chọn tới API streaming", async () => {
  const { result } = setup({ mode: "three_llm" });

  await act(async () => {
    await result.current.submit("Chiến thắng Bạch Đằng?");
  });

  expect(api.streamChat).toHaveBeenCalledTimes(1);
  expect(api.streamChat.mock.calls[0][0]).toMatchObject({
    conversationId: "c1",
    question: "Chiến thắng Bạch Đằng?",
    mode: "three_llm",
    finalK: 6,
  });
});

test("bỏ qua câu hỏi rỗng và khi đang chạy", async () => {
  const { result } = setup();
  await act(async () => {
    await result.current.submit("    ");
  });
  expect(api.streamChat).not.toHaveBeenCalled();

  const running = setup({ isRunning: true });
  await act(async () => {
    await running.result.current.submit("Có nội dung");
  });
  expect(api.streamChat).not.toHaveBeenCalled();
});

test("chuyển các sự kiện SSE thành action tương ứng", async () => {
  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({ event: "status", data: { stage: "hybrid_retrieval", mode: "hybrid" } });
    onEvent({ event: "answer_delta", data: { delta: "Xin " } });
    onEvent({ event: "answer_delta", data: "chào" });
    onEvent({ event: "sources", data: { items: [{ id: "s1" }] } });
    onEvent({ event: "done", data: {} });
  });

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("MESSAGES_APPENDED");
  expect(types).toContain("STREAM_STATUS");
  expect(types).toContain("STREAM_DELTA");
  expect(types).toContain("STREAM_SOURCES");
  expect(types).toContain("STREAM_DONE");

  const deltas = dispatch.mock.calls
    .map(([action]) => action)
    .filter((action) => action.type === "STREAM_DELTA")
    .map((action) => action.delta);
  expect(deltas).toEqual(["Xin ", "chào"]);
});

test("sự kiện error mang theo loại lỗi để reducer chọn đúng thông báo", async () => {
  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({
      event: "error",
      data: { message: "Evidence critic từ chối", type: "evidence_contract_error" },
    });
  });

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const errorAction = dispatch.mock.calls
    .map(([action]) => action)
    .find((action) => action.type === "STREAM_ERROR");

  expect(errorAction.kind).toBe("evidence_contract_error");
  expect(errorAction.message).toBe("Evidence critic từ chối");
});

test("AbortError sinh ra STREAM_ABORTED chứ không phải STREAM_ERROR", async () => {
  const abortError = new Error("Aborted");
  abortError.name = "AbortError";
  api.streamChat.mockRejectedValue(abortError);

  const { dispatch, result } = setup();
  await act(async () => {
    await result.current.submit("Hỏi");
  });

  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("STREAM_ABORTED");
  expect(types).not.toContain("STREAM_ERROR");
});

test("stop() huỷ request đang chạy", async () => {
  let capturedSignal;
  api.streamChat.mockImplementation(async ({ signal }) => {
    capturedSignal = signal;
    await new Promise((resolve) => setTimeout(resolve, 50));
  });

  const { result } = setup();
  let pending;
  await act(async () => {
    pending = result.current.submit("Hỏi");
    await Promise.resolve();
  });

  act(() => result.current.stop());
  await act(async () => { await pending; });

  expect(capturedSignal.aborted).toBe(true);
});

test("không tạo được hội thoại thì báo lỗi và không gọi streaming", async () => {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockRejectedValue(new Error("Không thể tạo cuộc trò chuyện."));
  const { result } = renderHook(() => useChatStream({
    dispatch, isRunning: false, mode: "hybrid", showDebugTrace: false, ensureActiveConversation,
  }));

  await act(async () => {
    await result.current.submit("Hỏi");
  });

  expect(api.streamChat).not.toHaveBeenCalled();
  expect(dispatch).toHaveBeenCalledWith(
    expect.objectContaining({ type: "ERROR_SET", message: "Không thể tạo cuộc trò chuyện." }),
  );
});
```

- [ ] **Step 3: Chạy test để xác nhận đỏ**

Run: `npx vitest run tests/ui/useChatStream.test.jsx`
Expected: FAIL, không resolve được `src/hooks/useChatStream.js`.

- [ ] **Step 4: Tạo `src/hooks/useChatStream.js`**

Việc đồng bộ lại sau khi stream xong (`listConversations` + `getConversation`) giữ nguyên như `App.jsx:411-421`.

```js
import { useRef } from "react";

import {
  BACKEND_UNREACHABLE_MESSAGE,
  CONVERSATION_CREATE_FAILURE_MESSAGE,
  STREAM_FAILURE_MESSAGE,
} from "../config/messages.js";
import { createLocalId } from "../state/ids.js";
import { getSources, normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import { getConversation, listConversations, streamChat } from "../services/api.js";

export function useChatStream({ dispatch, isRunning, mode, showDebugTrace, ensureActiveConversation }) {
  const abortControllerRef = useRef(null);

  const stop = () => abortControllerRef.current?.abort();

  const submit = async (question) => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isRunning) return;

    let conversationId;
    try {
      conversationId = await ensureActiveConversation();
    } catch (requestError) {
      dispatch({
        type: "ERROR_SET",
        message: requestError.message || CONVERSATION_CREATE_FAILURE_MESSAGE,
      });
      return;
    }

    const assistantMessageId = createLocalId("assistant");
    const createdAt = new Date().toISOString();

    dispatch({
      type: "MESSAGES_APPENDED",
      messages: [
        {
          id: createLocalId("user"),
          role: "user",
          content: trimmedQuestion,
          sources: [],
          status: "done",
          created_at: createdAt,
        },
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          sources: [],
          status: "processing",
          created_at: createdAt,
        },
      ],
    });

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let streamFailed = false;

    try {
      await streamChat({
        conversationId,
        question: trimmedQuestion,
        mode,
        finalK: 6,
        debug: showDebugTrace,
        signal: controller.signal,
        onEvent: ({ event: eventName, data }) => {
          if (eventName === "status") {
            dispatch({
              type: "STREAM_STATUS",
              messageId: assistantMessageId,
              status: typeof data === "string" ? data : data?.stage ?? "processing",
              mode: data?.mode ?? mode,
            });
            return;
          }

          if (eventName === "answer_delta") {
            dispatch({
              type: "STREAM_DELTA",
              messageId: assistantMessageId,
              delta: typeof data === "string" ? data : data?.delta ?? "",
            });
            return;
          }

          if (eventName === "sources") {
            dispatch({
              type: "STREAM_SOURCES",
              messageId: assistantMessageId,
              sources: getSources(data),
            });
            return;
          }

          if (eventName === "debug_trace" || eventName === "debug") {
            dispatch({ type: "STREAM_DEBUG", messageId: assistantMessageId, trace: data });
            return;
          }

          if (eventName === "error") {
            streamFailed = true;
            dispatch({
              type: "STREAM_ERROR",
              messageId: assistantMessageId,
              message: typeof data === "string" ? data : data?.message ?? STREAM_FAILURE_MESSAGE,
              kind: data?.type,
              trace: data?.debug_trace,
            });
            return;
          }

          if (eventName === "done") {
            dispatch({ type: "STREAM_DONE", messageId: assistantMessageId });
          }
        },
      });

      if (streamFailed) return;

      const [conversationPayload, detailPayload] = await Promise.all([
        listConversations(),
        getConversation(conversationId),
      ]);

      dispatch({
        type: "STREAM_SYNCED",
        conversations: normalizeConversationList(conversationPayload),
        detail: normalizeConversationDetail(detailPayload),
      });
    } catch (requestError) {
      if (requestError.name === "AbortError") {
        dispatch({ type: "STREAM_ABORTED", messageId: assistantMessageId });
      } else {
        console.error(requestError);
        dispatch({
          type: "STREAM_ERROR",
          messageId: assistantMessageId,
          message: requestError.message || BACKEND_UNREACHABLE_MESSAGE,
        });
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  return { submit, stop };
}
```

- [ ] **Step 5: Chạy test**

Run: `npx vitest run tests/ui/useChatStream.test.jsx`
Expected: PASS 7 test.

- [ ] **Step 6: Chạy toàn bộ**

Run: `npm test`
Expected: `node --test` giảm còn 8 test (bỏ 1 test so khớp văn bản nguồn), fail 0. Vitest fail 0.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/hooks/useChatStream.js frontend/tests/ui/useChatStream.test.jsx frontend/tests/chatModes.test.js
git commit -m "refactor: tách useChatStream và thay test so khớp nguồn bằng test hành vi"
```

---

### Task 11: `useAttachments`

**Files:**
- Create: `frontend/src/hooks/useAttachments.js`
- Test: `frontend/tests/ui/useAttachments.test.jsx`

**Interfaces:**
- Consumes: `validateUploadSelection`, `createUploadItems` (Task 4); `uploadAttachment`, `deleteAttachment`, `listConversations`, `getConversation` từ `src/services/api.js`.
- Produces:

```js
useAttachments({ dispatch, activeConversationId, isRunning, ensureActiveConversation }): {
  upload(selectedFiles: File[]): Promise<void>,
  remove(attachmentId: string): Promise<void>,
}
```

- [ ] **Step 1: Viết test**

Create `frontend/tests/ui/useAttachments.test.jsx`:

```jsx
import { act, renderHook } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  uploadAttachment: vi.fn(),
  deleteAttachment: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
}));

const api = await import("../../src/services/api.js");
const { useAttachments } = await import("../../src/hooks/useAttachments.js");

function makeFile(name, type, size = 1024) {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

function setup({ isRunning = false, activeConversationId = "c1" } = {}) {
  const dispatch = vi.fn();
  const ensureActiveConversation = vi.fn().mockResolvedValue("c1");
  const { result } = renderHook(() => useAttachments({
    dispatch, activeConversationId, isRunning, ensureActiveConversation,
  }));
  return { dispatch, ensureActiveConversation, result };
}

beforeEach(() => {
  api.uploadAttachment.mockResolvedValue({ attachment: { id: "att-1", name: "a.pdf" } });
  api.deleteAttachment.mockResolvedValue({});
  api.listConversations.mockResolvedValue([]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
});

test("từ chối file sai định dạng TRƯỚC khi gọi mạng", async () => {
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("virus.exe", "application/x-msdownload")]);
  });

  expect(api.uploadAttachment).not.toHaveBeenCalled();
  expect(dispatch).toHaveBeenCalledWith({
    type: "ERROR_SET",
    message: "Không hỗ trợ định dạng của virus.exe. Chỉ nhận PDF, PNG, JPEG và WebP.",
  });
});

test("từ chối file quá 20 MB trước khi gọi mạng", async () => {
  const { result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("to.pdf", "application/pdf", 21 * 1024 * 1024)]);
  });

  expect(api.uploadAttachment).not.toHaveBeenCalled();
});

test("không upload khi đang stream", async () => {
  const { result } = setup({ isRunning: true });

  await act(async () => {
    await result.current.upload([makeFile("ok.pdf", "application/pdf")]);
  });

  expect(api.uploadAttachment).not.toHaveBeenCalled();
});

test("upload hợp lệ đi qua đủ chuỗi action queued, processing, settled", async () => {
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("ok.pdf", "application/pdf")]);
  });

  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("UPLOAD_QUEUED");
  expect(types).toContain("UPLOAD_PROGRESS");
  expect(types).toContain("UPLOAD_SETTLED");
  expect(types).toContain("ATTACHMENTS_SYNCED");
  expect(api.uploadAttachment).toHaveBeenCalledTimes(1);
});

test("upload hỏng vẫn phải rời hàng đợi kèm lỗi", async () => {
  api.uploadAttachment.mockRejectedValue(new Error("OCR hỏng"));
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.upload([makeFile("ok.pdf", "application/pdf")]);
  });

  const settled = dispatch.mock.calls
    .map(([action]) => action)
    .find((action) => action.type === "UPLOAD_SETTLED");

  expect(settled.error).toBe("OCR hỏng");
  expect(settled.attachment).toBeUndefined();
});

test("remove xoá tài liệu rồi làm mới danh sách hội thoại", async () => {
  const { dispatch, result } = setup();

  await act(async () => {
    await result.current.remove("att-1");
  });

  expect(api.deleteAttachment).toHaveBeenCalledWith("c1", "att-1");
  const types = dispatch.mock.calls.map(([action]) => action.type);
  expect(types).toContain("ATTACHMENT_REMOVED");
  expect(types).toContain("CONVERSATIONS_SET");
});

test("remove không làm gì khi đang stream", async () => {
  const { result } = setup({ isRunning: true });

  await act(async () => {
    await result.current.remove("att-1");
  });

  expect(api.deleteAttachment).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Chạy test để xác nhận đỏ**

Run: `npx vitest run tests/ui/useAttachments.test.jsx`
Expected: FAIL, không resolve được `src/hooks/useAttachments.js`.

- [ ] **Step 3: Tạo `src/hooks/useAttachments.js`**

```js
import {
  ATTACHMENT_DELETE_FAILURE_MESSAGE,
  CONVERSATION_CREATE_FAILURE_MESSAGE,
} from "../config/messages.js";
import { normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import { createUploadItems, validateUploadSelection } from "../state/uploadQueue.js";
import {
  deleteAttachment,
  getConversation,
  listConversations,
  uploadAttachment,
} from "../services/api.js";

export function useAttachments({ dispatch, activeConversationId, isRunning, ensureActiveConversation }) {
  const upload = async (selectedFiles) => {
    if (isRunning) return;

    const { error, files } = validateUploadSelection(selectedFiles);
    if (error) {
      dispatch({ type: "ERROR_SET", message: error });
      return;
    }

    let conversationId;
    try {
      conversationId = await ensureActiveConversation();
    } catch (requestError) {
      dispatch({
        type: "ERROR_SET",
        message: requestError.message || CONVERSATION_CREATE_FAILURE_MESSAGE,
      });
      return;
    }

    const items = createUploadItems(files);
    dispatch({ type: "UPLOAD_QUEUED", items });

    for (const item of items) {
      dispatch({ type: "UPLOAD_PROGRESS", id: item.id, status: "processing" });

      try {
        const payload = await uploadAttachment(conversationId, item.file);
        dispatch({
          type: "UPLOAD_SETTLED",
          id: item.id,
          attachment: payload?.attachment ?? payload,
        });
      } catch (requestError) {
        console.error(requestError);
        dispatch({
          type: "UPLOAD_SETTLED",
          id: item.id,
          error: requestError.message || `Không thể xử lý ${item.name}.`,
        });
      }
    }

    try {
      const [conversationPayload, detailPayload] = await Promise.all([
        listConversations(),
        getConversation(conversationId),
      ]);

      dispatch({
        type: "ATTACHMENTS_SYNCED",
        conversations: normalizeConversationList(conversationPayload),
        attachments: normalizeConversationDetail(detailPayload).attachments,
      });
    } catch (refreshError) {
      console.warn("Could not refresh attachments", refreshError);
    }
  };

  const remove = async (attachmentId) => {
    if (!activeConversationId || isRunning) return;

    try {
      await deleteAttachment(activeConversationId, attachmentId);
      dispatch({ type: "ATTACHMENT_REMOVED", attachmentId });

      const payload = await listConversations();
      dispatch({ type: "CONVERSATIONS_SET", conversations: normalizeConversationList(payload) });
    } catch (requestError) {
      console.error(requestError);
      dispatch({
        type: "ERROR_SET",
        message: requestError.message || ATTACHMENT_DELETE_FAILURE_MESSAGE,
      });
    }
  };

  return { upload, remove };
}
```

- [ ] **Step 4: Chạy test**

Run: `npx vitest run tests/ui/useAttachments.test.jsx`
Expected: PASS 7 test.

- [ ] **Step 5: Chạy toàn bộ**

Run: `npm test`
Expected: fail 0.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useAttachments.js frontend/tests/ui/useAttachments.test.jsx
git commit -m "refactor: tách useAttachments cùng chặn upload khi đang stream"
```

---

### Task 12: Dọn `App.jsx` và nối dây

**Files:**
- Modify: `frontend/src/App.jsx` (thay gần như toàn bộ phần logic, giữ nguyên phần JSX)
- Test: `frontend/tests/ui/App.test.jsx`

**Interfaces:**
- Consumes: toàn bộ hook từ Task 8-11.
- Produces: `App.jsx` chỉ còn state UI cục bộ và phần render. Không task nào phụ thuộc vào task này.

**Quan trọng:** phần JSX từ `App.jsx:549` (`return (`) tới hết **không được sửa cấu trúc**, chỉ đổi tên biến/handler cho khớp. Đây là chốt chặn để giao diện không đổi.

- [ ] **Step 1: Viết test smoke cho App**

Create `frontend/tests/ui/App.test.jsx`:

```jsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("../../src/services/api.js", () => ({
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  createConversation: vi.fn(),
  updateConversation: vi.fn(),
  deleteConversation: vi.fn(),
  uploadAttachment: vi.fn(),
  deleteAttachment: vi.fn(),
  streamChat: vi.fn(),
  EVIDENCE_CONTRACT_FAILURE_MESSAGE: "Không thể hoàn tất câu trả lời do bước đánh giá bằng chứng thất bại.",
}));

const api = await import("../../src/services/api.js");
const { default: App } = await import("../../src/App.jsx");

beforeEach(() => {
  window.localStorage.clear();
  window.matchMedia = (query) => ({
    matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {},
  });
  Element.prototype.scrollIntoView = vi.fn();

  api.listConversations.mockResolvedValue([{ id: "c1", title: "Nhà Trần" }]);
  api.getConversation.mockResolvedValue({ messages: [], attachments: [] });
  api.streamChat.mockResolvedValue(undefined);
});

test("hiển thị hội thoại đã có sau khi bootstrap", async () => {
  render(<App />);
  expect(await screen.findByText("Nhà Trần")).toBeInTheDocument();
});

test("gửi câu hỏi thì text stream về hiện trên màn hình", async () => {
  api.streamChat.mockImplementation(async ({ onEvent }) => {
    onEvent({ event: "answer_delta", data: { delta: "Nhà Trần suy yếu vì " } });
    onEvent({ event: "answer_delta", data: { delta: "nhiều nguyên nhân." } });
    onEvent({ event: "done", data: {} });
  });

  render(<App />);
  await screen.findByText("Nhà Trần");

  const textarea = screen.getByRole("textbox");
  await userEvent.type(textarea, "Vì sao nhà Trần suy yếu?");
  await userEvent.keyboard("{Enter}");

  await waitFor(() => {
    expect(screen.getByText(/Nhà Trần suy yếu vì nhiều nguyên nhân\./)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Cài `@testing-library/user-event`**

```bash
npm install -D @testing-library/user-event
```

- [ ] **Step 3: Chạy test trên `App.jsx` CŨ, phải XANH**

Run: `npx vitest run tests/ui/App.test.jsx`
Expected: **PASS 2 test** — chạy trên bản `App.jsx` chưa refactor.

Đây là điểm khác mọi task trước, và là chỗ dễ làm sai nếu áp dụng TDD máy móc. Hai test này không đặc tả tính năng mới, chúng **đặc tả hành vi hiện có** để làm lưới an toàn cho việc thay ruột ở Step 4 (spec mục 5). Test đỏ ở đây nghĩa là test viết sai, không phải code thiếu — sửa test cho tới khi nó xanh trên code cũ, rồi mới được refactor.

Sau Step 4 hai test này phải vẫn xanh mà **không sửa một chữ nào trong kỳ vọng**. Phải sửa kỳ vọng tức là hành vi đã đổi: dừng lại, tìm hiểu tại sao.

- [ ] **Step 4: Thay phần logic của `App.jsx`**

Xoá toàn bộ từ đầu file tới hết `const toggleTheme = ...` (`App.jsx:1-548`) và thay bằng khối dưới đây. **Giữ nguyên phần JSX** từ `return (` trở đi.

```jsx
import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle, BookOpenText, Landmark, Moon, PanelRightClose, PanelRightOpen,
  Paperclip, ScrollText, Sun, Swords, Trash2, X,
} from "lucide-react";

import AttachmentTray from "./components/AttachmentTray";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import ChatSidebar, { SidebarOpenButton } from "./components/ChatSidebar";
import LogoMark from "./components/LogoMark";
import RetrievedChunks from "./components/RetrievedChunks";
import StatusIndicator from "./components/StatusIndicator";
import { useAttachments } from "./hooks/useAttachments";
import { useChatMode } from "./hooks/useChatMode";
import { useChatSession } from "./hooks/useChatSession";
import { useChatStream } from "./hooks/useChatStream";
import { useTheme } from "./hooks/useTheme";
import {
  CONVERSATION_CREATE_NEW_FAILURE_MESSAGE,
  CONVERSATION_DELETE_FAILURE_MESSAGE,
  CONVERSATION_RENAME_FAILURE_MESSAGE,
} from "./config/messages";
import { shouldShowDebugTrace } from "./services/debugTrace";
import "./App.css";

const SHOW_DEBUG_TRACE = shouldShowDebugTrace(import.meta.env);
const SUGGESTIONS = [
  { icon: Landmark, label: "Một bước ngoặt lịch sử", question: "Chiến thắng Bạch Đằng năm 938 có ý nghĩa như thế nào?" },
  { icon: ScrollText, label: "Một triều đại", question: "Nguyên nhân nào dẫn đến sự suy yếu của nhà Trần?" },
  { icon: Swords, label: "So sánh sự kiện", question: "So sánh Cách mạng Tháng Tám và chiến thắng Điện Biên Phủ." },
];

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 840);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [conversationToDelete, setConversationToDelete] = useState(null);
  const [isDeletingConversation, setIsDeletingConversation] = useState(false);
  const bottomRef = useRef(null);

  const { theme, toggleTheme } = useTheme();
  const { mode: inferenceMode, setMode: setInferenceMode } = useChatMode();

  const session = useChatSession();
  const { state, dispatch, isRunning, ensureActiveConversation } = session;
  const {
    conversations, activeConversationId, messages, attachments, pendingUploads,
    sources, status, error, isLoadingConversations, isLoadingConversation,
  } = state;

  const stream = useChatStream({
    dispatch,
    isRunning,
    mode: inferenceMode,
    showDebugTrace: SHOW_DEBUG_TRACE,
    ensureActiveConversation,
  });

  const uploads = useAttachments({
    dispatch,
    activeConversationId,
    isRunning,
    ensureActiveConversation,
  });

  const isUploading = pendingUploads.length > 0;

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeConversationId),
    [activeConversationId, conversations],
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: status === "streaming" ? "auto" : "smooth" });
  }, [messages, status]);

  const closeSidebarOnMobile = () => {
    if (window.matchMedia("(max-width: 839px)").matches) setSidebarOpen(false);
  };

  const setError = (message) => dispatch({ type: "ERROR_SET", message });

  const handleSelectConversation = async (conversationId) => {
    if (isRunning || conversationId === activeConversationId) return;
    try {
      await session.loadConversation(conversationId);
      closeSidebarOnMobile();
    } catch (requestError) {
      // loadConversation đã dispatch CONVERSATION_LOAD_FAILED, ở đây chỉ ghi log.
      console.error(requestError);
    }
  };

  const handleNewConversation = async () => {
    try {
      await session.createNewConversation();
      setQuestion("");
      closeSidebarOnMobile();
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_CREATE_NEW_FAILURE_MESSAGE);
    }
  };

  const handleRenameConversation = async (conversation, title) => {
    try {
      await session.renameConversation(conversation, title);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_RENAME_FAILURE_MESSAGE);
    }
  };

  const confirmDeleteConversation = async () => {
    if (!conversationToDelete || isRunning) return;

    setIsDeletingConversation(true);
    setError("");

    try {
      const wasActive = activeConversationId === conversationToDelete.id;
      const remaining = await session.removeConversation(conversationToDelete.id);
      if (wasActive && remaining.length > 0) await session.loadConversation(remaining[0]);
      setConversationToDelete(null);
    } catch (requestError) {
      console.error(requestError);
      setError(requestError.message || CONVERSATION_DELETE_FAILURE_MESSAGE);
    } finally {
      setIsDeletingConversation(false);
    }
  };

  const handleSubmit = async (event) => {
    event?.preventDefault();
    // Chặn TRƯỚC khi xoá ô nhập. Xoá trước rồi mới để hook từ chối sẽ làm mất
    // câu người dùng vừa gõ khi họ bấm gửi lúc đang stream.
    if (!question.trim() || isRunning) return;

    const pending = question;
    setQuestion("");
    await stream.submit(pending);
  };

  const showMessageSources = (message) => {
    if (!message.sources?.length) return;
    dispatch({ type: "SOURCES_SHOWN", sources: message.sources });
    setSourcesOpen(true);
  };
```

Phần JSX giữ nguyên, chỉ đổi ba chỗ gọi handler:

```jsx
              onStop={stream.stop}
              onFilesSelected={uploads.upload}
```

```jsx
              onDelete={uploads.remove}
```

- [ ] **Step 5: Chạy test App**

Run: `npx vitest run tests/ui/App.test.jsx`
Expected: PASS 2 test.

- [ ] **Step 6: Chạy toàn bộ test**

Run: `npm test`
Expected: fail 0 ở cả hai runner.

- [ ] **Step 7: Kiểm tra lint và build**

Run: `npm run lint && npm run build`
Expected: cả hai sạch, không warning về import thừa. Nếu ESLint báo biến không dùng (ví dụ `AlertCircle` nếu lỡ xoá nhầm phần JSX), sửa ngay.

- [ ] **Step 8: Đếm lại số dòng**

Run: `npx wc -l src/App.jsx`
Expected: khoảng 300 dòng hoặc ít hơn, giảm từ 705.

- [ ] **Step 9: Kiểm tra thủ công trên trình duyệt**

Run: `npm run dev` (cần backend chạy sẵn, hoặc dùng `VITE_API_BASE_URL` trỏ tới Modal đang chạy)

Đối chiếu bằng mắt, tất cả phải y như trước refactor:
- Danh sách hội thoại hiện ra, đổi tên và xoá được
- Gửi câu hỏi, chữ chảy dần ra màn hình
- Bấm nút dừng giữa chừng thì text đã stream vẫn còn
- Upload một PDF, tray hiện trạng thái rồi thành tài liệu
- Drawer nguồn mở ra có nội dung
- Đổi dark/light, tải lại trang vẫn giữ lựa chọn

- [ ] **Step 10: Commit**

```bash
git add frontend/src/App.jsx frontend/tests/ui/App.test.jsx frontend/package.json frontend/package-lock.json
git commit -m "refactor: App.jsx chỉ còn render và nối dây các hook"
```

---

## Kiểm chứng cuối cùng

- [ ] `npm test` — fail 0 ở cả hai runner
- [ ] `npm run lint` — sạch
- [ ] `npm run build` — sạch
- [ ] `src/App.jsx` còn khoảng 300 dòng hoặc ít hơn
- [ ] Không file nào trong `src/state/` hay `src/config/` import `src/services/api.js`. Kiểm tra: `grep -rn "services/api" src/state src/config` phải không ra kết quả nào
- [ ] Không có test mới nào dùng `readFile` để so khớp văn bản nguồn. Kiểm tra: `grep -rn "readFile" tests/` chỉ ra đúng một kết quả còn lại trong `chatModes.test.js` (test về `ChatInput.jsx`)
- [ ] Đã kiểm tra thủ công trên trình duyệt theo checklist ở Task 12 Step 9
