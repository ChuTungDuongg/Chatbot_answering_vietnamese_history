# Frontend

React 19 and Vite provide the Vietnamese history chat UI. The composer offers two inference modes:

| Mode | Server path |
|---|---|
| `hybrid` | Existing history retrieval followed by vanilla `Qwen/Qwen3-4B-Instruct-2507` generation. |
| `central` | Vanilla `Qwen/Qwen3-8B` with tools, including the same history retrieval subsystem. |

The UI also supports conversation history, PDF/image attachments, cited sources, Markdown, light/dark themes, and responsive layout.

## Run locally

From the repository root:

```powershell
Copy-Item frontend/.env.example frontend/.env
npm install
npm --prefix frontend install
npm run dev
```

Set `VITE_API_BASE_URL` in `frontend/.env` to the backend origin before starting Vite. `npm run dev` starts the frontend and the Modal backend. If the backend is already running, use `npm run frontend`.

Vite embeds the API origin at build time. Restart Vite after changing it.

## API and streaming

The frontend calls `POST /api/v1/chat/stream` with `conversation_id`, `question`, and one of the two mode values. It sends an `X-Client-ID` header to associate conversations with an anonymous browser. This header is not authentication.

The SSE client handles `status`, `answer_delta`, `sources`, `done`, and `error` events. Each `answer_delta` is appended to the visible answer as it arrives; `sources` updates the citation drawer. After successful completion, the UI loads the stored conversation to use server message IDs and the final persisted answer. Stop aborts the request and retains any text already received.

The backend is responsible for generating actual model text incrementally. The frontend does not add a delay or divide a completed answer into simulated chunks.

## Browser state

Only the anonymous client ID, theme, and selected mode are stored in `localStorage`. The selected mode uses `vn-history-chat-mode-v2`; an unsupported saved value falls back to `hybrid`. Messages and attachments are held by the backend.

## Key files

| File | Purpose |
|---|---|
| [`src/config/chatModes.js`](src/config/chatModes.js) | Two-mode contract and saved selection. |
| [`src/services/api.js`](src/services/api.js) | REST requests and SSE parsing. |
| [`src/hooks/useChatStream.js`](src/hooks/useChatStream.js) | Stream lifecycle and event dispatch. |
| [`src/state/chatSessionReducer.js`](src/state/chatSessionReducer.js) | Incremental answer and source state. |
| [`src/components/ChatMessage.jsx`](src/components/ChatMessage.jsx) | Markdown and citation rendering. |

## Check changes

```powershell
npm --prefix frontend test
npm run frontend:lint
npm run frontend:build
```

UI smoke checks should cover both modes, a partial streamed answer, completion, errors, cancellation, citations, attachment upload, and desktop/mobile layouts.
