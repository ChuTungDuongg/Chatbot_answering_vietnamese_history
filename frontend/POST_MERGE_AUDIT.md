POST-MERGE FRONTEND LOGIC AUDIT — 2026-09-03

Audited `main` at `553abc29db1ca2076c06a07aaf7419a708cd75f3`, starting from a clean working tree. HEAD is the merge of `758f393` and `cb3e4a8`. HEAD has the same tree as its second parent. The inner merge `cb3e4a8` combined the hook refactor (`c3fa677`) with the newer main-branch UI and clipboard work. Both parent diffs and the complete resulting App, hooks, reducer, configuration, API, components and tests were inspected.

MERGE AUDIT RESULT: ISSUES FOUND AND FIXED LOCALLY.

The initial build passed despite the broken application. Initial lint reported 12 App errors; rendering App failed with `ReferenceError: isEmpty is not defined`. The 64 Node tests passed, but the existing installation lacked Vitest. A clean install then failed because the lockfile lacked the Inter font package. After repairing test dependencies, both existing App tests and all 13 initial audit regression cases failed for the behaviors below.

| Finding | Root cause and local correction | Evidence |
| --- | --- | --- |
| App could not render; conversation/source actions referenced removed variables | The merge retained JSX and handlers from main while removing their definitions. Restored the shared composer, source selection index, scroll hook and busy guards. | Initial lint and App render failures; App integration tests and existing browser tests. |
| OCR/clipboard contract was lost during extraction | The hooks omitted ready attachment IDs, attachment-only submission, upload origin, previews, pending cancellation and total-count validation. Restored the pre-merge frontend behavior through the existing API. | Image-only and clipboard hook failures; all existing clipboard browser cases, including actual native mixed paste. |
| Two upload validators disagreed | `state/uploadQueue.js` accepted renamed unsupported MIME types, empty files and more than five attachments across batches. It now delegates to the original `services/attachments.js` validator. | Existing validation tests plus composed-hook invalid MIME, empty-file and conversation-limit cases. |
| Stream ownership did not span the whole operation | The guard became active only after conversation creation; `central_loading` was omitted; `done` released the UI before final refresh; unmount had no abort cleanup. Added a synchronous request reservation, lifecycle flag, scoped dispatch, guarded callbacks and cancellation. | Double submit during creation, central loading, delayed synchronization, switch, stop, unmount and EOF cases. |
| Stale async results could replace another thread | Loads had no latest-request guard; stream/upload refresh actions carried no conversation scope. Loads now invalidate older controllers; async actions are scoped and abandoned work cannot update the new thread. Shared creation promises prevent concurrent first-conversation creation. | Out-of-order success/failure, StrictMode replay, duplicate creation, late stream events and late upload results after select/new. |
| Latest answer could show older sources | Send did not clear drawer sources; the normalizer skipped an assistant answer with no sources and returned an older answer's citations. Clear the drawer on a new request and stop at the latest assistant, even when its sources are empty. | New-turn/EOF hook case and a failing-before-fix normalizer regression. Historical message source buttons still select their own sources. |
| Installation and test integration were incomplete | The merge removed Playwright's script/dependency, left orphaned lock entries, omitted the font lock entry, and retained tests tied to old App internals/placeholder text. Restored `test:ui` and its original dependency, repaired the lock, updated the accessible selector and made the repository-level error test execute the reducer. | Initial `npm ci` and missing Playwright failures; final clean installation and verification below. |

The conversation load/creation races, missing stream-unmount cleanup and latest-nonempty-source lookup also existed in the first parent's monolithic App. They are confirmed current-state audit findings, not all newly introduced by the merge. The App wiring, clipboard omissions, missing `central_loading`, missing EOF completion fallback and test/install integration are merge/refactor regressions.

1. State ownership audit

| State | Canonical owner |
| --- | --- |
| Theme | `useTheme` state; one effect writes the root dataset and localStorage |
| Inference/chat mode | `useChatMode` state; initialization/migration and persistence use `config/chatModes.js` |
| Conversations and activeConversationId | `useChatSession` reducer |
| Messages and each message's sources/debug trace | `useChatSession` reducer |
| Attachments and pendingUploads | `useChatSession` reducer; `useAttachments` performs API work and dispatches |
| Sources currently displayed in the drawer | Reducer `sources`; deliberately selectable independently of historical message sources |
| Question/composer text | App local state |
| Status and user-visible error | Reducer `status` and `error` |
| List/detail/create loading flags | Reducer `isLoadingConversations`, `isLoadingConversation`, `isCreatingConversation` |
| Conversation deletion target and busy flag | App local state; synchronous action ref prevents re-entry |
| Stream operation running | Reducer `streamRunning`; `isRunning` also recognizes active progress statuses |
| Upload operation running | Reducer `isUploading`, including creation, cancellation cleanup and final refresh |
| Stream AbortController and current request identity | `useChatStream` request ref |
| Upload cancellation and preview URL lifetimes | `useAttachments` refs |
| Sidebar/drawer visibility and active citation index | App local state |

There are no independent competing stores for thread data. Session's ref mirrors committed reducer state solely for reading after awaits. Request refs reserve work synchronously before React rerenders; they are not alternative message stores. `streamFailed` in the reducer protects terminal UI state, while the request-local flag stops later SSE processing immediately. Progress status and operation-running flags deliberately describe different phases: a `done` answer can still be synchronizing.

2. useChatSession audit: bootstrap preserves the conversation list when default-detail loading fails. Selected details atomically replace messages, sources and attachments. Superseded loads cannot clear a newer spinner or display an old error. Creation is deduplicated; deletion uses the latest committed list. App serializes user new/delete actions. Rename patches its target and does not restart bootstrap.

3. useChatStream audit: submit captures mode and ready attachments, reserves one operation, ensures a conversation, appends one user/assistant pair, handles status/delta/sources/debug/error/done, then synchronizes. Responses and cleanup are guarded by request identity and conversation scope. EOF settles the status even without a done event.

4. useAttachments/OCR audit: picker, drop and clipboard share the original validation and multipart API. Clipboard origin, PDF/image MIME handling, 20 MB limit, five-per-conversation limit, previews, image-only sends, failed OCR records and removable attachments are preserved. Queued removal skips upload; in-flight removal waits for the server ID and deletes it. URL cleanup occurs on removal, switch/new and unmount. A cancelled visible item does not prematurely enable send while cleanup remains outstanding.

5. useTheme audit: saved dark/light values, system fallback, lazy initialization, root dataset, toggle and persistence retain the original implementation. App has no duplicate persistence effect. No initialization-timing change was needed.

6. useChatMode audit: all three visible choices (`hybrid`, `three_llm`, `central`) are exercised through App and verified against requests. Stored values and legacy migration remain normalized by the original helper. The initializer's migration write and the hook's persistence write are intentional existing behavior; App has no second persistence owner.

7. Effect/callback dependency audit: bootstrap has an empty dependency list and abort cleanup. Ordinary recreated event callbacks are not bootstrap dependencies. Only `stop` needed stable identity because the conversation-change effect calls it. No circular hook dependency or effect-driven re-bootstrap was found. App passes reducer state, stable dispatch and explicit session APIs to the stream/upload hooks.

8. Conversation switching race audit: latest selection wins for out-of-order success and failure. UI send/upload actions are blocked during loading. UI switching/new is blocked while streaming or uploading; direct hook-level transitions also reject stale async output. Mobile selection/new closes the sidebar; desktop stays open. Both media cases are tested.

9. Streaming/abort audit: running remains true through first-conversation creation, `central_loading` and post-done refresh. Stop cancels immediately and permits a new request without old finally blocks clearing it. Unmount aborts; late transport callbacks are ignored. StrictMode's development effect replay aborts its first bootstrap rather than committing stale results.

10. Message/source alignment audit: thread replacement is atomic. Each assistant retains its own citations and debug trace; a new request clears the drawer's old answer sources. Latest uncited answers have an empty drawer. Failure/abort can retain citations already received for that same partial answer; they do not inherit the previous answer's sources.

11. Error/loading state audit: bootstrap/load errors dispatch in session; App logs load errors without redispatching them. Explicit new/rename/delete failures are surfaced by App. Stream and attachment failures are surfaced by their hooks through the reducer. Replacement-load failure after deletion has one error path. An optional attachment refresh failure is logged while retaining the already-settled attachment data and any original upload error.

12. Merge artifact/dead duplicate audit: removed dangling App references and the inert bottom-scroll effect; reconnected the existing scroll hook. Consolidated the behaviorally conflicting upload validator. No direct App API calls or duplicate App theme/mode persistence remain. Existing API contracts and backend source were left intact.

13. Tests added/updated: `postMerge.test.jsx` adds 14 composed-hook race/lifecycle cases. App tests now cover all three visible modes, mobile/desktop select/new/delete and pending-upload draft preservation. Session tests cover recoverable bootstrap-detail failure and rename during loading. Added an uncited-latest-answer regression. Existing assertions were preserved or corrected to verify the relocated behavior, not removed to make failures disappear. The 39 existing browser cases were reused without changes.

14. Final verification: all passed on the final production diff.

| Command (frontend directory unless noted) | Result |
| --- | --- |
| `npm ci --ignore-scripts --no-audit --no-fund` | Clean installation succeeded; 321 packages |
| `npm test` | 65 Node tests + 49 Vitest UI/hook tests passed |
| `npm run test:ui` | All 39 existing Chrome cases passed across desktop, tablet and mobile |
| `npm run build` | Passed |
| `npm run lint` | Passed; no lint errors |
| `python -m pytest tests/test_frontend_stream_error_contract.py -q` (repository root) | 2 passed |
| `git diff --check` (repository root) | Passed |

Node test runs print pre-existing Vite HMR port warnings; browser runs print color-environment warnings. Concurrent browser/unit execution caused timing failures during an intermediate run; the final full suites ran sequentially with no timeout or assertion changes. The mixed-paste mounting failure found during verification was corrected and the unchanged browser regression passed. HEAD remains `553abc2`.

15. Exact files changed (all local, uncommitted):

- `frontend/POST_MERGE_AUDIT.md` (this report)
- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/src/App.jsx`
- `frontend/src/components/ChatInput.jsx`
- `frontend/src/hooks/useAttachments.js`
- `frontend/src/hooks/useChatSession.js`
- `frontend/src/hooks/useChatStream.js`
- `frontend/src/state/chatSessionReducer.js`
- `frontend/src/state/normalizers.js`
- `frontend/src/state/uploadQueue.js`
- `frontend/tests/normalizers.test.js`
- `frontend/tests/uploadQueue.test.js`
- `frontend/tests/ui/App.test.jsx`
- `frontend/tests/ui/postMerge.test.jsx`
- `frontend/tests/ui/useChatSession.test.jsx`
- `tests/test_frontend_stream_error_contract.py` (frontend-only contract test)

16. Remaining risks: validation uses mocked APIs and local Chrome, including desktop/tablet/mobile viewports. Real OCR/provider output, real backend persistence and other browser engines were intentionally not exercised. Remote cleanup still depends on the server/network accepting the deletion, as before. No claim is made about backend or model quality.

NO SPECULATIVE FIXES WERE MADE.

ROOT CAUSE WAS IDENTIFIED BEFORE EACH FIX.

NO BACKEND AGENT LOGIC WAS MODIFIED.

OCR / CLIPBOARD BEHAVIOR WAS PRESERVED.

NO MODAL COMMAND WAS EXECUTED.

NO GPU OR MODEL INFERENCE WAS EXECUTED.

NO COMMIT WAS CREATED.

NO PUSH WAS PERFORMED.
