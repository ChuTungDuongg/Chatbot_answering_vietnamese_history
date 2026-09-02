# Chat viewport regression

This report covers the frontend layout request. The subsequent attached biography-relation request is implemented and reported separately in `docs/central_v2_relationship_regression.md`.

1. **Exact root cause.** The application already had a viewport-height flex shell and a `min-height: 0` message scroller. The regression was escaped positioned overflow plus ancestor scrolling: visually hidden user-author labels were absolutely positioned relative to `.chat-workspace`, outside the message scroller's containing block. As later turns were added, those labels expanded the workspace's scrollable overflow. The unconditional bottom marker's `scrollIntoView()` scrolled every eligible ancestor, including the `overflow: hidden` application shell. Hidden overflow remains programmatically scrollable.

2. **Responsible rules/components.** `.user-message .message-author` used absolute positioning; `.message-row` and `.thread-scroll` did not establish a containing block. `App.jsx` ran `bottomRef.current.scrollIntoView()` on every messages/status update. A regression against the original code measured `.app-shell.clientHeight=1000`, `scrollHeight=1460`, `scrollTop=460`, with both workspace and sidebar at `top=-460`. The composer bottom was 540 in a 1000-pixel viewport. Body and document remained at zero scroll, proving this was hidden shell scrolling, rather than ordinary body growth.

3. **Why multiple turns exposed it.** Once message positions exceeded the visible message region, the escaped author labels created overflow outside that region. The third-turn fixture reproduced the screenshot symptom; the bug was content-size dependent, not an intentional layout switch at three turns.

4. **Old hierarchy.** Viewport-height app → sidebar + column workspace → header + message scroller/800px content column + composer footer. The composer was already structurally outside the message list. The defective overflow/scroll behavior crossed those structural boundaries.

5. **New hierarchy.** Preserve that DOM hierarchy and visual design. The app is the sole viewport-height owner, using `100dvh` with a `100vh` fallback. Workspace and sidebar inherit its height and have `min-height: 0`. Workspace clips escaped overflow. Each message and the scroller establish their own positioning context. The source overlay uses its existing fixed insets without another independent viewport-height declaration. There are no compensating margins, huge viewport paddings or new `!important` rules.

6. **Scrolling owner.** `.thread-scroll` owns conversation overflow with `flex: 1 1 0%`, `min-height: 0`, vertical scrolling and horizontal containment. `.thread-content` only provides the centered width/padding. The sidebar history retains its own independent scrollbar. Code/table/trace disclosures retain their intentional local overflow behavior.

7. **Composer positioning.** The existing footer remains a natural `flex: 0 0 auto` sibling of the scroller. It is neither fixed nor an overlay, so messages need only the existing modest bottom padding. Safe-area padding remains. Resizing the viewport or growing the textarea changes the message region's available height.

8. **Loading-message fix.** Loading already had natural content height; no viewport-sized loading rule needed removal. Containing the author labels and scroller resolves its displaced appearance. The third-turn test checks the entire assistant loading row is below 130px and visible above the composer. Individual messages do not grow to fill the viewport.

9. **Auto-scroll.** `useChatScroll` directs `scrollTo` only at the message scroller. Submitting or selecting a conversation explicitly enables following. Incoming status/content updates follow only while the reader is within 120px of the bottom. Manual scrollback is preserved across multiple chunks, completion and persisted-message refresh. A ResizeObserver handles content, fonts, composer growth and viewport changes while respecting the same following state. Browser scroll anchoring is disabled in this scroller so it does not compete with explicit following. Empty state remains explicitly `messages.length === 0`, with its current centered design.

10. **Files.** Updated `frontend/src/App.jsx` and styles in `shell.css`, `conversation.css`, `composer.css`, `sidebar.css`, `sources.css`. Added `frontend/src/hooks/useChatScroll.js`, `frontend/e2e/chatLayout.spec.js` and this report. API schemas, backend behavior, request format, citations, source drawer, Agent Trace, typography, theme and brand are unchanged by these frontend changes.

11. **Third-turn result.** A new fixture accumulates five complete turns in the same chat instead of replacing history. It checks two completed turns plus the third loading response, third completion and subsequent turns. Assertions cover docked composer, zero document/body/shell/workspace scrolling, natural loading height, accessible older messages, visible latest turn and no horizontal overflow. The old code failed with a 460px composer displacement; the fix passes on desktop, tablet and mobile.

12. **Long-response result.** A response with 35 paragraphs plus an unbroken string remains inside the message scroller. The composer stays docked through viewport-height changes to 580px and 940px, desktop sidebar collapse/reopen, and mobile/tablet drawer changes. Incremental response tests cover both near-bottom following and preservation of a reader's 180px scrollback position.

13. **Frontend tests.** `npm test`: **14 passed**. `npm run test:ui`: **18 passed in 42.3 seconds**, including the existing nine browser cases and nine new layout cases across desktop/tablet/mobile. Tests use the local Vite server, mocked APIs and a controllable SSE stream; no production backend is contacted. Third-turn and long-response screenshots are produced under `frontend/test-results` and the corrected desktop loading screenshot was visually inspected.

14. **Build/lint.** `npm run build` and `npm run lint` passed. No Modal commands, model inference or training were executed.

For this frontend layout patch specifically:

CHAT SCROLLER OWNS CONVERSATION OVERFLOW  
COMPOSER REMAINS DOCKED TO VIEWPORT  
INDIVIDUAL MESSAGES DO NOT CONSUME VIEWPORT HEIGHT  
THIRD-TURN LOADING LAYOUT IS REGRESSION-TESTED  
NO BACKEND BEHAVIOR CHANGED  
NO MODAL COMMAND EXECUTED
