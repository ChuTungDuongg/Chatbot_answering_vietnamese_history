# Enable Eval Studio and expose its navigation

User authorized enabling the existing Studio and pushing to main. Keep the current
chat design, owner authentication, feature off switch, and all stored data.

Design: advertise Studio availability through the existing authenticated readiness
response, then show a same-origin link in the chat sidebar. A direct URL alone was
not discoverable. Always showing the link would send users to a disabled workspace.
Enable Studio by default for this owner-only Modal app; retain the general Settings
default of false and honor an explicit false override.

Implementation:

- [x] Add failing readiness and navigation regression tests.
- [x] Wire the capability, sidebar link, Modal flag and deployment documentation.
- [x] Run backend/frontend suites, lint/build and browser navigation checks.

Delivery procedure: selectively stage only these changes, commit and push main;
deploy the verified commit and confirm owner-only access and the visible link.

Local verification: 822 backend, 67 Node, 123 Vitest, 24 Studio Playwright and
51 chatbot Playwright tests passed; ESLint and Vite build passed. Isolate the
Modal packaging test's Windows event-loop policy side effect to preserve MCP
subprocess tests in the full suite.

Do not call model APIs, modify RAG behavior, or commit unrelated GPU/test changes.
