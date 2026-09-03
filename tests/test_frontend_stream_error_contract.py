from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_stream_error_stops_reader_and_exports_evidence_message():
    api_source = (ROOT / "frontend" / "src" / "services" / "api.js").read_text(encoding="utf-8")

    assert "EVIDENCE_CONTRACT_FAILURE_MESSAGE" in api_source
    assert "if (parsed.event === \"error\")" in api_source
    assert "sawStreamError = true" in api_source
    assert "await reader.cancel()" in api_source


def test_session_uses_specific_placeholder_for_evidence_contract_error():
    subprocess.run([
        "node", "--input-type=module", "-e", """
        import assert from 'node:assert/strict';
        import { chatSessionReducer as reduce, initialChatSessionState } from './frontend/src/state/chatSessionReducer.js';
        import { EVIDENCE_CONTRACT_FAILURE_MESSAGE } from './frontend/src/config/messages.js';
        let state = reduce(initialChatSessionState, {
          type: 'MESSAGES_APPENDED', messages: [{ id: 'assistant', role: 'assistant', content: '' }],
        });
        state = reduce(state, { type: 'STREAM_ERROR', messageId: 'assistant',
          kind: 'evidence_contract_error', message: 'Evidence rejected' });
        state = reduce(state, { type: 'STREAM_DONE', messageId: 'assistant' });
        assert.equal(state.messages[0].content, EVIDENCE_CONTRACT_FAILURE_MESSAGE);
        assert.equal(state.messages[0].status, 'error');
        assert.equal(state.status, 'error');
        assert.equal(state.error, 'Evidence rejected');
        """,
    ], cwd=ROOT, check=True)
