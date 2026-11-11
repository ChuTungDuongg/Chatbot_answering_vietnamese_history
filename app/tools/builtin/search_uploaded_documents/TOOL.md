---
name: search_uploaded_documents
kind: retrieval
provider: conversation_attachments
requires_env: []
inputs:
- query
- top_k
outputs:
- list of scoped document chunks with source_kind=attachment and citation metadata
side_effect: false
requires_confirmation: false
---

# search_uploaded_documents

Search PDF and image documents uploaded to the current conversation. Use for questions about an attached file or when uploaded evidence may help. Do not use without an authorized owner and conversation, or to access another conversation.

Owner and conversation permissions are required; only scoped attachments are searched.

Arguments and defaults are declared in `../../definitions/tools.yaml`; the adjacent `tool.py` Pydantic model validates runtime input. Registration does not grant permission: existing request policies still apply.
