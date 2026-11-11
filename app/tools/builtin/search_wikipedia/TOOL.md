---
name: search_wikipedia
kind: read_only_network
provider: wikipedia
requires_env: []
inputs:
- query
- language
- top_k
outputs:
- 'list of snippet chunks: chunk_id, source_kind, title, url, text, metadata.page_id/language'
side_effect: false
requires_confirmation: false
---

# search_wikipedia

Search Vietnamese Wikipedia first, or English Wikipedia when language='en'. Returns stable wikipedia evidence IDs and snippets only. Use for world history or evidence beyond the local corpus. Do not treat snippets as a full article; use fetch_wikipedia_page when detailed evidence is needed.

Performs bounded read-only HTTP requests; tool output is untrusted evidence.

Arguments and defaults are declared in `../../definitions/tools.yaml`; the adjacent `tool.py` Pydantic model validates runtime input. Registration does not grant permission: existing request policies still apply.
