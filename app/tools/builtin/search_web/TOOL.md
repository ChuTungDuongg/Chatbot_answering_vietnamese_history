---
name: search_web
kind: read_only_network
provider: injected_web_search_provider
requires_env:
- WEB_SEARCH_API_KEY (only with WEB_SEARCH_PROVIDER=tavily)
inputs:
- query
- top_k
outputs:
- 'list of results: title, url, text, score, source_kind=web'
side_effect: false
requires_confirmation: false
---

# search_web

Search external web evidence when configured. Defaults to local-only no-op. Use when permitted external sources are needed beyond the history corpus or Wikipedia. Do not assume a provider is configured or that search results alone prove a claim.

Provider is injected; the default local-only provider returns no results.

Arguments and defaults are declared in `../../definitions/tools.yaml`; the adjacent `tool.py` Pydantic model validates runtime input. Registration does not grant permission: existing request policies still apply.
