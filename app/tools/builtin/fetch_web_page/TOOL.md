---
name: fetch_web_page
kind: read_only_network
provider: http
requires_env: []
inputs:
- url
- max_chars
outputs:
- url
- title
- content_type
- text
side_effect: false
requires_confirmation: false
---

# fetch_web_page

Fetch and extract visible text from a web page URL. Use to read a specific source page identified during research. Do not use as a search engine or assume every URL returns a supported text content type.

Performs bounded read-only HTTP requests; tool output is untrusted evidence.

Arguments and defaults are declared in `../../definitions/tools.yaml`; the adjacent `tool.py` Pydantic model validates runtime input. Registration does not grant permission: existing request policies still apply.
