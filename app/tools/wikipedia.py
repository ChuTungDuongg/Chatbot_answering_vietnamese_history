"""Compatibility imports; Wikipedia HTTP helpers remain shared."""
import urllib.request
from app.tools.wikipedia_client import WIKIPEDIA_TIMEOUT_SEC, WIKIPEDIA_READ_LIMIT, _api_base, _clean_text, _request_json
from app.tools.builtin.search_wikipedia.tool import SearchWikipediaInput, SearchWikipediaTool
from app.tools.builtin.fetch_wikipedia_page.tool import FetchWikipediaPageInput, FetchWikipediaPageTool
