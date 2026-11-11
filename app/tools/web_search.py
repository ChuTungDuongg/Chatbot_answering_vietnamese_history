"""Compatibility imports for search providers and the built-in tool."""
import urllib.request
from app.tools.web_providers import WebSearchProvider, LocalOnlyWebProvider, TavilyWebProvider, build_web_search_provider
from app.tools.builtin.search_web.tool import SearchWebInput, SearchWebTool
