"""Explicit built-in inventory. Importing does not construct clients or retrievers."""
from .search_history.tool import SearchHistoryTool
from .search_uploaded_documents.tool import SearchUploadedDocumentsTool
from .search_wikipedia.tool import SearchWikipediaTool
from .fetch_wikipedia_page.tool import FetchWikipediaPageTool
from .search_web.tool import SearchWebTool
from .fetch_web_page.tool import FetchPageTool

BUILTIN_TOOL_TYPES = (
    SearchHistoryTool,
    SearchUploadedDocumentsTool,
    SearchWikipediaTool,
    FetchWikipediaPageTool,
    SearchWebTool,
    FetchPageTool,
)
