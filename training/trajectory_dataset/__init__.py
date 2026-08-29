"""Reproducible trajectory data pipeline for a future central history agent.

Importing this package deliberately performs no dataset downloads, model loads,
or retrieval initialization.
"""

from .schema import SEARCH_HISTORY_TOOL, canonical_id, make_trajectory

__all__ = ["SEARCH_HISTORY_TOOL", "canonical_id", "make_trajectory"]
