"""
Search module — public service API.

This file acts as the boundary surface for search-related operations.
All other modules (specifically orchestrator) calling search logic
must go through this file.

Dependency rules:
  ❌ Must NOT import from similar/
  ❌ Must NOT import from orchestrator/
"""

from typing import Dict, List, Any
import numpy as np
from src.modules.search.api import _load_all_stone_data_batched, search_with_embedding_v3


def get_hydrated_stone_list() -> List[Dict]:
    """Return the fully hydrated stone list from the in-memory cache.
    This is the only public entry point for cross-domain stone data access.
    Callers: orchestrator.services only.
    """
    _, stones = _load_all_stone_data_batched()
    return stones


async def search_with_embedding_v3_facade(
    query_embedding: np.ndarray,
    q: str = "",
    filters_json: str = "{}",
    limit: int = 60,
    offset: int = 0
) -> Dict[str, Any]:
    """Public search boundary facade for embedding-based visual search."""
    return await search_with_embedding_v3(
        query_embedding=query_embedding,
        q=q,
        filters_json=filters_json,
        limit=limit,
        offset=offset
    )



