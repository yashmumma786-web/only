"""
Search v3 Classifier - Stub Implementation

Since the prototype database (boundary_inference.db) is not present/disabled,
this module exposes a simplified, functional stub of enrich_with_v3 to avoid
runtime import or attribute errors in the search API.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def enrich_with_v3(stone_results: List[dict]) -> List[dict]:
    """
    Enrich a list of stone result dicts in-place with v3 pattern family predictions.
    Sets pattern_family_v3 and pattern_family_v3_source to None as boundary_inference.db is not present.
    """
    for stone in stone_results:
        stone["pattern_family_v3"] = None
        stone["pattern_family_v3_source"] = None
    return stone_results
