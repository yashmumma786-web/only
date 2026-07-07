"""
Similar module — public service API.

This is the ONLY file that other modules (specifically orchestrator.services)
are allowed to import from within ``src.modules.similar``.

All other files in this module are internal implementation details.

Dependency rules (enforced at this boundary):
  ✅ Callers may import: similar.services
  ❌ Callers must NOT import: similar.explorer, similar.repository, similar.router

Public surface:
  resolve()                — 3-section similar-stones resolver
  get_visual_similar()     — CLIP-similarity ranking for a stone
  log_event()              — write to similar_explorer.db event log
"""

from typing import Any, Dict, List, Optional
from src.modules.similar.explorer import (resolve as _resolve,log_event as _log_event,compute_visual_similar)


def resolve(
    anchor_id: str,
    stones: List[Dict],
    mode: str = "buyer",
    search_version: str = "v2",
) -> Dict[str, Any]:
    """Resolve the 3-section similar-stones contract for an anchor.

    ``stones`` is the fully hydrated stone list — passed in by the
    orchestrator (which fetched it from search.services).  This module
    does NOT fetch stones itself.

    Returns ``{anchor, sections: {section_a, section_b, section_c}, meta}``.
    """
    return _resolve(anchor_id, stones, mode=mode, search_version=search_version)


def log_event(
    event_type: str,
    anchor_id: str = "",
    candidate_id: str = "",
    section: str = "",
    tier: str = "",
    role: str = "",
    search_version: str = "",
    detail: str = "",
) -> None:
    """Write an interaction event to similar_explorer.db."""
    _log_event(
        event_type=event_type,
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        section=section,
        tier=tier,
        role=role,
        search_version=search_version,
        detail=detail,
    )


def get_visual_similar(company_stone_id: str, stones: List[Dict], limit: int = 50) -> Optional[Dict[str, Any]]:
    """Get similar stones based on visual embedding."""
    return compute_visual_similar(company_stone_id, stones, limit=limit)

