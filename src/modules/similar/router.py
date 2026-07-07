"""
Similar Stone Explorer — FastAPI HTTP Router.

This router manages all endpoints for the similar stone explorer feature.
Crucial dependency rule:
- Uses similar.services for self-contained similar-domain operations.
- NEVER imports from search/
- NEVER imports from orchestrator/
"""

import json
from fastapi import APIRouter
from src.modules.similar import services as similar_svc

router = APIRouter(prefix="/api", tags=["similar"])


@router.post("/similar-explorer/event")
async def similar_explorer_event(body: dict):
    """Log a Similar Explorer UI event."""
    similar_svc.log_event(
        event_type=body.get("event_type", "unknown"),
        anchor_id=body.get("anchor_id", ""),
        candidate_id=body.get("candidate_id", ""),
        section=body.get("section", ""),
        tier=body.get("tier", ""),
        role=body.get("role", ""),
        search_version=body.get("search_version", ""),
        detail=json.dumps(body.get("detail", {})) if body.get("detail") else "",
    )
    return {"ok": True}
