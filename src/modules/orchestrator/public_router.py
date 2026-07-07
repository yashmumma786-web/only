import io
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Header, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from PIL import Image
from src.modules.ml_inference import services as ml_svc
from src.modules.search import services as search_svc
from src.modules.orchestrator import services as orch_svc
from src.modules.similar import services as similar_svc
from src.modules.similar.models import RecordSiblingVoteResponse

logger = logging.getLogger(__name__)

router = APIRouter()
_templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent.parent / "templates"))

_SECRET_ENV_KEY = "REPLIT_INTERNAL_SECRET"


def _require_auth(provided: Optional[str]) -> None:
    """Enforce internal secret. Raises HTTP 401 on any failure — no detail leaked."""
    expected = os.environ.get(_SECRET_ENV_KEY)
    if not expected:
        logger.error(
            "REPLIT_INTERNAL_SECRET is not set. "
            "All /internal/* requests are being rejected until the secret is configured."
        )
        raise HTTPException(status_code=401, detail="Unauthorized")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


class InternalSimilarRequest(BaseModel):
    company_stone_id: str = Field(..., description="Canonical stone identity key. stock_id alias not accepted.")
    limit: int = Field(default=50, ge=1, le=100)


@router.post("/api/search-v3/by-image")
async def search_by_image_v3_orchestrator(
    file: UploadFile = File(...),
    q: str = Form(""),
    filters_json: str = Form("{}"),
    limit: int = Form(60),
    offset: int = Form(0)
):

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    del image_bytes

    query_embedding = ml_svc.compute_image_embedding(image)
    del image

    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to compute image embedding")

    return await search_svc.search_with_embedding_v3_facade(
        query_embedding=query_embedding,
        q=q,
        filters_json=filters_json,
        limit=limit,
        offset=offset
    )


@router.post("/internal/search/similar")
async def internal_search_similar(
    request: InternalSimilarRequest,
    x_replit_internal_secret: Optional[str] = Header(default=None),
):
    """
    Typed similar-stone retrieval by company_stone_id.

    Accepts only company_stone_id — stock_id alias is not a valid field on this
    endpoint. Main Server must reject stock_id at its own boundary (HTTP 422)
    before reaching this endpoint.

    Returns the same shape as GET /api/stones/{id}/similar.
    """
    _require_auth(x_replit_internal_secret)

    result = orch_svc.get_visual_similar_stones(
        company_stone_id=request.company_stone_id,
        limit=request.limit,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No embedding found for this stone")
    return result

@router.get("/api/stones/{company_stone_id}/similar", tags=["similar"])
async def get_similar_stones(
    company_stone_id: str,
    limit: int = Query(50, ge=1, le=100)
):
    """Get similar stones based on visual embedding."""
    result = orch_svc.get_visual_similar_stones(company_stone_id, limit=limit)
    if not result:
        raise HTTPException(status_code=404, detail="No embedding found for this stone")
    return result


@router.get("/api/similar-explorer", tags=["similar"])
async def similar_explorer(
    anchor_id: str = Query(..., description="company_stone_id of the anchor stone"),
    mode: str = Query("buyer", description="buyer or admin"),
    search_version: str = Query("v2", description="legacy, v2 or v3"),
):
    """Similar Stone Explorer — resolver for a stone."""
    result = orch_svc.resolve_similar_stones(anchor_id, mode=mode, search_version=search_version)

    similar_svc.log_event(
        event_type="similar_drawer_opened",
        anchor_id=anchor_id,
        role=mode,
        search_version=search_version,
    )

    return result


@router.post("/api/similar-explorer/mark-not-related", response_model=RecordSiblingVoteResponse, tags=["similar"])
async def mark_not_related(body: dict):
    """Mark a candidate as not related to anchor."""
    anchor_id = body.get("anchor_id", "")
    candidate_id = body.get("candidate_id", "")
    reason = body.get("reason", "")

    orch_svc.record_sibling_vote(
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        vote=-1,
        admin_user_id=body.get("admin_user_id"),
    )

    similar_svc.log_event(
        event_type="similar_mark_not_related",
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        role=body.get("role", "admin"),
        search_version=body.get("search_version", ""),
        detail=reason,
    )
    return RecordSiblingVoteResponse(ok=True, message=f"Marked not related: {reason}")


@router.post("/api/similar-explorer/mark-true-linked", response_model=RecordSiblingVoteResponse, tags=["similar"])
async def mark_true_linked(body: dict):
    """Mark a candidate as truly linked to anchor."""
    anchor_id = body.get("anchor_id", "")
    candidate_id = body.get("candidate_id", "")

    orch_svc.record_sibling_vote(
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        vote=1,
        admin_user_id=body.get("admin_user_id"),
    )

    similar_svc.log_event(
        event_type="similar_mark_true_linked",
        anchor_id=anchor_id,
        candidate_id=candidate_id,
        role=body.get("role", "admin"),
        search_version=body.get("search_version", ""),
    )
    return RecordSiblingVoteResponse(ok=True, message="Marked as true linked")


@router.get("/similar/{company_stone_id}", response_class=HTMLResponse, tags=["similar-pages"])
async def similar_explorer_page(
    request: Request,
    company_stone_id: str,
):
    """Similar Stone Explorer — HTML Page."""
    is_admin = bool(request.session.get("is_admin")) if hasattr(request, "session") else False
    mode = "admin" if is_admin else "buyer"
    search_version = "v3"

    result = orch_svc.resolve_similar_stones(company_stone_id, mode=mode, search_version=search_version)

    return _templates.TemplateResponse(request, "similar_explorer_page.html", {
        "request": request,
        "anchor_id": company_stone_id,
        "admin_mode": is_admin,
        "search_version": search_version,
        "anchor": result["anchor"],
        "sections": result["sections"],
        "meta": result.get("meta") or {},
    })

