"""
Replit Internal Search Endpoints

These endpoints are PRIVATE — accessible only to the Main Server via a shared
secret header.  They must never be exposed publicly.

Endpoints:
    GET  /internal/search/health          — readiness probe
    POST /internal/search/query           — text + filter search
    POST /internal/search/by-image        — visual search (multipart upload)

Authentication:
    Every request must carry the header:
        X-Replit-Internal-Secret: <value>
    Value must match the REPLIT_INTERNAL_SECRET environment variable.
    Missing or incorrect secret → HTTP 401, no body detail leaked.

Design rules:
    - No logic duplication: all handlers delegate to existing search_router functions.
    - No public fields added or removed here; sanitisation is done by Main Server.
    - Auth check happens before any processing.
    - REPLIT_INTERNAL_SECRET absence is logged as an error and treated as auth failure.
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form

import io
from PIL import Image
from src.modules.ml_inference import services as ml_service
from src.modules.search import api as _sr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])

_SECRET_ENV_KEY = "REPLIT_INTERNAL_SECRET"


def _require_auth(provided: Optional[str]) -> None:
    """Enforce internal secret.  Raises HTTP 401 on any failure — no detail leaked."""
    expected = os.environ.get(_SECRET_ENV_KEY)
    if not expected:
        logger.error(
            "REPLIT_INTERNAL_SECRET is not set. "
            "All /internal/* requests are being rejected until the secret is configured."
        )
        raise HTTPException(status_code=401, detail="Unauthorized")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")




@router.get("/search/health")
async def internal_search_health(
    x_replit_internal_secret: Optional[str] = Header(default=None),
):
    """Readiness probe.  Returns 200 when the search service is up."""
    _require_auth(x_replit_internal_secret)
    return {"status": "ok"}


@router.post("/search/query")
async def internal_search_query(
    request: _sr.SearchRequest,
    x_replit_internal_secret: Optional[str] = Header(default=None),
):
    """
    Text + filter search.  Accepts the full internal SearchRequest schema.

    Main Server is responsible for:
    - Mapping the public search request → this internal schema.
    - Stripping admin-only fields (admin_user_id, green_hunt, include_suppressed)
      before forwarding.
    - Sanitising the response before returning it to buyers.
    """
    _require_auth(x_replit_internal_secret)
    return await _sr.search_stones(request)


@router.post("/search/by-image")
async def internal_search_by_image(
    file: UploadFile = File(...),
    q: str = Form(""),
    filters_json: str = Form("{}"),
    limit: int = Form(60),
    offset: int = Form(0),
    x_replit_internal_secret: Optional[str] = Header(default=None),
):
    """
    Visual search.  Main Server forwards the uploaded image bytes as multipart.

    Main Server is responsible for:
    - Forwarding the raw image bytes to this endpoint without modification.
    - Sanitising the response before returning it to buyers.
    """
    _require_auth(x_replit_internal_secret)
    
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    del image_bytes
    
    query_embedding = ml_service.compute_image_embedding(image)
    del image
    
    if query_embedding is None:
        raise HTTPException(status_code=500, detail="Failed to compute image embedding")
        
    return await _sr.search_with_embedding(
        query_embedding=query_embedding,
        q=q,
        filters_json=filters_json,
        limit=limit,
        offset=offset,
    )

