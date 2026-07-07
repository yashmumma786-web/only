"""
Main Server Search Gateway Router

Live FastAPI router that implements the Main Server search boundary.
Registered in main.py at /api/search* — the only public search surface.

This router enforces the full SearchProjectionV1 contract:
  - Public response normalization is ALLOWLIST-based.  No passthrough.
  - Hydration is batch-only.  Never N+1.
  - Failure contract: 1 attempt, 0 retries, 1500ms timeout.
    Replit 5xx / timeout → generic 503.  No topology leaked.
    Replit 4xx → propagated as-is (e.g. 404 for unknown stone).
  - thumbnail_url is always null in buyer-facing responses.
  - stock_id alias is rejected at schema validation → HTTP 422.
  - in_stock_only is absent from the public schema (deferred).
  - filters_json on by-image is rebuilt through PublicSearchFilters;
    all forbidden/internal fields stripped before forwarding to Replit.
  - Every request logs: request_id, replit_rt_ms, main_hydration_ms, total_ms.

Transport: httpx → /internal/search/* (Replit private endpoints).
  Auth: X-Replit-Internal-Secret header (REPLIT_INTERNAL_SECRET env var).
  Timeout: 1500 ms.  Zero retries.
  Request ID forwarded as X-Request-Id for log correlation.

On the actual separate Main Server deployment, configure:
    REPLIT_SEARCH_URL       — private Replit service base URL
    REPLIT_INTERNAL_SECRET  — shared secret
and the routes at /api/search* become the sole public search surface.
"""

import json as _json
import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, model_validator
from src.modules.search import api as _sr
from src.modules.ingestion import services as ingestion_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search-gateway"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_REPLIT_TIMEOUT_S: float = 1.5  # 1500 ms — non-negotiable failure contract


def _replit_base_url() -> str:
    url = os.environ.get("REPLIT_SEARCH_URL").rstrip("/")
    return url

def _replit_secret() -> str:
    return os.environ.get("REPLIT_INTERNAL_SECRET")

# ---------------------------------------------------------------------------
# Phase-1 static filter vocabulary (locked; no DB call)
# ---------------------------------------------------------------------------

_STATIC_FILTERS: Dict[str, Any] = {
    "base_colors": [
        "grey", "white", "blue", "black", "pink", "red", "rose",
        "brown", "yellow", "orange", "green", "beige", "charcoal", "gold", "purple",
    ],
    "accent_colors": [
        "green", "gold", "red", "pink", "yellow", "blue", "purple", "orange",
    ],
    "pattern_families": [
        "uniform", "linear", "webbed", "breccia", "cloudy", "bold_veined",
    ],
    "stone_types": [],
    "dominant_tones": ["neutral", "green", "red", "warm", "cool"],
    "dominant_hues": [
        "neutral", "green", "red", "orange_brown", "yellow_beige", "blue_grey",
    ],
}

_FORBIDDEN_FILTER_KEYS = frozenset({
    "in_stock_only",
    "admin_user_id",
    "include_suppressed",
    "debug_scores",
    "green_hunt",
    "green_hunt_unlabeled_only",
    "green_hunt_include_suppressed",
})

# ---------------------------------------------------------------------------
# Public request schemas
# ---------------------------------------------------------------------------

class PublicSearchFilters(BaseModel):
    base_colors: List[str] = Field(default_factory=list)
    base_color_mode: str = "smart"
    veining_min: float = 0.0
    veining_max: float = 1.0
    patterns: List[str] = Field(
        default_factory=list,
        description=(
            "DEPRECATED: use `pattern_families`. Pattern P4 contract repair: "
            "non-empty values are alias-normalized through "
            "tag_storage.PATTERN_FAMILY_ALIASES and folded into "
            "`pattern_families` at request validation. After validation this "
            "field is a one-way mirror of `pattern_families`."
        ),
    )
    accent_colors: List[str] = Field(default_factory=list)
    min_confidence: float = 0.0
    stone_types: List[str] = Field(default_factory=list)
    cloudiness_pref: Optional[float] = None
    cloudiness_strict: bool = False
    cloudiness_min: Optional[float] = None
    cloudiness_max: Optional[float] = None
    dominant_tones: List[str] = Field(default_factory=list)
    dominant_tone_strict: bool = False
    dominant_hues: List[str] = Field(default_factory=list)
    dominant_hue_strict: bool = False
    strict_exclude: bool = False
    pattern_families: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consolidate_pattern_filter_fields(self) -> "PublicSearchFilters":
        """Pattern P4 (J.6) — same consolidation as the internal SearchFilters.

        Delegates to the shared helper in `search_router` so the public
        gateway and the internal schema cannot drift.
        """
        _sr._consolidate_pattern_filter(self)
        return self


class PublicSearchRequest(BaseModel):
    q: str = ""
    filters: PublicSearchFilters = Field(default_factory=PublicSearchFilters)
    sort: str = "best_match"
    limit: int = 60
    offset: int = 0


class PublicSimilarRequest(BaseModel):
    company_stone_id: str = Field(
        ...,
        description="Canonical stone identity key.  stock_id alias is not accepted.",
    )
    limit: int = Field(default=50, ge=1, le=100)

# ---------------------------------------------------------------------------
# Public response schemas
# ---------------------------------------------------------------------------

class PublicAccentColor(BaseModel):
    color: str
    strength: float


class PublicStoneResult(BaseModel):
    id: str
    stone_name: Optional[str] = None
    base_color: Optional[str] = None
    base_color_confidence: float = 0.0
    pattern_family: Optional[str] = None
    vein_intensity: Optional[str] = None
    accent_colors: List[PublicAccentColor] = Field(default_factory=list)
    visual_busyness: float = 0.0
    drama_score: float = 0.0
    dominant_tone: Optional[str] = None
    dominant_tone_confidence: float = 0.0
    dominant_hue: Optional[str] = None
    dominant_hue_confidence: float = 0.0
    similarity_score: Optional[float] = None
    thumbnail_url: None = None  # Always null — vendor-code path not safe for buyers


class PublicSearchResponse(BaseModel):
    results: List[PublicStoneResult]
    total: int
    query: str
    request_id: str
    search_request_id: str = Field(
        default="",
        description=(
            "Server-issued UUID from the internal search engine for this request. "
            "Matches the search_log_db entry. Use in search feedback submissions."
        ),
    )
    replit_rt_ms: float
    main_hydration_ms: float
    total_ms: float


class PublicSimilarResponse(BaseModel):
    source_id: str
    results: List[PublicStoneResult]
    count: int
    request_id: str
    replit_rt_ms: float
    main_hydration_ms: float
    total_ms: float


# ---------------------------------------------------------------------------
# Request mapper — public → internal Replit schema
# ---------------------------------------------------------------------------

def _map_search_request(pub: PublicSearchRequest) -> Dict[str, Any]:
    """
    Convert a public search request to the internal Replit SearchRequest schema.
    Invariants:
    - in_stock_only: always False (deferred per architecture lock).
    - admin_user_id / include_suppressed / green_hunt: never forwarded.
    - debug_scores: always False for public callers.
    """
    return {
        "q": pub.q,
        "sort": pub.sort,
        "limit": pub.limit,
        "offset": pub.offset,
        "debug_scores": False,
        "filters": {
            "in_stock_only": False,
            "stone_types": pub.filters.stone_types,
            "base_colors": pub.filters.base_colors,
            "base_color_mode": pub.filters.base_color_mode,
            "veining_min": pub.filters.veining_min,
            "veining_max": pub.filters.veining_max,
            "patterns": pub.filters.patterns,
            "accent_colors": pub.filters.accent_colors,
            "min_confidence": pub.filters.min_confidence,
            "cloudiness_pref": pub.filters.cloudiness_pref,
            "cloudiness_strict": pub.filters.cloudiness_strict,
            "cloudiness_min": pub.filters.cloudiness_min,
            "cloudiness_max": pub.filters.cloudiness_max,
            "dominant_tones": pub.filters.dominant_tones,
            "dominant_tone_strict": pub.filters.dominant_tone_strict,
            "dominant_hues": pub.filters.dominant_hues,
            "dominant_hue_strict": pub.filters.dominant_hue_strict,
            "strict_exclude": pub.filters.strict_exclude,
            "pattern_families": pub.filters.pattern_families,
        },
    }


def _sanitize_by_image_filters(raw_json: str) -> str:
    """
    Parse, strip forbidden fields, rebuild through PublicSearchFilters, re-serialize.
    Forbidden: in_stock_only, admin_user_id, include_suppressed, debug_scores,
    green_hunt, green_hunt_unlabeled_only, green_hunt_include_suppressed.
    Malformed JSON → safe empty filter set (fail-closed).
    """
    try:
        raw: Dict[str, Any] = _json.loads(raw_json) if raw_json else {}
        for key in _FORBIDDEN_FILTER_KEYS:
            raw.pop(key, None)
        allowed = set(PublicSearchFilters.model_fields.keys())
        safe = {k: v for k, v in raw.items() if k in allowed}
        return _json.dumps(PublicSearchFilters(**safe).model_dump())
    except Exception:
        return _json.dumps(PublicSearchFilters().model_dump())


# ---------------------------------------------------------------------------
# Batch hydration — Main-owned data source
# ---------------------------------------------------------------------------

def _batch_hydrate(
    company_stone_ids: List[str],
    request_id: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Batch hydration: fetch Main-owned attributes for all result stone IDs in ONE query.
    This function is the N+1 gate — must never be called once per stone.

    One SELECT covers all IDs: WHERE company_stone_id IN (…).
    Timeout enforced at 1500ms.  Any failure → HTTPException(503), no detail leaked.

    Phase-1: No additional Main-owned fields are merged into public results
    (all displayable fields already come from Replit).  The N+1-free pattern is
    established here for phase-2 extensions (live availability, pricing, visibility).

    Phase-2 adoption: replace the SELECT stub with the real Main DB query and add
    the returned fields to PublicStoneResult.
    """
    if not company_stone_ids:
        return {}

    t0 = time.perf_counter()
    try:
        hydrated_ids = ingestion_service.get_batch_hydrate(company_stone_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > _REPLIT_TIMEOUT_S * 1000:
            logger.error(
                "batch_hydrate_timeout request_id=%s elapsed_ms=%.1f",
                request_id, elapsed_ms,
            )
            raise HTTPException(status_code=503, detail="Service unavailable")

        return {cid: {} for cid in hydrated_ids}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "batch_hydrate_error request_id=%s error_type=%s",
            request_id, type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Service unavailable")


# ---------------------------------------------------------------------------
# Sanitizer — allowlist-based, never passthrough
# ---------------------------------------------------------------------------

def _sanitize_stone(raw: Dict[str, Any]) -> PublicStoneResult:
    """
    Allowlist normalization.  Fields renamed: company_stone_id → id.
    Stripped: vendor_name, has_manual_override, final_score, tags, dataset_id,
    batch_id, block_id, block_no, knn_base_color, bg_color, main_color,
    cloudiness_score, network_thickness, has_bold_accent_vein, thumbnail_url.
    """
    accent_raw = raw.get("accent_colors") or []
    accent_public = [
        PublicAccentColor(color=a["color"], strength=a["strength"])
        for a in accent_raw
        if isinstance(a, dict) and "color" in a and "strength" in a
    ]
    return PublicStoneResult(
        id=raw["company_stone_id"],
        stone_name=raw.get("stone_name"),
        base_color=raw.get("base_color"),
        base_color_confidence=float(raw.get("base_color_confidence") or 0.0),
        pattern_family=raw.get("pattern_family"),
        vein_intensity=raw.get("vein_intensity"),
        accent_colors=accent_public,
        visual_busyness=float(raw.get("visual_busyness") or 0.0),
        drama_score=float(raw.get("drama_score") or 0.0),
        dominant_tone=raw.get("dominant_tone"),
        dominant_tone_confidence=float(raw.get("dominant_tone_confidence") or 0.0),
        dominant_hue=raw.get("dominant_hue"),
        dominant_hue_confidence=float(raw.get("dominant_hue_confidence") or 0.0),
        similarity_score=raw.get("similarity_score"),
        thumbnail_url=None,
    )


def _sanitize_results(raw_results: List[Dict[str, Any]]) -> List[PublicStoneResult]:
    return [
        _sanitize_stone(r)
        for r in raw_results
        if isinstance(r, dict) and r.get("company_stone_id")
    ]


# ---------------------------------------------------------------------------
# Replit HTTP adapter
# ---------------------------------------------------------------------------

async def _call_replit(
    method: str,
    path: str,
    *,
    json: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict] = None,
    request_id: str,
) -> Dict[str, Any]:
    """
    HTTP adapter for Replit internal endpoints.  Enforced failure contract:
      - 1 attempt, 0 retries
      - 1500 ms timeout
      - Replit 5xx / network error / timeout → HTTPException(503), no detail leaked
      - Replit 4xx → HTTPException(status=replit_status, detail=replit_detail)
        (e.g. 404 for unknown stone propagates as 404 to the public caller)
      - X-Replit-Internal-Secret header on every request
      - X-Request-Id forwarded for log correlation

    REPLIT_INTERNAL_SECRET absent → immediate 503 + error log (config guard).
    REPLIT_SEARCH_URL and REPLIT_INTERNAL_SECRET absent → immediate 503 (fail-closed).
    """
    secret = _replit_secret()
    if not secret:
        logger.error(
            "REPLIT_INTERNAL_SECRET not configured request_id=%s", request_id
        )
        raise HTTPException(status_code=503, detail="Service unavailable")

    base_url = _replit_base_url()
    if not base_url:
        logger.error(
            "REPLIT_SEARCH_URL not configured request_id=%s", request_id
        )
        raise HTTPException(status_code=503, detail="Service unavailable")
    headers: Dict[str, str] = {
        "X-Replit-Internal-Secret": secret,
        "X-Request-Id": request_id,
    }

    try:
        async with httpx.AsyncClient(timeout=_REPLIT_TIMEOUT_S) as client:
            if method == "POST" and files:
                resp = await client.post(
                    f"{base_url}{path}",
                    data=data,
                    files=files,
                    headers=headers,
                )
            elif method == "POST":
                resp = await client.post(
                    f"{base_url}{path}",
                    json=json,
                    headers={**headers, "Content-Type": "application/json"},
                )
            else:
                resp = await client.get(
                    f"{base_url}{path}",
                    headers=headers,
                )

        if resp.status_code >= 500:
            logger.error(
                "replit_5xx request_id=%s status=%d path=%s",
                request_id, resp.status_code, path,
            )
            raise HTTPException(status_code=503, detail="Service unavailable")

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", "Request failed")
            except Exception:
                detail = "Request failed"
            logger.warning(
                "replit_4xx request_id=%s status=%d path=%s",
                request_id, resp.status_code, path,
            )
            raise HTTPException(status_code=resp.status_code, detail=detail)

        return resp.json()

    except httpx.TimeoutException:
        logger.error(
            "replit_timeout request_id=%s path=%s timeout=%.1fs",
            request_id, path, _REPLIT_TIMEOUT_S,
        )
        raise HTTPException(status_code=503, detail="Service unavailable")

    except HTTPException:
        raise

    except Exception as exc:
        logger.error(
            "replit_adapter_error request_id=%s path=%s error_type=%s",
            request_id, path, type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Service unavailable")


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=PublicSearchResponse)
async def public_search(body: PublicSearchRequest) -> PublicSearchResponse:
    """POST /api/search — public text + filter search."""
    request_id = str(uuid.uuid4())
    t_total = time.perf_counter()

    t_replit = time.perf_counter()
    raw_response = await _call_replit(
        "POST", "/internal/search/query",
        json=_map_search_request(body),
        request_id=request_id,
    )
    replit_rt_ms = (time.perf_counter() - t_replit) * 1000

    raw_results: List[Dict[str, Any]] = raw_response.get("results", [])
    stone_ids = [r["company_stone_id"] for r in raw_results if r.get("company_stone_id")]

    t_hydration = time.perf_counter()
    hydrated_map = _batch_hydrate(stone_ids, request_id)
    main_hydration_ms = (time.perf_counter() - t_hydration) * 1000

    public_results = _sanitize_results(raw_results)
    total_ms = (time.perf_counter() - t_total) * 1000

    logger.info(
        "search request_id=%s replit_rt_ms=%.1f main_hydration_ms=%.1f total_ms=%.1f",
        request_id, replit_rt_ms, main_hydration_ms, total_ms,
    )

    return PublicSearchResponse(
        results=public_results,
        total=raw_response.get("total", len(public_results)),
        query=raw_response.get("query", body.q),
        request_id=request_id,
        search_request_id=raw_response.get("search_request_id", ""),
        replit_rt_ms=round(replit_rt_ms, 2),
        main_hydration_ms=round(main_hydration_ms, 2),
        total_ms=round(total_ms, 2),
    )


@router.post("/by-image", response_model=PublicSearchResponse)
async def public_search_by_image(
    file: UploadFile = File(...),
    q: str = Form(""),
    filters_json: str = Form("{}"),
    limit: int = Form(60),
    offset: int = Form(0),
) -> PublicSearchResponse:
    """POST /api/search/by-image — public visual search.

    filters_json is sanitized through _sanitize_by_image_filters before forwarding:
    all forbidden/deferred fields stripped, remaining fields rebuilt through
    PublicSearchFilters allowlist.
    """
    request_id = str(uuid.uuid4())
    t_total = time.perf_counter()

    safe_filters_json = _sanitize_by_image_filters(filters_json)
    image_bytes = await file.read()

    t_replit = time.perf_counter()
    raw_response = await _call_replit(
        "POST", "/internal/search/by-image",
        data={"q": q, "filters_json": safe_filters_json, "limit": limit, "offset": offset},
        files={"file": (file.filename or "upload.jpg", image_bytes, file.content_type or "image/jpeg")},
        request_id=request_id,
    )
    replit_rt_ms = (time.perf_counter() - t_replit) * 1000

    raw_results: List[Dict[str, Any]] = raw_response.get("results", [])
    stone_ids = [r["company_stone_id"] for r in raw_results if r.get("company_stone_id")]

    t_hydration = time.perf_counter()
    hydrated_map = _batch_hydrate(stone_ids, request_id)
    main_hydration_ms = (time.perf_counter() - t_hydration) * 1000

    public_results = _sanitize_results(raw_results)
    total_ms = (time.perf_counter() - t_total) * 1000

    logger.info(
        "search_by_image request_id=%s replit_rt_ms=%.1f main_hydration_ms=%.1f total_ms=%.1f",
        request_id, replit_rt_ms, main_hydration_ms, total_ms,
    )

    return PublicSearchResponse(
        results=public_results,
        total=raw_response.get("total", len(public_results)),
        query=raw_response.get("query", q),
        request_id=request_id,
        search_request_id=raw_response.get("search_request_id", ""),
        replit_rt_ms=round(replit_rt_ms, 2),
        main_hydration_ms=round(main_hydration_ms, 2),
        total_ms=round(total_ms, 2),
    )


@router.post("/similar", response_model=PublicSimilarResponse)
async def public_search_similar(body: PublicSimilarRequest) -> PublicSimilarResponse:
    """POST /api/search/similar — typed similar-stone retrieval.

    stock_id alias rejected by schema validation (HTTP 422) before any processing.
    Replit 404 (embedding not found) propagates as 404.
    Replit 5xx / timeout → generic 503.
    """
    request_id = str(uuid.uuid4())
    t_total = time.perf_counter()

    t_replit = time.perf_counter()
    raw_response = await _call_replit(
        "POST", "/internal/search/similar",
        json={"company_stone_id": body.company_stone_id, "limit": body.limit},
        request_id=request_id,
    )
    replit_rt_ms = (time.perf_counter() - t_replit) * 1000

    raw_results: List[Dict[str, Any]] = raw_response.get("similar_stones", [])
    stone_ids = [r["company_stone_id"] for r in raw_results if r.get("company_stone_id")]

    t_hydration = time.perf_counter()
    hydrated_map = _batch_hydrate(stone_ids, request_id)
    main_hydration_ms = (time.perf_counter() - t_hydration) * 1000

    public_results = _sanitize_results(raw_results)
    total_ms = (time.perf_counter() - t_total) * 1000

    logger.info(
        "search_similar request_id=%s source=%s replit_rt_ms=%.1f main_hydration_ms=%.1f total_ms=%.1f",
        request_id, body.company_stone_id, replit_rt_ms, main_hydration_ms, total_ms,
    )

    return PublicSimilarResponse(
        source_id=body.company_stone_id,
        results=public_results,
        count=len(public_results),
        request_id=request_id,
        replit_rt_ms=round(replit_rt_ms, 2),
        main_hydration_ms=round(main_hydration_ms, 2),
        total_ms=round(total_ms, 2),
    )


@router.get("/filters")
async def public_search_filters() -> Dict[str, Any]:
    """GET /api/search/filters — static phase-1 filter vocabulary.
    No DB call.  in_stock_only absent.  stone_types empty (Main populates).
    """
    return _STATIC_FILTERS
