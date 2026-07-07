"""
StoneStocks Search API Router

Endpoints:
- POST /api/search - Text + filter search with weighted ranking
- POST /api/search/by-image - Visual search with in-memory embedding
- POST /api/search-v3 - Search v3: same as /api/search + M4b prototype + selective U↔L rerank
- POST /api/search-v3/by-image - Search v3 image upload
- GET /api/stones/{id} - Stone details with tags
- GET /api/stones/{id}/similar - Similar stones via embedding

Hard rules:
- No disk writes for uploaded images
- Process images in-memory only
- Manual overrides always win over AI tags
"""

import json
import time
import uuid
import logging
import threading
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator

from src.modules.ingestion import services as ingestion_service
from src.modules.search import search_log_db as _slog
from src.modules.taxonomy import services as taxonomy_client

from src.modules.ml_inference import services as ml_svc
from src.modules.search import name_search
from src.modules.search.low_band_display_suppression import (
    get_config as _lbds_get_config,
    should_suppress_ai_row as _lbds_should_suppress,
)
from src.modules.search.v3_classifier import enrich_with_v3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api", tags=["search"])

pages_router = APIRouter(tags=["search-pages"])

_TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_BASE_COLOR_CHIP_ORDER = [
    "white", "black", "grey", "beige", "brown", "blue", "green", "pink", "red", "yellow", "multi",
]


@pages_router.get("/find", response_class=HTMLResponse)
def find_page(request: Request):
    """Public-facing stone search page."""
    is_admin = bool(request.session.get("is_admin")) if hasattr(request, "session") else False
    return _templates.TemplateResponse(
        request,
        "search_main.html",
        {
            "request": request,
            "trial_active": False,
            "active_dataset_id": ingestion_service.get_active_dataset_id() or "none",
            "base_color_chip_order": _BASE_COLOR_CHIP_ORDER,
            "is_admin": is_admin,
            "pattern_family_options": taxonomy_client.get_pattern_family_values(),
        },
    )

EMBEDDINGS_CACHE_DIR = Path("cache/embeddings_v3")

# ---------------------------------------------------------------------------
# Stone data TTL cache
# ---------------------------------------------------------------------------
# State machine (three states per request):
#   HIT        — age < soft TTL (96 s). Served immediately.
#   STALE_SERVE— age >= soft TTL but payload present. Served immediately from
#                stale data; a background daemon thread rebuilds concurrently.
#   MISS       — payload is None (first start or explicit invalidation).
#                Synchronous rebuild; concurrent requests wait on an Event.
#
# Active-dataset-id mini-cache: The active dataset_id is also cached in-process
# (TTL = _DATASET_ID_CACHE_TTL) so that cache-hit requests never execute any SQL.
# When the stone cache is invalidated (e.g. after a write), the dataset-id cache
# is also invalidated so the next miss picks up any dataset switch.
#
# A generation counter prevents a stale background rebuild from overwriting a
# fresh post-invalidation rebuild.

_STONE_CACHE_TTL: float = 120.0         # seconds — TTL for lightweight candidate cache
_DATASET_ID_CACHE_TTL: float = 5.0     # seconds — short enough to detect dataset switches
_stone_cache_lock = threading.Lock()

# Tier-1 cache: lightweight candidates (Phase A output, no display_tags).
# Used by text-only search path (Phase B deferred to final page).
_lw_cache_dataset_id: Optional[str] = None
_lw_cache_payload: Optional[List[Dict]] = None  # lightweight dicts (no display_tags)
_lw_cache_ts: float = 0.0
_lw_cache_hits: int = 0
_lw_cache_misses: int = 0

# Facet-tag cache: stores the merged display_tags for all lw candidates so that
# _hydrate_facet_tags_for_candidates() skips 4 SQL queries on every cached request.
# Cleared in sync with _lw_cache_payload by invalidate_stone_cache().
_lw_facet_tags: Optional[Dict[str, Dict]] = None  # {company_stone_id: display_tags_dict}

# Tier-2 cache: fully hydrated stones (Phase A + Phase B output, display_tags populated).
# Used by tag-filtered search path and non-search routes (image search, correction).
_stone_cache_dataset_id: Optional[str] = None
_stone_cache_payload: Optional[List[Dict]] = None
_stone_cache_ts: float = 0.0
_stone_cache_generation: int = 0       # incremented on every invalidation
_stone_cache_hits: int = 0
_stone_cache_misses: int = 0

# --- rebuild coordination lock (guards rebuilding flag + event) ---
_stone_cache_rebuilding: bool = False
_stone_cache_rebuild_event: Optional[threading.Event] = None

_active_dataset_id_cached: Optional[str] = None
_active_dataset_id_ts: float = 0.0

# Last sub-stage timing from most recent search (cache miss or page-hydration).
# Populated on every request and exposed in API debug payload.
_last_db_substage_timing: Dict[str, Any] = {}


def invalidate_stone_cache() -> None:
    """Force a fresh load on the next search request.

    Must be called by every write path that changes fields used in search-card
    payload, filtering, name matching, tag fallback resolution, or
    pattern/base-color display logic.  Also resets the active-dataset-id mini-cache
    so that a dataset switch is detected immediately on the next miss.
    """
    global _stone_cache_payload, _stone_cache_ts, _stone_cache_generation
    global _lw_cache_payload, _lw_cache_ts, _lw_facet_tags
    global _active_dataset_id_cached, _active_dataset_id_ts
    with _stone_cache_lock:
        _stone_cache_generation += 1
        _stone_cache_payload = None
        _stone_cache_ts = 0.0
        _lw_cache_payload = None
        _lw_cache_ts = 0.0
        _lw_facet_tags = None
        _active_dataset_id_cached = None
        _active_dataset_id_ts = 0.0

from src.utils.cache_manager import register_invalidation_callback
register_invalidation_callback(invalidate_stone_cache)


class SearchFilters(BaseModel):
    in_stock_only: bool = Field(
        default=False,
        description=(
            "NOT ENFORCED: field is accepted for API compatibility but is never evaluated "
            "in the filter chain (_apply_hard_filters_v2). Sending in_stock_only=true has "
            "no effect on results. Status: accepted-not-enforced."
        ),
    )
    stone_types: List[str] = Field(default_factory=list)
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
            "field is a one-way mirror of `pattern_families` kept only so the "
            "ranker (`_pattern_match_score`) keeps producing scores."
        ),
    )
    accent_colors: List[str] = Field(default_factory=list)
    min_confidence: float = 0.0
    location: Optional[str] = None
    thickness: Optional[str] = None
    finish: Optional[str] = None
    cloudiness_pref: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Pattern feel: 0=Linear, 1=Cloudy. Only applies ranking bias when set.")
    cloudiness_strict: bool = Field(default=False, description="Enable strict cloudiness filter (hard cutoff, not just ranking)")
    cloudiness_min: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum cloudiness score (strict mode only, e.g. 0.65 for cloudy)")
    cloudiness_max: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum cloudiness score (strict mode only, e.g. 0.35 for linear)")
    dominant_tones: List[str] = Field(default_factory=list, description="Undertone (LAB): neutral, green, red, warm, cool")
    dominant_tone_strict: bool = Field(default=False, description="Enable strict undertone filter (hard cutoff instead of ranking boost)")
    dominant_hues: List[str] = Field(default_factory=list, description="Main Hue (visual): neutral, green, red, orange_brown, yellow_beige, blue_grey")
    dominant_hue_strict: bool = Field(default=False, description="Enable strict hue filter (hard cutoff instead of ranking boost)")
    strict_exclude: bool = Field(default=False, description="GLOBAL strict mode: When ON, all active filters become hard WHERE conditions (AND logic). When OFF, only base_color_mode=strict excludes; others rank only.")
    pattern_families: List[str] = Field(default_factory=list, description="Pattern family filter: uniform, cloudy, linear, webbed, bold_veined, breccia. OR within list, AND with other filters. Empty = include unlabeled.")

    @model_validator(mode="after")
    def _consolidate_pattern_filter_fields(self) -> "SearchFilters":
        """Pattern P4 (J.6) — single source of truth for the pattern filter.

        After validation, `pattern_families` is the authoritative list and
        `patterns` is a one-way mirror kept only to feed
        `_pattern_match_score`. Inputs are alias-normalized through
        `tag_storage.PATTERN_FAMILY_ALIASES`. The unlabeled sentinel
        ("unlabeled") is preserved unchanged.

        Cases:
          - both empty: no-op.
          - only `patterns` populated: fold normalized values into
            `pattern_families`, then mirror back into `patterns`.
          - only `pattern_families` populated: normalize and mirror into
            `patterns`.
          - both populated and identical (after normalization): normalize
            and keep mirrored.
          - both populated and different: `pattern_families` wins; emit a
            single warning identifying the dropped `patterns` payload so
            future callers can be migrated.
        """
        _consolidate_pattern_filter(self, _pf_logger=logger)
        return self


def _normalize_pattern_filter_list(values: List[str]) -> List[str]:
    """Lower/strip + alias-normalize while preserving the unlabeled sentinel."""
    try:
        aliases = taxonomy_client.get_pattern_family_aliases()
    except Exception:
        aliases = {
            "brecciated": "breccia",
            "onyx-like": "linear",
            "dramatic-mix": "breccia",
            "web": "webbed",
            "bold veined": "bold_veined",
            "bold-veined": "bold_veined",
        }
    out: List[str] = []
    for v in values or []:
        if not isinstance(v, str):
            continue
        low = v.lower().strip()
        if not low:
            continue
        if low == "unlabeled":  # PATTERN_FAMILY_UNLABELED_SENTINEL (defined later in module)
            out.append(low)
        else:
            out.append(aliases.get(low, low))
    return out



def _consolidate_pattern_filter(filters_obj: Any, _pf_logger: Optional[logging.Logger] = None) -> None:
    """In-place P4 consolidation for any SearchFilters-shaped object.

    Reads/writes only `.patterns` and `.pattern_families`.
    """
    log_ = _pf_logger if _pf_logger is not None else logging.getLogger(__name__)
    raw_patterns = list(getattr(filters_obj, "patterns", []) or [])
    raw_families = list(getattr(filters_obj, "pattern_families", []) or [])
    norm_patterns = _normalize_pattern_filter_list(raw_patterns)
    norm_families = _normalize_pattern_filter_list(raw_families)

    if not norm_patterns and not norm_families:
        filters_obj.patterns = []
        filters_obj.pattern_families = []
        return

    if norm_families and norm_patterns:
        # Compare as multisets so duplicate/order drift (e.g.
        # patterns=['linear','linear'] vs pattern_families=['linear'])
        # is still surfaced as a migration warning. Identical multisets
        # (any order) stay silent.
        if Counter(norm_patterns) != Counter(norm_families):
            log_.warning(
                "pattern_p4_dual_filter_received: pattern_families wins over "
                "deprecated `patterns` field. dropped_patterns=%r kept_pattern_families=%r",
                raw_patterns,
                raw_families,
            )

    if norm_families:
        canonical = norm_families
    else:
        canonical = norm_patterns

    filters_obj.pattern_families = canonical
    filters_obj.patterns = list(canonical)


class SearchRequest(BaseModel):
    q: str = ""
    filters: SearchFilters = Field(default_factory=SearchFilters)
    sort: str = "best_match"
    limit: int = 60
    offset: int = 0
    debug_scores: bool = Field(default=False, description="Include detailed score breakdown in results")
    admin_user_id: Optional[str] = Field(default=None, description="Admin user ID for suppression filtering (omit for public users)")
    include_suppressed: bool = Field(default=False, description="When True, skip admin suppression filtering and show all matching stones (admin mode only)")
    green_hunt: bool = Field(default=False, description="Enable Missed Greens Hunt mode - find SMART failures with green signals")
    green_hunt_unlabeled_only: bool = Field(default=True, description="Exclude stones already labeled by this admin")
    green_hunt_include_suppressed: bool = Field(default=True, description="Include suppressed stones in hunt (default True to find all candidates)")


class AccentColor(BaseModel):
    color: str
    strength: float


class StoneResult(BaseModel):
    company_stone_id: str
    batch_id: Optional[str] = None
    stone_name: Optional[str] = None
    vendor_name: Optional[str] = None
    thumbnail_url: Optional[str] = None
    base_color: Optional[str] = None
    base_color_confidence: float = 0.0
    vein_intensity: Optional[str] = None
    pattern_family: Optional[str] = None
    accent_colors: List[AccentColor] = Field(default_factory=list)
    visual_busyness: float = 0.0
    drama_score: float = Field(
        default=0.0,
        description=(
            "NOT RANKING-ACTIVE: drama_score is computed and returned for informational "
            "purposes but its weight in _compute_search_score_v2() is 0. It does not "
            "affect result ordering. Status: serialized-not-ranked."
        ),
    )
    dominant_tone: Optional[str] = None
    dominant_tone_confidence: float = 0.0
    dominant_hue: Optional[str] = None
    dominant_hue_confidence: float = 0.0
    has_manual_override: bool = False
    similarity_score: Optional[float] = None
    final_score: float = 0.0
    tags: Dict[str, Any] = Field(default_factory=dict)
    # Task #390 — Smart-mode tier section assignment.  Present only when
    # the activation condition (Smart mode AND base_colors AND
    # pattern_families) is met; null/absent otherwise so existing
    # clients see no behaviour change.
    #   1 = colour-match + pattern-match
    #   2 = colour-match only
    #   3 = pattern-match only
    match_tier: Optional[int] = None


class SearchResponse(BaseModel):
    results: List[StoneResult]
    total: int
    query: str
    image_used: bool = False
    latency_ms: float = 0.0
    search_request_id: str = Field(
        default="",
        description=(
            "Server-issued UUID for this search request. "
            "Include in feedback submissions as search_request_id to allow "
            "server-side validation of result_position."
        ),
    )
    debug_filters: Optional[Dict[str, Any]] = None
    pattern_family_facets: Optional[Dict[str, int]] = None
    pattern_family_unlabeled: int = 0
    base_color_facets: Optional[Dict[str, int]] = None
    # Task #390 — Generic per-response metadata bag.  Currently used for
    # ``smart_tier_counts = {tier_1_both, tier_2_colour_only,
    # tier_3_pattern_only}`` when Smart-mode sectioning is active; null
    # otherwise so existing clients see no behaviour change.
    meta: Optional[Dict[str, Any]] = None


SEARCH_LOGS: List[Dict] = []

# Initialise durable log DB on module load.
_slog.init_db()

def _log_search(
    query: str,
    filters: dict,
    image_used: bool,
    result_count: int,
    latency_ms: float,
    *,
    mode: str = "text",
    dataset_id: Optional[str] = None,
    query_offset: Optional[int] = None,
    query_limit: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Log search for analytics (no image bytes).

    Returns a (search_request_id, timestamp) tuple so callers can:
      - include search_request_id in the search response
      - pass the same timestamp to log_results() for consistent linkage
      - allow feedback handlers to validate result_position against the durable log

    query_offset and query_limit are the pagination parameters from the request;
    they are stored durably so any page can be reconstructed from the log alone.
    """
    search_request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    log_entry = {
        "search_request_id": search_request_id,
        "timestamp": timestamp,
        "query": query,
        "filters": filters,
        "mode": mode,
        "image_used": image_used,
        "result_count": result_count,
        "latency_ms": round(latency_ms, 2),
        "dataset_id": dataset_id,
    }
    SEARCH_LOGS.append(log_entry)
    if len(SEARCH_LOGS) > 1000:
        SEARCH_LOGS.pop(0)

    _slog.log_search(
        search_request_id=search_request_id,
        timestamp=timestamp,
        query=query,
        filters=filters,
        mode=mode,
        result_count=result_count,
        latency_ms=latency_ms,
        image_used=image_used,
        dataset_id=dataset_id,
        query_offset=query_offset,
        query_limit=query_limit,
    )

    logger.info(
        "[Search] req=%s q='%s' mode=%s image=%s results=%d latency=%.1fms dataset=%s offset=%s limit=%s",
        search_request_id, query, mode, image_used, result_count, latency_ms, dataset_id,
        query_offset, query_limit,
    )
    return search_request_id, timestamp


_CACHE_REBUILD_TIMEOUT_S: float = 5.0  # Max seconds to wait for another thread's rebuild


def _load_all_stone_data_batched() -> Tuple[str, List[Dict]]:
    """Load all stone data with full tag hydration.

    Cache strategy:
    - Cache HIT (TTL=120s): returns fully-hydrated stones — zero SQL.
    - Cache MISS with rebuild already in progress: wait up to _CACHE_REBUILD_TIMEOUT_S
      for the rebuilding thread to finish, then return its result.  On timeout,
      if stale data is available it is returned with a warning; otherwise a fresh
      rebuild is triggered by this thread.
    - Cache MISS, no rebuild in progress: this thread performs the rebuild.

    Rebuild coordination uses _stone_cache_rebuild_event (threading.Event):
      - Set by the rebuilding thread when it holds the "rebuilding" slot.
      - Cleared (and waiters released) when the rebuild completes or fails.

    Returns:
        (dataset_id, list of fully-hydrated stone dicts with display_tags)
    """
    global _last_db_substage_timing
    global _stone_cache_dataset_id, _stone_cache_payload, _stone_cache_ts
    global _stone_cache_hits, _stone_cache_misses
    global _stone_cache_rebuilding, _stone_cache_rebuild_event

    now = time.time()

    # --- Phase 1: check for a valid cached payload ---
    t_cache_check = time.perf_counter()
    with _stone_cache_lock:
        if (
            _stone_cache_payload is not None
            and _stone_cache_dataset_id is not None
            and (now - _stone_cache_ts) < _STONE_CACHE_TTL
        ):
            _stone_cache_hits += 1
            cache_check_ms = (time.perf_counter() - t_cache_check) * 1000
            _last_db_substage_timing = {
                "cache_check_ms": round(cache_check_ms, 2),
                "sql_dataset_id_ms": 0.0,
                "phase_a_lightweight_ms": 0.0,
                "phase_a_total_ms": 0.0,
                "phase_b_total_ms": 0.0,
                "total_db_ms": round(cache_check_ms, 2),
                "total_stones": len(_stone_cache_payload),
                "mode": "cache_hit",
            }
            logger.debug(
                "stone_cache HIT (full hydration) dataset=%s age=%.1fs cache_check=%.2fms hits=%d",
                _stone_cache_dataset_id, now - _stone_cache_ts, cache_check_ms, _stone_cache_hits,
            )
            return _stone_cache_dataset_id, _stone_cache_payload

        # --- Cache miss: is another thread already rebuilding? ---
        already_rebuilding = _stone_cache_rebuilding
        rebuild_event: Optional[threading.Event] = _stone_cache_rebuild_event
        if not already_rebuilding:
            # Claim the rebuild slot before releasing the lock.
            _stone_cache_rebuilding = True
            _stone_cache_rebuild_event = threading.Event()
            rebuild_event = _stone_cache_rebuild_event

    cache_miss_check_ms = (time.perf_counter() - t_cache_check) * 1000

    if already_rebuilding and rebuild_event is not None:
        # --- Wait for the rebuilding thread to finish ---
        logger.debug(
            "stone_cache MISS — rebuild already in progress, waiting up to %.1fs",
            _CACHE_REBUILD_TIMEOUT_S,
        )
        finished = rebuild_event.wait(timeout=_CACHE_REBUILD_TIMEOUT_S)
        with _stone_cache_lock:
            if finished and _stone_cache_payload is not None:
                # Rebuilding thread succeeded; return its result.
                logger.debug("stone_cache: waiter unblocked, using rebuilt payload")
                return _stone_cache_dataset_id, _stone_cache_payload
            elif _stone_cache_payload is not None:
                # Timeout but stale data is still present — return stale rather than rebuild.
                logger.warning(
                    "stone_cache: rebuild wait timed out after %.1fs — returning stale payload "
                    "(age=%.1fs, stones=%d). Chosen path: STALE_ON_TIMEOUT.",
                    _CACHE_REBUILD_TIMEOUT_S,
                    time.time() - _stone_cache_ts,
                    len(_stone_cache_payload),
                )
                return _stone_cache_dataset_id, _stone_cache_payload
            else:
                # Timeout and no stale data — fall through to trigger our own rebuild.
                logger.warning(
                    "stone_cache: rebuild wait timed out after %.1fs and no stale data available "
                    "— triggering fresh rebuild on this thread.",
                    _CACHE_REBUILD_TIMEOUT_S,
                )
                _stone_cache_rebuilding = True
                new_event = threading.Event()
                _stone_cache_rebuild_event = new_event
                # Critical: rebind local so finally{} signals the new event, not the old one.
                rebuild_event = new_event

    # --- This thread owns the rebuild ---
    try:
        dataset_id, candidates, phase_a_timing = _get_lightweight_candidates()
        substage = _hydrate_stones_for_page_timed(candidates)

        total_db_ms = (
            cache_miss_check_ms
            + phase_a_timing.get("phase_a_total_ms", 0.0)
            + substage.get("phase_b_total_ms", 0.0)
        )
        combined_timing = {
            **phase_a_timing,
            **substage,
            "cache_check_ms": round(cache_miss_check_ms, 2),
            "total_db_ms": round(total_db_ms, 2),
        }

        with _stone_cache_lock:
            _stone_cache_misses += 1
            _stone_cache_dataset_id = dataset_id
            _stone_cache_payload = candidates
            _stone_cache_ts = time.time()
            _last_db_substage_timing = combined_timing

        logger.info(
            "stone_cache MISS (full) dataset=%s stones=%d "
            "phase_a=%.2f phase_b=%.2f total=%.2fms hits=%d misses=%d",
            dataset_id, len(candidates),
            phase_a_timing.get("phase_a_total_ms", 0),
            substage.get("phase_b_total_ms", 0),
            total_db_ms, _stone_cache_hits, _stone_cache_misses,
        )
        return dataset_id, candidates
    finally:
        # Always release the rebuild slot and signal any waiters.
        with _stone_cache_lock:
            _stone_cache_rebuilding = False
        if rebuild_event is not None:
            rebuild_event.set()


def _hydrate_stones_for_page(stones_page: List[Dict]) -> None:
    """
    Candidate-first Phase B: hydrate full tag data for only the final-page stones.

    Fills in display_tags for a pre-filtered, sorted, paginated subset of stones
    using the same 4 tag queries as the full loader, but restricted to the
    company_stone_ids actually on the page.  Used when the stone cache returns
    lightweight candidate dicts (no display_tags).

    This function mutates the passed stone dicts in place.
    """
    if not stones_page:
        return

    page_cids = list({s["company_stone_id"] for s in stones_page})

    JSON_TAGS = {"dominant_tone_debug", "dominant_hue_debug", "accent_colors", "hue_ratios"}

    def _parse_tag_value(tag_name: str, tag_value: str) -> Any:
        if tag_name in JSON_TAGS and tag_value:
            try:
                if tag_value.startswith("{") or tag_value.startswith("["):
                    return json.loads(tag_value)
            except (json.JSONDecodeError, TypeError):
                pass
        return tag_value

    company_to_stones: Dict[str, List[Dict]] = {}
    for s in stones_page:
        cid = s["company_stone_id"]
        if cid not in company_to_stones:
            company_to_stones[cid] = []
        company_to_stones[cid].append(s)

    res = taxonomy_client.get_page_hydration_data(page_cids)
    ai_rows = res.get("ai_rows", [])
    manual_rows = res.get("manual_rows", [])
    agg_rows = res.get("agg_rows", [])
    ovr_rows = res.get("ovr_rows", [])

    _ai_by_company: Dict[str, list] = {}
    for row in ai_rows:
        cid = row["company_stone_id"]
        if cid not in _ai_by_company:
            _ai_by_company[cid] = []
        _ai_by_company[cid].append(row)

    _lbds_cfg = _lbds_get_config()
    for cid, c_rows in _ai_by_company.items():
        for s in company_to_stones.get(cid, []):
            tags_dict = s["display_tags"]
            _stone_vendor = s.get("vendor_name")
            for row in c_rows:
                if _lbds_cfg.is_active and _lbds_should_suppress(
                    vendor_name=_stone_vendor,
                    tag_name=row["tag_name"],
                    confidence=row["confidence"],
                    config=_lbds_cfg,
                ):
                    continue
                tags_dict[row["tag_name"]] = {
                    "value": _parse_tag_value(row["tag_name"], row["tag_value"]),
                    "confidence": row["confidence"],
                    "source": "ai",
                }

    _manual_by_company: Dict[str, list] = {}
    manual_company_ids: set = set()
    for row in manual_rows:
        cid = row["company_stone_id"]
        manual_company_ids.add(cid)
        if cid not in _manual_by_company:
            _manual_by_company[cid] = []
        _manual_by_company[cid].append(row)
    for cid, c_rows in _manual_by_company.items():
        for s in company_to_stones.get(cid, []):
            tags_dict = s["display_tags"]
            for row in c_rows:
                tags_dict[row["tag_name"]] = {
                    "value": _parse_tag_value(row["tag_name"], row["tag_value"]),
                    "confidence": row["confidence"],
                    "source": "manual",
                }
    for cid in manual_company_ids:
        for s in company_to_stones.get(cid, []):
            s["has_manual_override"] = True

    for row in agg_rows:
        cid = row["company_stone_id"]
        field = row["field_name"]
        agg_val = row["aggregated_value"]
        if field == "pattern_family":
            if agg_val is None or (
                isinstance(agg_val, str) and not agg_val.strip()
            ):
                continue
            for s in company_to_stones.get(cid, []):
                existing = s["display_tags"].get(field)
                if existing and existing.get("source") == "manual":
                    continue
                s["display_tags"][field] = {
                    "value": agg_val,
                    "confidence": 1.0,
                    "source": "aggregated",
                }
        elif agg_val is not None:
            for s in company_to_stones.get(cid, []):
                s["display_tags"][field] = {
                    "value": agg_val,
                    "confidence": 1.0,
                    "source": "aggregated",
                }

    for row in ovr_rows:
        cid = row["company_stone_id"]
        field = row["field_name"]
        for s in company_to_stones.get(cid, []):
            s["display_tags"][field] = {
                "value": row["override_value"],
                "confidence": 1.0,
                "source": "stock_override",
            }

    for s in stones_page:
        tags = s.get("display_tags", {})
        nt_val = tags.get("network_thickness", {}).get("value")
        bav_val = tags.get("has_bold_accent_vein", {}).get("value")
        is_bold = (nt_val == "bold") or (str(bav_val).lower() == "true" if bav_val else False)
        s["_bold_veined"] = is_bold


def _hydrate_facet_tags_for_candidates(candidates: List[Dict]) -> None:
    """
    Partial hydration for the deferred candidate-first path.

    Loads pattern_family and base_color using the same 4-source precedence order
    as full hydration (AI tags -> manual tags -> aggregated -> overrides), but
    restricted to only those 2 tag names.  This ensures exact facet parity with
    the full-hydration path.

    Field-specific precedence (Task #293):
      - pattern_family: override > manual > aggregated > ai. The aggregated
        write loop skips `pattern_family` rows that would overwrite an
        existing manual stock_tags entry, and additionally rejects None / ''
        / whitespace-only aggregated values so they cannot clobber an AI
        fallback.
      - All other fields (base_color, etc.): override > aggregated > manual
        > ai (aggregated continues to overwrite manual, by design).

    Full display_tag hydration is deferred to _hydrate_stones_for_page() on the
    final page; this function only provides the fields needed by the facet loop.

    Mutates the passed stone dicts in place (display_tags).  Called in text-only
    (deferred) path before the facet loop so facets are accurate.

    Cache: results are stored in _lw_facet_tags (module-level dict keyed by
    company_stone_id) and reused on subsequent requests within the same cache
    cycle.  Cleared by invalidate_stone_cache().  On cache hit, zero SQL queries.
    """
    global _lw_facet_tags

    if not candidates:
        return

    # Cache hit: apply pre-computed facet tags without running any SQL.
    if _lw_facet_tags is not None:
        for s in candidates:
            cached = _lw_facet_tags.get(s["company_stone_id"])
            if cached:
                s["display_tags"].update(cached)
        return

    all_cids = list({s["company_stone_id"] for s in candidates})
    placeholders = ",".join("?" for _ in all_cids)

    company_to_stones: Dict[str, List[Dict]] = {}
    for s in candidates:
        cid = s["company_stone_id"]
        if cid not in company_to_stones:
            company_to_stones[cid] = []
        company_to_stones[cid].append(s)

    _FACET_TAGS = ('pattern_family', 'base_color', 'bg_color', 'main_color')
    _FACET_AGG_FIELDS = ('pattern_family', 'base_color')
    _FACET_OVR_FIELDS = ('pattern_family', 'base_color')

    tag_in_clause = ",".join(f"'{t}'" for t in _FACET_TAGS)
    agg_field_placeholders = ",".join(f"'{f}'" for f in _FACET_AGG_FIELDS)
    ovr_field_placeholders = ",".join(f"'{f}'" for f in _FACET_OVR_FIELDS)

    facet_data = taxonomy_client.get_batch_facet_data(
        company_stone_ids=all_cids,
        tag_names=list(_FACET_TAGS),
        agg_fields=list(_FACET_AGG_FIELDS),
        ovr_fields=list(_FACET_OVR_FIELDS)
    )
    ai_rows = facet_data.get("ai_rows", [])
    manual_rows = facet_data.get("manual_rows", [])
    agg_rows = facet_data.get("agg_rows", [])
    ovr_rows = facet_data.get("ovr_rows", [])

    for row in ai_rows:
        cid = row["company_stone_id"]
        tag = row["tag_name"]
        for s in company_to_stones.get(cid, []):
            s["display_tags"][tag] = {
                "value": row["tag_value"],
                "confidence": row["confidence"],
                "source": "ai",
            }

    for row in manual_rows:
        cid = row["company_stone_id"]
        tag = row["tag_name"]
        for s in company_to_stones.get(cid, []):
            s["display_tags"][tag] = {
                "value": row["tag_value"],
                "confidence": row["confidence"],
                "source": "manual",
            }

    for row in agg_rows:
        cid = row["company_stone_id"]
        field = row["field_name"]
        agg_val = row["aggregated_value"]
        # Task #293 — pattern_family precedence fix: scoped to pattern_family.
        #   1. Empty-string guard: reject None / '' / whitespace-only so an
        #      empty aggregated row cannot clobber a valid AI value.
        #   2. Manual precedence: do not overwrite a manual stock_tags row
        #      so the effective order becomes
        #      override > manual > aggregated > ai (matches the documented
        #      contract and tag_router._resolve_pattern_source_tier).
        # Other fields keep the existing behaviour by design — manual
        # stock_tags stays subordinate to aggregated for base_color etc.
        if field == "pattern_family":
            if agg_val is None or (
                isinstance(agg_val, str) and not agg_val.strip()
            ):
                continue
            for s in company_to_stones.get(cid, []):
                existing = s["display_tags"].get(field)
                if existing and existing.get("source") == "manual":
                    continue
                s["display_tags"][field] = {
                    "value": agg_val,
                    "confidence": 1.0,
                    "source": "aggregated",
                }
        elif agg_val is not None:
            for s in company_to_stones.get(cid, []):
                s["display_tags"][field] = {
                    "value": agg_val,
                    "confidence": 1.0,
                    "source": "aggregated",
                }

    for row in ovr_rows:
        cid = row["company_stone_id"]
        field = row["field_name"]
        for s in company_to_stones.get(cid, []):
            s["display_tags"][field] = {
                "value": row["override_value"],
                "confidence": 1.0,
                "source": "stock_override",
            }

    # Cache the merged facet tags for subsequent requests in this cache cycle.
    # company_to_stones has one entry per cid (lw candidates are deduplicated);
    # snapshot display_tags now that all 4 sources have been applied.
    _lw_facet_tags = {
        cid: dict(stones[0]["display_tags"])
        for cid, stones in company_to_stones.items()
        if stones[0]["display_tags"]
    }


def _hydrate_stones_for_page_timed(stones_page: List[Dict]) -> Dict[str, Any]:
    """
    Phase B with named sub-stage timers.

    Same semantics as _hydrate_stones_for_page() but returns a dict of precise
    per-query, per-merge timing and row-count metrics for the DB_PATH_AUDIT and
    API debug payload:

    Top-level aggregate keys (aligned with spec):
      object_materialize_ms, sql_execute_ms, fetchall_ms, tag_merge_ms
      bold_compute_ms, phase_b_total_ms
      stones_hydrated, unique_cids_hydrated
      rows_ai, rows_manual, rows_agg, rows_ovr

    Per-query breakdown keys:
      sql_execute_ai_ms, fetchall_ai_ms, tag_merge_ai_ms
      sql_execute_manual_ms, fetchall_manual_ms, tag_merge_manual_ms
      sql_execute_agg_ms, fetchall_agg_ms, tag_merge_agg_ms
      sql_execute_ovr_ms, fetchall_ovr_ms, tag_merge_ovr_ms
    """
    t_b_start = time.perf_counter()

    if not stones_page:
        return {"phase_b_total_ms": 0.0, "stones_hydrated": 0, "unique_cids_hydrated": 0}

    page_cids = list({s["company_stone_id"] for s in stones_page})
    placeholders = ",".join("?" for _ in page_cids)

    JSON_TAGS = {"dominant_tone_debug", "dominant_hue_debug", "accent_colors", "hue_ratios"}

    def _parse_tag_value(tag_name: str, tag_value: str) -> Any:
        if tag_name in JSON_TAGS and tag_value:
            try:
                if tag_value.startswith("{") or tag_value.startswith("["):
                    return json.loads(tag_value)
            except (json.JSONDecodeError, TypeError):
                pass
        return tag_value

    t_obj_mat = time.perf_counter()
    company_to_stones: Dict[str, List[Dict]] = {}
    for s in stones_page:
        cid = s["company_stone_id"]
        if cid not in company_to_stones:
            company_to_stones[cid] = []
        company_to_stones[cid].append(s)
    t_obj_mat_ms = (time.perf_counter() - t_obj_mat) * 1000

    t_exec_all = time.perf_counter()
    res = taxonomy_client.get_page_hydration_data(page_cids)
    t_exec_all_ms = (time.perf_counter() - t_exec_all) * 1000

    ai_rows = res.get("ai_rows", [])
    manual_rows = res.get("manual_rows", [])
    agg_rows = res.get("agg_rows", [])
    ovr_rows = res.get("ovr_rows", [])

    t_merge_ai = time.perf_counter()
    _ai_by_company: Dict[str, list] = {}
    for row in ai_rows:
        cid = row["company_stone_id"]
        if cid not in _ai_by_company:
            _ai_by_company[cid] = []
        _ai_by_company[cid].append(row)
    # Task #292 — LOW-band display suppression experiment.
    # Resolved once per page; flag-off (default) makes the per-row
    # check a single boolean read.
    _lbds_cfg = _lbds_get_config()
    for cid, c_rows in _ai_by_company.items():
        for s in company_to_stones.get(cid, []):
            tags_dict = s["display_tags"]
            _stone_vendor = s.get("vendor_name")
            for row in c_rows:
                if _lbds_cfg.is_active and _lbds_should_suppress(
                    vendor_name=_stone_vendor,
                    tag_name=row["tag_name"],
                    confidence=row["confidence"],
                    config=_lbds_cfg,
                ):
                    continue
                tags_dict[row["tag_name"]] = {
                    "value": _parse_tag_value(row["tag_name"], row["tag_value"]),
                    "confidence": row["confidence"],
                    "source": "ai",
                }
    t_merge_ai_ms = (time.perf_counter() - t_merge_ai) * 1000

    t_merge_manual = time.perf_counter()
    _manual_by_company: Dict[str, list] = {}
    manual_company_ids: set = set()
    for row in manual_rows:
        cid = row["company_stone_id"]
        manual_company_ids.add(cid)
        if cid not in _manual_by_company:
            _manual_by_company[cid] = []
        _manual_by_company[cid].append(row)
    for cid, c_rows in _manual_by_company.items():
        for s in company_to_stones.get(cid, []):
            tags_dict = s["display_tags"]
            for row in c_rows:
                tags_dict[row["tag_name"]] = {
                    "value": _parse_tag_value(row["tag_name"], row["tag_value"]),
                    "confidence": row["confidence"],
                    "source": "manual",
                }
    for cid in manual_company_ids:
        for s in company_to_stones.get(cid, []):
            s["has_manual_override"] = True
    t_merge_manual_ms = (time.perf_counter() - t_merge_manual) * 1000

    t_merge_agg = time.perf_counter()
    for row in agg_rows:
        cid = row["company_stone_id"]
        field = row["field_name"]
        if row["aggregated_value"] is not None:
            for s in company_to_stones.get(cid, []):
                s["display_tags"][field] = {
                    "value": row["aggregated_value"],
                    "confidence": 1.0,
                    "source": "aggregated",
                }
    t_merge_agg_ms = (time.perf_counter() - t_merge_agg) * 1000

    t_merge_ovr = time.perf_counter()
    for row in ovr_rows:
        cid = row["company_stone_id"]
        field = row["field_name"]
        for s in company_to_stones.get(cid, []):
            s["display_tags"][field] = {
                "value": row["override_value"],
                "confidence": 1.0,
                "source": "stock_override",
            }
    t_merge_ovr_ms = (time.perf_counter() - t_merge_ovr) * 1000

    t_bold = time.perf_counter()
    for s in stones_page:
        tags = s.get("display_tags", {})
        nt_val = tags.get("network_thickness", {}).get("value")
        bav_val = tags.get("has_bold_accent_vein", {}).get("value")
        is_bold = (nt_val == "bold") or (str(bav_val).lower() == "true" if bav_val else False)
        s["_bold_veined"] = is_bold
    t_bold_ms = (time.perf_counter() - t_bold) * 1000

    phase_b_total_ms = (time.perf_counter() - t_b_start) * 1000

    sql_execute_ms = round(t_exec_all_ms, 2)
    fetchall_ms = 0.0
    tag_merge_ms = round(t_merge_ai_ms + t_merge_manual_ms + t_merge_agg_ms + t_merge_ovr_ms, 2)

    return {
        "object_materialize_ms": round(t_obj_mat_ms, 2),
        "sql_execute_ms": sql_execute_ms,
        "fetchall_ms": fetchall_ms,
        "tag_merge_ms": tag_merge_ms,
        "bold_compute_ms": round(t_bold_ms, 2),
        "phase_b_total_ms": round(phase_b_total_ms, 2),
        "stones_hydrated": len(stones_page),
        "unique_cids_hydrated": len(page_cids),
        "rows_ai": len(ai_rows),
        "rows_manual": len(manual_rows),
        "rows_agg": len(agg_rows),
        "rows_ovr": len(ovr_rows),
        "sql_execute_ai_ms": round(t_exec_all_ms, 2),
        "fetchall_ai_ms": 0.0,
        "tag_merge_ai_ms": round(t_merge_ai_ms, 2),
        "sql_execute_manual_ms": 0.0,
        "fetchall_manual_ms": 0.0,
        "tag_merge_manual_ms": round(t_merge_manual_ms, 2),
        "sql_execute_agg_ms": 0.0,
        "fetchall_agg_ms": 0.0,
        "tag_merge_agg_ms": round(t_merge_agg_ms, 2),
        "sql_execute_ovr_ms": 0.0,
        "fetchall_ovr_ms": 0.0,
        "tag_merge_ovr_ms": round(t_merge_ovr_ms, 2),
    }


def _load_candidates_lightweight(dataset_id: str) -> List[Dict]:
    """
    Candidate-first Phase A: lightweight SQL returning the minimal fields
    needed for text-based filter/score/sort/pagination — no full tag hydration.

    Returns a list of minimal stone dicts with:
      batch_id, company_stone_id, stone_name, vendor_name, canonical_family,
      thumbnail_url, display_tags={}, has_manual_override (accurate), _bold_veined=False

    has_manual_override is loaded from stock_tags so that manual_boost in
    _compute_search_score_v2 is correct for text-only deferred-path searches.
    All other display_tags remain empty until _hydrate_stones_for_page() is called
    on the final-page subset.
    """
    t0 = time.perf_counter()
    res = ingestion_service.get_all_stone_data(dataset_id)
    rows = res.get("stones", [])
    
    candidates = [
        {
            "batch_id": r["batch_id"],
            "company_stone_id": r["company_stone_id"],
            "stone_name": r["stone_name_raw"] or "",
            "vendor_name": r["vendor_name"] or "",
            "canonical_family": r["canonical_family"] or "",
            "thumbnail_url": r["thumbnail_url"] or "",
            "display_tags": {},
            "has_manual_override": False,
            "_bold_veined": False,
        }
        for r in rows
    ]
    logger.debug(
        "candidate_first Phase A: dataset=%s candidates=%d elapsed=%.1fms",
        dataset_id, len(candidates), (time.perf_counter() - t0) * 1000,
    )
    return candidates


# ---------------------------------------------------------------------------
# Candidate-first architecture: the search route's two-phase data loader.
# ---------------------------------------------------------------------------
# The stone cache stores FULLY HYDRATED candidates (display_tags populated).
# Phase A (lightweight SQL): fast, uses covering index on dataset_images.
#   Returns all candidate dicts with: batch_id, company_stone_id, name, vendor,
#   canonical_family, thumbnail_url. display_tags={}, _bold_veined=False.
# Phase B (tag hydration): hydrates all candidates immediately after Phase A.
#   Records named per-query sub-timers (sql_execute_ms, fetchall_ms, tag_merge_ms,
#   object_materialize_ms, rows_ai, rows_manual, rows_agg, rows_ovr).
#
# Cache hit: returns fully hydrated candidates — zero SQL, correct scoring.
# Cache miss: Phase A + Phase B, then stores fully hydrated candidates.
# _load_all_stone_data_batched() is the only writer to the cache.
# _get_lightweight_candidates() is the Phase A worker (no caching).

def _get_lightweight_candidates() -> Tuple[str, List[Dict], Dict[str, Any]]:
    """
    Phase A entry point: return lightweight candidates from the tier-1 cache
    (or SQL on cache miss).

    Tier-1 cache (_lw_cache_*): stores minimal stone dicts (no display_tags,
    display_tags={}, _bold_veined=False). TTL = 120s. Invalidated by
    invalidate_stone_cache(). Zero SQL on cache hit.

    Cache miss:
      1. Resolve active dataset_id via 5-second mini-cache or SQL.
      2. Run _load_candidates_lightweight() via covering index (~17ms).
      3. Store in _lw_cache_*.

    Returns:
        (dataset_id, lightweight stone dicts, phase_a_timing_dict)

    phase_a_timing_dict keys (consumed by callers for sub-stage reporting):
        cache_check_ms, sql_dataset_id_ms, phase_a_lightweight_ms,
        phase_a_total_ms, total_stones, mode
    """
    global _active_dataset_id_cached, _active_dataset_id_ts
    global _lw_cache_dataset_id, _lw_cache_payload, _lw_cache_ts
    global _lw_cache_hits, _lw_cache_misses

    t_cache_check = time.perf_counter()
    now = time.time()

    with _stone_cache_lock:
        if (
            _lw_cache_payload is not None
            and _lw_cache_dataset_id is not None
            and (now - _lw_cache_ts) < _STONE_CACHE_TTL
        ):
            _lw_cache_hits += 1
            cache_check_ms = (time.perf_counter() - t_cache_check) * 1000
            timing = {
                "cache_check_ms": round(cache_check_ms, 2),
                "sql_dataset_id_ms": 0.0,
                "phase_a_lightweight_ms": 0.0,
                "phase_a_total_ms": 0.0,
                "total_stones": len(_lw_cache_payload),
                "mode": "lw_cache_hit",
            }
            logger.debug(
                "lw_cache HIT dataset=%s age=%.1fs cache_check=%.2fms hits=%d",
                _lw_cache_dataset_id, now - _lw_cache_ts, cache_check_ms, _lw_cache_hits,
            )
            return _lw_cache_dataset_id, _lw_cache_payload, timing

    cache_miss_check_ms = (time.perf_counter() - t_cache_check) * 1000

    t_ds_id = time.perf_counter()
    with _stone_cache_lock:
        if (
            _active_dataset_id_cached is not None
            and (now - _active_dataset_id_ts) < _DATASET_ID_CACHE_TTL
        ):
            dataset_id = _active_dataset_id_cached
            t_sql_dataset_id_ms = 0.0
        else:
            t_sql = time.perf_counter()
            dataset_id = ingestion_service.get_active_dataset_id()
            t_sql_dataset_id_ms = (time.perf_counter() - t_sql) * 1000
            _active_dataset_id_cached = dataset_id
            _active_dataset_id_ts = now
    t_ds_id_ms = (time.perf_counter() - t_ds_id) * 1000

    if not dataset_id:
        raise HTTPException(
            status_code=400,
            detail="No active dataset selected. Please select a dataset first."
        )

    t_phase_a = time.perf_counter()
    candidates = _load_candidates_lightweight(dataset_id)
    t_phase_a_ms = (time.perf_counter() - t_phase_a) * 1000

    total_phase_a_ms = cache_miss_check_ms + t_ds_id_ms + t_phase_a_ms

    with _stone_cache_lock:
        _lw_cache_misses += 1
        _lw_cache_dataset_id = dataset_id
        _lw_cache_payload = candidates
        _lw_cache_ts = now

    timing: Dict[str, Any] = {
        "cache_check_ms": round(cache_miss_check_ms, 2),
        "sql_dataset_id_ms": round(t_sql_dataset_id_ms, 2),
        "phase_a_lightweight_ms": round(t_phase_a_ms, 2),
        "phase_a_total_ms": round(total_phase_a_ms, 2),
        "total_stones": len(candidates),
        "mode": "candidate_first",
    }

    logger.info(
        "lw_cache MISS dataset=%s stones=%d sql_ds_id=%.2f phase_a=%.2f total=%.2fms hits=%d misses=%d",
        dataset_id, len(candidates),
        t_sql_dataset_id_ms, t_phase_a_ms, total_phase_a_ms,
        _lw_cache_hits, _lw_cache_misses,
    )

    return dataset_id, candidates, timing


def _needs_tags_for_search(filters) -> bool:
    """
    Return True if any active filter or scoring signal requires display_tags.

    Used to decide whether the search route can use the deferred Phase B path
    (text-only deferred hydration) or must use the broad Phase B path (full
    hydration of all candidates before filter/score/sort/paginate).

    The deferred path is only safe when:
    - No filter reads tag data for filtering (hard excludes)
    - No scoring component reads tag data for ranking (soft filters/boosts)

    Tag-dependent HARD filters (change which stones are visible):
    - base_colors (any mode — base_score used for ranking, strict for hard exclude)
    - pattern_families (hard filter on pattern_family tag)
    - patterns (any mode — soft ranking or hard filter)
    - dominant_hues (hard filter + use_rescue path reads hue scores from tags)
    - dominant_tones (any mode — soft boost or hard filter)
    - cloudiness_strict (hard filter on cloudiness_score tag)
    - min_confidence > 0 (reads base_color confidence from tags)
    - strict_exclude (changes all soft filters to hard excludes)

    Tag-dependent SOFT filters (change ranking order):
    - accent_colors (any mode — accent_score reads accent_colors tag)
    - cloudiness_pref (reads cloudiness_score tag for alignment ranking)

    Text-only safe filters (do NOT read display_tags):
    - in_stock_only, stone_types, location, thickness, finish, veining_min/max
      (veining uses visual_busyness tag but score is 0 when no filter active)

    Note on veining: veining_min=0.0 and veining_max=1.0 are the defaults, meaning
    the full range is accepted. _veining_match_score returns 1.0 for all stones with
    the default range, so scoring is not tag-dependent when veining filters are at default.
    """
    if filters.base_colors:
        return True
    if filters.pattern_families:
        return True
    if filters.patterns:
        return True
    if filters.dominant_hues:
        return True
    if filters.dominant_tones:
        return True
    if filters.cloudiness_strict:
        return True
    if filters.cloudiness_pref is not None:
        return True
    if filters.min_confidence > 0:
        return True
    if filters.accent_colors:
        return True
    if filters.strict_exclude:
        return True
    if filters.veining_min > 0.0 or filters.veining_max < 1.0:
        return True
    return False


def _get_display_tag(stone: Dict, tag_name: str, default: Any = None) -> Tuple[Any, float, str]:
    """Get tag value, confidence, and source from display_tags."""
    tags = stone.get("display_tags", {})
    tag_info = tags.get(tag_name)
    if tag_info and isinstance(tag_info, dict):
        return tag_info.get("value", default), tag_info.get("confidence", 0.0), tag_info.get("source", "unknown")
    return default, 0.0, "missing"


def _get_effective_tone_for_filter(stone: Dict) -> Tuple[Optional[str], str]:
    """
    Get effective dominant_tone with manual > ai precedence for filtering.
    
    The display_tags already have manual tags overwriting AI tags during hydration,
    so we just need to read from display_tags and return the source.
    
    Returns:
        (tone_value, source) where source is 'manual', 'ai', or 'missing'
    """
    tone, conf, source = _get_display_tag(stone, "dominant_tone")
    if tone and source in ("manual", "ai"):
        return tone.lower() if isinstance(tone, str) else tone, source
    return None, "missing"


_CHROMATIC_BG_COLORS = {"green", "red", "pink", "rose"}
_CHROMATIC_NORMALIZE = {"rose": "pink", "multicolor": "multi"}

def _normalize_chromatic(color: str) -> str:
    """Normalize chromatic color values (e.g. rose -> pink)."""
    return _CHROMATIC_NORMALIZE.get(color, color)


# Task #442 — Yellow / Gold chip on the buyer-facing colour filter.
# The chip's data-value is "yellow" (the most populous member of the
# GOLD retrieval family in the active cohort: 26 yellow + 16 orange on
# 20260525_202825_clean).  Anywhere `filters.base_colors` is compared
# against a stone's effective base_color, the chip is OR-expanded to the
# full GOLD-family vocabulary defined in
# similar_explorer._COLOR_FAMILY so that stones whose only colour
# signal is `main_color = orange / gold / amber / …` are admitted.
# This is a read-side expansion only — the admin / override taxonomy
# (tag_storage.TAXONOMY_BASE_COLOR_VALUES) is intentionally unchanged;
# correction writes continue to use the canonical set.  Other chip
# values pass through unchanged.
_GOLD_FAMILY_COLORS = frozenset({
    "yellow", "orange", "gold", "amber",
    "copper", "bronze", "ochre", "terra cotta",
})


def _expand_color_filter_set(filter_colors) -> set:
    """Lower-case the filter colours and OR-expand the buyer-facing
    ``yellow`` chip into the full GOLD-family vocabulary (Task #442).
    Returns an empty set when ``filter_colors`` is empty / None."""
    if not filter_colors:
        return set()
    out: set = set()
    for c in filter_colors:
        if not isinstance(c, str):
            continue
        cl = c.lower()
        if cl == "yellow":
            out |= _GOLD_FAMILY_COLORS
        else:
            out.add(cl)
    return out

def _get_effective_base_color_for_filter(stone: Dict) -> Tuple[Optional[str], str]:
    """
    Effective base_color for filtering.

    Resolution chain (high → low):
      1. stock_overrides.base_color           (canonical manual writer)
      2. stock_tags.base_color (source='manual')  (legacy manual tag layer)
      3. stock_aggregated.base_color          (aggregated)
      4. stock_tags_ai.base_color             (AI base_color)        ← DEAD-IN-PROD
      5. stock_tags_ai.knn_base_color         (kNN AI base_color)    ← DEAD-IN-PROD
      6. stock_tags_ai.main_color             (AI main_color)
      +  bg_color rescue (chromatic)          (computed at request time, not persisted)
      +  _normalize_chromatic post-step       (display-only)

    NOTE (Phase 2 audit, 2026-04-18): tiers 4 and 5 currently have ZERO rows in
    production stock_tags_ai. The kNN rescue and AI base_color classifiers exist
    (fingerprints_v3/knn_experiment.py, fingerprints_v3/bg_color_classifier.py)
    but their outputs are not persisted in the current enrichment pipeline. Do NOT
    silently remove these tiers — a future enrichment pass is expected to fill them.
    See COLOR_TRUTH_PHASE1_REPORT.md §1.4 and COLOR_TRUTH_PHASE2_DESIGN.md §0.4.

    Returns:
        (base_color_value, source) where source is 'manual', 'ai', or 'missing'
    """
    bg, bg_conf, bg_source = _get_display_tag(stone, "bg_color")
    if bg and bg_source in ("manual", "ai"):
        bg_lower = bg.lower().strip()
        if bg_lower in _CHROMATIC_BG_COLORS:
            return _normalize_chromatic(bg_lower), bg_source
    
    color, conf, source = _get_display_tag(stone, "base_color")
    if not color:
        color, conf, source = _get_display_tag(stone, "main_color")
    
    if color and source in ("manual", "ai", "color_agent", "aggregated", "stock_override"):
        val = color.lower() if isinstance(color, str) else color
        return _normalize_chromatic(val), source
    return None, "missing"


# Sentinel used by the buyer "Unlabeled" chip to opt unlabeled stones into a
# pattern_families filter (Phase P1, audit §K.5). Lower-cased on receipt.
PATTERN_FAMILY_UNLABELED_SENTINEL = "unlabeled"


def _get_effective_pattern_family_for_filter(stone: Dict) -> Tuple[Optional[str], str]:
    """
    Effective pattern_family for filtering.

    Resolution chain (high → low), mirroring the precedence encoded in
    _hydrate_stones_for_page() / _hydrate_facet_tags_for_candidates() for
    pattern_family (Task #293 fix):

      1. stock_overrides.pattern_family   (canonical manual writer)
      2. stock_tags.pattern_family (source='manual')  (legacy manual layer)
      3. stock_aggregated.pattern_family  (aggregated)
      4. stock_tags_ai.pattern_family     (AI / auto_suggest)

    This matches tag_router._resolve_pattern_source_tier() exactly.
    Empty-string aggregated rows are rejected by the hydrator gate so they
    cannot clobber a valid AI fallback.

    image_observations is NOT a fallback tier — PATTERN_SEARCH_AUDIT.md §A.5
    confirmed it rescues 0 stones (every image-tier stone is already covered
    by a stone-tier source). Do not add it here.

    Returns:
        (pattern_family_value, source) where source is one of
        'stock_override' | 'manual' | 'aggregated' | 'ai' | 'missing'.
        Returns (None, 'missing') when all four tiers are null/empty.
    """
    pf_val, _, pf_source = _get_display_tag(stone, "pattern_family")
    if pf_val and pf_source in ("stock_override", "manual", "aggregated", "ai"):
        val = pf_val.lower().strip() if isinstance(pf_val, str) else pf_val
        if val:
            return val, pf_source
    return None, "missing"


_BOLD_TRUTHY = {"true", "1", "yes", "y", "t"}
_BOLD_WIDEN_SOURCES = ("manual", "ai", "stock_override", "aggregated", "color_agent")


def _stone_matches_bold_veined_widening(stone: Dict) -> bool:
    """
    Task #391 — accept stones into the ``bold_veined`` filter when either
    ``has_bold_accent_vein`` is truthy or ``network_thickness`` is the
    string ``"bold"``.  Tag values can be hydrated as either Python
    booleans or strings (``"true"``, ``"1"``); accept both shapes.  Only
    counts when the tag value comes from a trusted source (the same
    source priority chain the rest of the search uses).
    """
    hbav_val, _, hbav_src = _get_display_tag(stone, "has_bold_accent_vein")
    if hbav_src in _BOLD_WIDEN_SOURCES:
        if hbav_val is True:
            return True
        if isinstance(hbav_val, str) and hbav_val.strip().lower() in _BOLD_TRUTHY:
            return True
        if isinstance(hbav_val, (int, float)) and not isinstance(hbav_val, bool) and hbav_val == 1:
            return True
    nt_val, _, nt_src = _get_display_tag(stone, "network_thickness")
    if (
        nt_src in _BOLD_WIDEN_SOURCES
        and isinstance(nt_val, str)
        and nt_val.strip().lower() == "bold"
    ):
        return True
    return False


def _stone_matches_section_colour(stone: Dict, section_colour_set: set) -> bool:
    """
    Task #390 — mirror the hard-filter colour semantics when classifying
    section tier hits.  Direct match against the override → manual →
    aggregated → AI fallback chain via
    ``_get_effective_base_color_for_filter`` covers most cases; the
    ``brown`` chip additionally admits ``beige`` stones whose
    ``dominant_hue`` is ``brown`` / ``orange`` (matching the smart-mode
    ``BROWN EXPANDED`` branch in ``_apply_hard_filters_v2``), so those
    stones land in tier 1/2 instead of being dropped as colour-misses.
    """
    bc_val, _ = _get_effective_base_color_for_filter(stone)
    bc_lower = bc_val.lower() if isinstance(bc_val, str) else None
    # Task #442 — OR-expand the "yellow" chip into the full GOLD
    # family so colour+pattern sectioning routes orange/gold/amber/…
    # stones into tier 1 with the matching pattern.
    expanded_section = _expand_color_filter_set(section_colour_set)
    if bc_lower and bc_lower in expanded_section:
        return True
    if "brown" in section_colour_set and bc_lower == "beige":
        dom_hue, _, _ = _get_display_tag(stone, "dominant_hue")
        if isinstance(dom_hue, str) and dom_hue.strip().lower() in ("brown", "orange"):
            return True
    return False


def _stone_matches_section_pattern(stone: Dict, section_pattern_set: set) -> bool:
    """
    Task #390 + #391 — pattern_family direct match, plus the bold_veined
    widening signals when ``bold_veined`` is part of the active filter
    set.  Keeps tier classification semantically aligned with the
    relaxed hard filter the section path passes into
    ``_apply_hard_filters_v2``.
    """
    pf_val, _ = _get_effective_pattern_family_for_filter(stone)
    if pf_val and pf_val in section_pattern_set:
        return True
    if "bold_veined" in section_pattern_set and _stone_matches_bold_veined_widening(stone):
        return True
    return False


# -------------------------------------------------------------------------
# Task #404 — Tiered ranking + vendor cap for main search default sort.
# -------------------------------------------------------------------------
# Local chip-level colour adjacency for the main search tiered sort.  This
# is deliberately a small, hand-curated map covering only the buyer-facing
# base_color chip vocabulary.  It is NOT shared with similar_explorer,
# which has its own retrieval-family adjacency model — keeping them
# independent prevents accidental cross-coupling of the two ranking paths.
_MAIN_SEARCH_COLOR_ADJACENCY: Dict[str, set] = {
    "white": {"beige", "grey"},
    "beige": {"white", "brown", "gold"},
    "brown": {"beige", "red", "gold"},
    "grey":  {"white", "black", "blue"},
    "gray":  {"white", "black", "blue"},
    "black": {"grey", "gray"},
    "red":   {"brown", "pink"},
    "pink":  {"red"},
    "green": {"blue"},
    "blue":  {"green", "grey", "gray"},
    "gold":  {"beige", "brown"},
}


def _compute_main_color_tier(stone: Dict, query_colour_set: set) -> int:
    """Task #404 — colour tier for the main-search default sort.

    1 = stone's effective base_color is in the active filter set.
    2 = stone's effective base_color is adjacent to any active filter colour.
    3 = no match or missing colour.

    When ``query_colour_set`` is empty (no colour filter) every stone is
    tier 1 — colour is not a discriminator, the sort degrades cleanly to
    ``(-final_score, -completeness)``.
    """
    if not query_colour_set:
        return 1
    bc_val, _ = _get_effective_base_color_for_filter(stone)
    if not bc_val:
        return 3
    bc = bc_val.lower()
    # Task #442 — expand the "yellow" chip to the full GOLD family
    # so a stone with effective base_color in {orange, gold, amber,
    # …} is tier 1 when the buyer picks Yellow / Gold.
    expanded_query = _expand_color_filter_set(query_colour_set)
    if bc in expanded_query:
        return 1
    adj = _MAIN_SEARCH_COLOR_ADJACENCY.get(bc, set())
    if adj & expanded_query:
        return 2
    return 3


def _compute_main_pattern_tier(stone: Dict, query_pattern_set: set) -> int:
    """Task #404 — pattern tier for the main-search default sort.

    1 = stone's effective pattern_family is in the active filter set (or
        passes the ``bold_veined`` widening when that chip is active).
    2 = stone has a pattern_family but it does not match the filter set.
    3 = stone has no pattern_family at all.

    Empty ``query_pattern_set`` → tier 1 for every stone (pattern is not
    a discriminator).
    """
    if not query_pattern_set:
        return 1
    pf_val, _ = _get_effective_pattern_family_for_filter(stone)
    if pf_val and pf_val in query_pattern_set:
        return 1
    if "bold_veined" in query_pattern_set and _stone_matches_bold_veined_widening(stone):
        return 1
    if pf_val:
        return 2
    return 3


def _compute_completeness_score(stone: Dict) -> int:
    """Task #404 — 0..3 score for stones with a thumbnail, a base_color
    and a pattern_family.  Tiebreaker only — never changes which stones
    are eligible, only the order within a tier-tied group."""
    score = 0
    if stone.get("thumbnail_url"):
        score += 1
    bc_val, _ = _get_effective_base_color_for_filter(stone)
    if bc_val:
        score += 1
    pf_val, _ = _get_effective_pattern_family_for_filter(stone)
    if pf_val:
        score += 1
    return score


# Vendor cap parameters.  See Task #404 brief.
_VENDOR_CAP_MIN_DISTINCT = 5
_VENDOR_CAP_TIGHT = 3
_VENDOR_CAP_RELAXED = 5
_VENDOR_CAP_MIN_DISPLAYED = 8


def _apply_vendor_cap(stones: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
    """Task #404 — same-vendor cap for the main-search display layer.

    Operates on an already-ranked list.  Cards over the per-vendor quota
    are NOT dropped — they are pushed to the end of the list so they
    remain reachable via deeper pagination.  Two-step relaxation:
        1. tight cap = 3/vendor;
        2. if that would leave fewer than 8 cards in-cap, retry at 5/vendor;
        3. if still under 8, skip the cap entirely.

    Activation requires at least 5 distinct vendors in the eligible pool.
    """
    meta: Dict[str, Any] = {
        "vendor_cap_applied": False,
        "vendor_cap_relaxed": False,
        "distinct_vendor_count": 0,
    }
    if not stones:
        return stones, meta

    distinct = {(s.get("vendor_name") or "").strip().lower() for s in stones}
    distinct.discard("")
    meta["distinct_vendor_count"] = len(distinct)
    if len(distinct) < _VENDOR_CAP_MIN_DISTINCT:
        return stones, meta

    def _partition(cap: int) -> Tuple[List[Dict], List[Dict]]:
        counts: Dict[str, int] = {}
        in_cap: List[Dict] = []
        over_cap: List[Dict] = []
        for s in stones:
            v = (s.get("vendor_name") or "").strip().lower()
            counts[v] = counts.get(v, 0) + 1
            (in_cap if counts[v] <= cap else over_cap).append(s)
        return in_cap, over_cap

    in_cap, over_cap = _partition(_VENDOR_CAP_TIGHT)
    if len(in_cap) >= _VENDOR_CAP_MIN_DISPLAYED:
        meta["vendor_cap_applied"] = True
        return in_cap + over_cap, meta

    in_cap_r, over_cap_r = _partition(_VENDOR_CAP_RELAXED)
    if len(in_cap_r) >= _VENDOR_CAP_MIN_DISPLAYED:
        meta["vendor_cap_applied"] = True
        meta["vendor_cap_relaxed"] = True
        return in_cap_r + over_cap_r, meta

    return stones, meta


def _build_search_meta(
    smart_tier_counts: Optional[Dict[str, int]],
    vendor_cap_meta: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Task #404 — merge optional meta blocks into the SearchResponse.meta.

    Returns ``None`` when no block has anything to surface so non-sectioning,
    non-capping responses keep ``meta=null`` (Task #390 contract preserved).
    """
    meta: Dict[str, Any] = {}
    if smart_tier_counts is not None:
        meta["smart_tier_counts"] = smart_tier_counts
    if vendor_cap_meta and (
        vendor_cap_meta.get("vendor_cap_applied")
        or vendor_cap_meta.get("distinct_vendor_count", 0) > 0
    ):
        meta["vendor_cap_applied"] = vendor_cap_meta.get("vendor_cap_applied", False)
        meta["vendor_cap_relaxed"] = vendor_cap_meta.get("vendor_cap_relaxed", False)
        meta["distinct_vendor_count"] = vendor_cap_meta.get("distinct_vendor_count", 0)
    return meta or None


def _get_tag_value(tags: taxonomy_client.MergedStockTags, tag_name: str, default: Any = None) -> Any:
    """Get tag value with fallback."""
    return tags.get_tag(tag_name) if tags else default


def _base_color_match_score(
    stone_base_color: Optional[str],
    stone_confidence: float,
    filter_colors: List[str]
) -> float:
    """
    Compute base color match score.
    """
    if not filter_colors:
        return 0.5
    
    if not stone_base_color:
        return 0.3
    
    stone_base_lower = stone_base_color.lower()
    filter_lower = [c.lower() for c in filter_colors]
    # Task #442 — OR-expand the "yellow" chip into the full GOLD family
    # so the soft colour score treats orange/gold/amber/… as a literal
    # match, not just adjacent.
    expanded_filter_set = _expand_color_filter_set(filter_lower)

    if stone_base_lower in expanded_filter_set:
        return 1.0
    
    adjacent_families = {
        "white": ["beige", "grey"],
        "beige": ["white", "brown", "grey"],
        "grey": ["white", "beige", "black"],
        "black": ["grey"],
        "brown": ["beige"],
    }
    
    adjacent = adjacent_families.get(stone_base_lower, [])
    if any(c in adjacent for c in filter_lower):
        partial_score = 0.6
        if stone_confidence < 0.4:
            partial_score = 0.75
        return partial_score
    
    if stone_confidence < 0.4:
        return 0.4
    
    return 0.2


def _veining_match_score(
    stone_busyness: float,
    veining_min: float,
    veining_max: float
) -> float:
    """Compute veining preference score based on visual_busyness."""
    if veining_min == 0.0 and veining_max == 1.0:
        return 0.5
    
    target_center = (veining_min + veining_max) / 2
    target_range = (veining_max - veining_min) / 2
    
    distance = abs(stone_busyness - target_center)
    
    if distance <= target_range:
        return 1.0
    else:
        penalty = (distance - target_range) * 2
        return max(0.2, 1.0 - penalty)


def _pattern_match_score(
    stone_pattern: Optional[str],
    filter_patterns: List[str]
) -> float:
    """Compute pattern match score."""
    if not filter_patterns:
        return 0.5
    
    if not stone_pattern:
        return 0.3
    
    if stone_pattern.lower() in [p.lower() for p in filter_patterns]:
        return 1.0
    
    return 0.3


def _accent_match_score(
    stone_accents: List[Dict],
    filter_accents: List[str]
) -> float:
    """Compute accent color match score."""
    if not filter_accents:
        return 0.5
    
    if not stone_accents:
        return 0.3
    
    stone_accent_names = [a.get("color", "").lower() for a in stone_accents]
    filter_lower = [c.lower() for c in filter_accents]
    
    matches = sum(1 for c in filter_lower if c in stone_accent_names)
    
    return 0.3 + (0.7 * matches / len(filter_lower))


HUE_RESCUE_KEYWORDS = {
    "green": ["green", "verde", "verdi", "esmeralda", "emerald"],
    "red": ["red", "rosso", "rojo", "ruby", "rubino"],
    "orange": ["orange", "arancio", "coral"],
    "brown": ["brown", "marron", "chocolate", "cafe"],
}

BASE_COLOR_RESCUE_ALLOWED = {"grey", "black", "brown"}


def _evaluate_hue_rescue(
    stone: Dict,
    target_hue: str,
    family_stats: Dict[str, Dict],
    block_stats: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Evaluate if a stone qualifies for hue rescue (family rescue feature).
    
    Base-color guard: Rescue is ONLY allowed for base_color in grey/black/brown.
    White/beige stones are never rescued even if they have high green ratios.
    This guard does NOT affect direct matches (dominant_hue = target_hue).
    
    Returns:
        {
            "rescue_applied": bool,
            "rescue_reason": "family" | "name_assisted" | None,
            "block_hue_ratio": float,
            "family_hue_ratio": float,
            "direct_match": bool,
            "base_color": str,
            "rescue_blocked_by_base_color": bool
        }
    """
    stone_id = stone.get("company_stone_id", "")
    vendor = stone.get("vendor_name", "")
    stone_name = stone.get("stone_name", "")
    
    dominant_hue, hue_conf, _ = _get_display_tag(stone, "dominant_hue")
    base_color, _, _ = _get_display_tag(stone, "main_color")
    base_color_lower = (base_color or "").lower()
    
    block = block_stats.get(stone_id, {})
    block_ratio = block.get("block_hue_ratio", 0.0)
    
    family_key = f"{vendor}|||{stone_name}"
    family = family_stats.get(family_key, {})
    family_ratio = family.get("family_hue_ratio", 0.0)
    
    result = {
        "rescue_applied": False,
        "rescue_reason": None,
        "block_hue_ratio": block_ratio,
        "family_hue_ratio": family_ratio,
        "direct_match": False,
        "base_color": base_color_lower,
        "rescue_blocked_by_base_color": False
    }
    
    if dominant_hue and dominant_hue.lower() == target_hue.lower() and (hue_conf or 0.5) >= 0.50:
        result["direct_match"] = True
        return result
    
    if target_hue.lower() == "green":
        if base_color_lower not in BASE_COLOR_RESCUE_ALLOWED:
            result["rescue_blocked_by_base_color"] = True
            return result
    
    keywords = HUE_RESCUE_KEYWORDS.get(target_hue.lower(), [])
    name_lower = stone_name.lower() if stone_name else ""
    has_keyword = any(kw in name_lower for kw in keywords)
    
    if family_ratio >= 0.60 and block_ratio >= 0.20:
        result["rescue_applied"] = True
        result["rescue_reason"] = "family"
        return result
    
    if has_keyword and family_ratio >= 0.55 and block_ratio >= 0.15:
        result["rescue_applied"] = True
        result["rescue_reason"] = "name_assisted"
        return result
    
    return result


def _compute_search_score_v2(
    stone: Dict,
    query: str,
    filters: SearchFilters,
    visual_similarity: Optional[float] = None,
    debug: bool = False,
    color_browse_mode: bool = False
) -> float:
    """
    Compute final search score using display_tags.

    Standard text search (no image):
    FinalScore = 0.35*text + 0.20*base_color + 0.15*veining + 0.10*pattern + 0.10*accent + 0.10*manual

    Color browse mode (q="" AND base_colors set):
    FinalScore = 0.00*text + 0.55*base_color + 0.15*veining + 0.10*pattern + 0.10*accent + 0.10*manual
    (text contribution neutralized; base_color becomes dominant signal)

    Image search:
    FinalScore = 0.55*visual + 0.15*text + 0.10*base_color + 0.10*veining + 0.05*accent + 0.05*manual

    Cloudiness preference adds ranking bias when slider is touched:
    - Text/color browse search: +0.20 weight for cloudiness alignment
    - Image search: +0.15 weight for cloudiness alignment

    Smart mode: missing tags get null + confidence=0, small down-rank (no exclusion)

    If debug=True, stores breakdown in stone["_score_breakdown"].
    When color_browse_mode=True, sets stone["_color_browse_rescue"]=bool for bucket assignment.
    """
    base_color, _ = _get_effective_base_color_for_filter(stone)
    _, base_conf, _ = _get_display_tag(stone, "base_color")

    if not base_conf:
        _, base_conf, _ = _get_display_tag(stone, "main_color")
    base_conf = base_conf or 0.0
    
    busyness, _, _ = _get_display_tag(stone, "visual_busyness", 0.5)
    if isinstance(busyness, str):
        try:
            busyness = float(busyness)
        except:
            busyness = 0.5
    if busyness is None:
        busyness = 0.5
    
    pattern, _, _ = _get_display_tag(stone, "pattern_family")
    
    accents, _, _ = _get_display_tag(stone, "accent_colors", [])
    if isinstance(accents, str):
        try:
            accents = json.loads(accents)
        except:
            accents = []
    if not accents:
        accents = []
    
    cloudiness_score_raw, cloud_conf, _ = _get_display_tag(stone, "cloudiness_score")
    if isinstance(cloudiness_score_raw, str):
        try:
            cloudiness_score_raw = float(cloudiness_score_raw)
        except:
            cloudiness_score_raw = None
    has_cloudiness = cloudiness_score_raw is not None
    stone_cloudiness = cloudiness_score_raw if has_cloudiness else 0.5
    cloudiness_weight_factor = cloud_conf if (has_cloudiness and cloud_conf) else 0.3
    
    dominant_tone, tone_conf, _ = _get_display_tag(stone, "dominant_tone")
    if not dominant_tone:
        dominant_tone = "neutral"
        tone_conf = 0.5
    
    dominant_hue, hue_conf, _ = _get_display_tag(stone, "dominant_hue")
    if not dominant_hue:
        dominant_hue = "neutral"
        hue_conf = 0.5
    
    text_score = name_search.score_name_match(query, stone, debug=debug)
    base_score = _base_color_match_score(base_color, base_conf, filters.base_colors)

    green_hue_rescue_applied = False
    if (
        filters.base_colors and
        "green" in [c.lower() for c in filters.base_colors] and
        dominant_hue and dominant_hue.lower() == "green" and
        (hue_conf or 0.0) >= 0.65
    ):
        if base_score < 1.0:
            base_score = max(base_score, 1.0)
            green_hue_rescue_applied = True

    vein_score = _veining_match_score(busyness, filters.veining_min, filters.veining_max)
    pattern_score = _pattern_match_score(pattern, filters.patterns)
    accent_score = _accent_match_score(accents, filters.accent_colors)
    manual_boost = 1.0 if stone.get("has_manual_override") else 0.5
    
    tone_boost = 0.0
    if filters.dominant_tones:
        if dominant_tone and dominant_tone.lower() in [t.lower() for t in filters.dominant_tones]:
            tone_boost = 0.10 * (tone_conf if tone_conf else 0.5)
    
    hue_boost = 0.0
    if filters.dominant_hues:
        if dominant_hue and dominant_hue.lower() in [h.lower() for h in filters.dominant_hues]:
            hue_boost = 0.10 * (hue_conf if hue_conf else 0.5)
    
    if visual_similarity is not None:
        final = (
            0.55 * visual_similarity +
            0.15 * text_score +
            0.10 * base_score +
            0.10 * vein_score +
            0.05 * accent_score +
            0.05 * manual_boost
        )
        if filters.cloudiness_pref is not None:
            cloudiness_alignment = 1.0 - abs(stone_cloudiness - filters.cloudiness_pref)
            final += 0.15 * cloudiness_alignment * cloudiness_weight_factor
        final += tone_boost + hue_boost
        stone.pop("_color_browse_rescue", None)
    elif color_browse_mode:
        final = (
            0.55 * base_score +
            0.15 * vein_score +
            0.10 * pattern_score +
            0.10 * accent_score +
            0.10 * manual_boost
        )
        if filters.cloudiness_pref is not None:
            cloudiness_alignment = 1.0 - abs(stone_cloudiness - filters.cloudiness_pref)
            final += 0.20 * cloudiness_alignment * cloudiness_weight_factor
        final += tone_boost + hue_boost
        stone["_color_browse_rescue"] = green_hue_rescue_applied
    else:
        final = (
            0.35 * text_score +
            0.20 * base_score +
            0.15 * vein_score +
            0.10 * pattern_score +
            0.10 * accent_score +
            0.10 * manual_boost
        )
        if filters.cloudiness_pref is not None:
            cloudiness_alignment = 1.0 - abs(stone_cloudiness - filters.cloudiness_pref)
            final += 0.20 * cloudiness_alignment * cloudiness_weight_factor
        final += tone_boost + hue_boost
        stone.pop("_color_browse_rescue", None)
    
    if debug:
        stone["_score_breakdown"] = {
            "text_score": round(text_score, 4),
            "base_color_score": round(base_score, 4),
            "vein_score": round(vein_score, 4),
            "pattern_score": round(pattern_score, 4),
            "accent_score": round(accent_score, 4),
            "manual_boost": round(manual_boost, 4),
            "tone_boost": round(tone_boost, 4),
            "hue_boost": round(hue_boost, 4),
            "final_pre_rescue": round(final, 4),
            "green_hue_rescue_applied": green_hue_rescue_applied,
            "color_browse_mode": color_browse_mode,
        }
        ns_debug = stone.pop("_name_search_debug", None)
        if ns_debug:
            stone["_score_breakdown"]["name_search_debug"] = ns_debug
    
    return final


def _stone_to_result_v2(stone: Dict, similarity: Optional[float] = None) -> StoneResult:
    """Convert stone dict to StoneResult using display_tags."""
    accents_raw, _, acc_source = _get_display_tag(stone, "accent_colors", [])
    if isinstance(accents_raw, str):
        try:
            accents_raw = json.loads(accents_raw)
        except:
            accents_raw = []
    if not accents_raw:
        accents_raw = []
    
    accent_list = [
        AccentColor(color=a.get("color", ""), strength=a.get("strength", 0.0))
        for a in accents_raw if isinstance(a, dict)
    ]
    
    bg_color_val, bg_conf, bg_src = _get_display_tag(stone, "bg_color")
    if bg_color_val and bg_src in ("manual", "ai") and bg_color_val.lower().strip() in _CHROMATIC_BG_COLORS:
        base_color = _normalize_chromatic(bg_color_val.lower().strip())
        base_conf = bg_conf
    else:
        base_color, base_conf, base_source = _get_display_tag(stone, "base_color")

        if not base_color:
            base_color, base_conf, base_source = _get_display_tag(stone, "main_color")
        if base_color:
            base_color = _normalize_chromatic(base_color.lower())
    
    vein_intensity, vein_conf, vein_source = _get_display_tag(stone, "vein_intensity")
    pattern_family, _, _ = _get_display_tag(stone, "pattern_family")
    visual_busyness, _, _ = _get_display_tag(stone, "visual_busyness", 0.0)
    drama_score, _, _ = _get_display_tag(stone, "drama_score", 0.0)
    dominant_tone, tone_conf, _ = _get_display_tag(stone, "dominant_tone")
    dominant_hue, hue_conf, _ = _get_display_tag(stone, "dominant_hue")
    
    try:
        visual_busyness = float(visual_busyness) if visual_busyness else 0.0
    except:
        visual_busyness = 0.0
    try:
        drama_score = float(drama_score) if drama_score else 0.0
    except:
        drama_score = 0.0
    try:
        tone_conf = float(tone_conf) if tone_conf else 0.0
    except:
        tone_conf = 0.0
    try:
        hue_conf = float(hue_conf) if hue_conf else 0.0
    except:
        hue_conf = 0.0
    
    tags = {k: v for k, v in stone.get("display_tags", {}).items()
            if not k.endswith("_debug")}

    rescue_info = stone.get("_rescue_info")
    if rescue_info:
        rescue_info["bucket"] = stone.get("_rank_bucket", 3)
        tags["_rescue"] = rescue_info

    rank_bucket = stone.get("_rank_bucket")
    if rank_bucket is not None:
        tags["_rank_bucket"] = rank_bucket

    cb_rescue = stone.get("_color_browse_rescue")
    if cb_rescue is not None:
        tags["_color_browse_rescue"] = cb_rescue
    
    score_breakdown = stone.get("_score_breakdown")
    if score_breakdown:
        tags["_score_breakdown"] = score_breakdown
    
    return StoneResult(
        company_stone_id=stone.get("company_stone_id", ""),
        batch_id=stone.get("batch_id"),
        stone_name=stone.get("stone_name"),
        vendor_name=stone.get("vendor_name"),
        thumbnail_url=stone.get("thumbnail_url"),
        base_color=base_color,
        base_color_confidence=base_conf,
        vein_intensity=vein_intensity,
        pattern_family=pattern_family,
        accent_colors=accent_list,
        visual_busyness=visual_busyness,
        drama_score=drama_score,
        dominant_tone=dominant_tone,
        dominant_tone_confidence=tone_conf,
        dominant_hue=dominant_hue,
        dominant_hue_confidence=hue_conf,
        has_manual_override=stone.get("has_manual_override", False),
        similarity_score=similarity,
        final_score=stone.get("final_score", 0.0),
        tags=tags,
        match_tier=stone.get("_match_tier"),
    )


def _apply_hard_filters_v2(stones: List[Dict], filters: SearchFilters) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Apply hard filters using display_tags.
    
    GLOBAL STRICT MODE (strict_exclude=True):
        All active filters become hard WHERE conditions (AND logic).
        - Business Tone: WHERE effective_tone IN selected_tones
        - Base Color: WHERE effective_base_color IN selected_colors
        - Patterns: WHERE pattern IN selected_patterns
        - Cloudiness: WHERE cloudiness BETWEEN min AND max
    
    SMART MODE (strict_exclude=False):
        Only base_color_mode=strict excludes; all other filters are ranking-only.
    
    Manual overrides always win: COALESCE(manual.value, ai.value)
    
    Returns:
        (filtered_stones, debug_counts)
    """
    global_strict = filters.strict_exclude
    
    debug = {
        "input_count": len(stones),
        "strict_exclude": global_strict,
        "removed_by_confidence": 0,
        "removed_by_base_color": 0,
        "removed_by_stone_type": 0,
        "removed_by_pattern": 0,
        "removed_by_accent": 0,
        "removed_by_cloudiness": 0,
        "removed_by_tone": 0,
        "removed_by_hue": 0,
    }
    
    filtered = []
    
    for stone in stones:
        base_color, base_source = _get_effective_base_color_for_filter(stone)
        _, base_conf, _ = _get_display_tag(stone, "base_color")

        if not base_conf:
            _, base_conf, _ = _get_display_tag(stone, "main_color")
        base_conf = base_conf or 0.0
        
        if filters.min_confidence > 0:
            if base_conf < filters.min_confidence:
                debug["removed_by_confidence"] += 1
                continue
        
        base_color_strict = False
        filter_colors_lower = [c.lower() for c in filters.base_colors] if filters.base_colors else []

        # Per-chip colour gating.
        #
        # Task #422 — the "multi" chip is semantically exclusive: a
        # stone either is or isn't multi-colour, so when "multi" is in
        # the filter set the gate becomes a hard exclusion in BOTH
        # Smart and Strict modes (and regardless of which other colour
        # chips are also selected — multi + grey still returns only
        # multi stones).  This branch runs before the brown / non-brown
        # evaluation so it fully owns the multi case without disturbing
        # the existing BROWN EXACT / EXPANDED logic or the Smart-mode
        # admit-anything semantics of the other colour chips when multi
        # is NOT present.
        #
        # Task #415 (prior) unified per-chip pass-evaluation for the
        # non-multi case: each active colour chip evaluates independently
        # against the stone's effective base_color and the stone is
        # dropped only when every active chip would reject it.  This
        # preserves the brown Smart-mode expansion (beige + warm
        # dominant_hue) and keeps every non-brown chip's existing
        # Smart-vs-Strict semantics intact.
        brown_in_filter = "brown" in filter_colors_lower
        multi_in_filter = "multi" in filter_colors_lower
        # Task #442 — `_expand_color_filter_set` OR-expands the
        # buyer-facing "yellow" chip into the full GOLD family
        # ({yellow, orange, gold, amber, copper, bronze, ochre,
        # terra cotta}) so a stone whose only signal is
        # `main_color = orange` is admitted by the strict-mode chip
        # gate.  Non-yellow chips pass through unchanged.
        other_chip_colors = _expand_color_filter_set([
            c for c in filter_colors_lower if c not in ("brown", "multi")
        ])
        any_chip_active = (
            brown_in_filter or multi_in_filter or bool(other_chip_colors)
        )

        if any_chip_active:
            effective_base_color, base_color_source = (
                _get_effective_base_color_for_filter(stone)
            )

            # --- Multi hard exclude (Task #422) ---
            # Fires whenever "multi" is in the filter set, in any mode,
            # alongside any combination of other chips.  Drops the
            # stone unconditionally when its effective base_color is
            # not "multi"; passing stones fall through to the rest of
            # the hard filters (stone_types, pattern_families, etc.).
            if multi_in_filter:
                if effective_base_color != "multi":
                    debug["removed_by_base_color"] += 1
                    debug.setdefault("multi_hard_rejected", 0)
                    debug["multi_hard_rejected"] += 1
                    continue
                debug.setdefault("multi_hard_passed", 0)
                debug["multi_hard_passed"] += 1
            else:
                # --- Brown evaluation (Exact vs Expanded) ---
                brown_pass = False
                brown_pass_via_brown = False
                brown_beige_trigger_hue = None
                if brown_in_filter:
                    if base_color_strict:
                        if effective_base_color == "brown":
                            brown_pass = True
                            brown_pass_via_brown = True
                    else:
                        if effective_base_color == "brown":
                            brown_pass = True
                            brown_pass_via_brown = True
                        elif effective_base_color == "beige":
                            dom_hue, _, _ = _get_display_tag(stone, "dominant_hue")
                            if dom_hue and dom_hue.lower() in ("brown", "orange"):
                                brown_pass = True
                                brown_beige_trigger_hue = dom_hue.lower()

                # --- Non-brown chip evaluation ---
                # Strict mode: stone must literally match one of these chips.
                # Smart mode: these chips are ranking-only and admit anything,
                #             so being in the filter set is enough to keep the
                #             stone alive past base-color gating.
                other_chip_pass = False
                if other_chip_colors:
                    if base_color_strict:
                        if (
                            effective_base_color
                            and effective_base_color in other_chip_colors
                        ):
                            other_chip_pass = True
                    else:
                        other_chip_pass = True

                if not (brown_pass or other_chip_pass):
                    debug["removed_by_base_color"] += 1
                    if brown_in_filter:
                        if base_color_strict:
                            debug.setdefault("brown_exact_rejected", 0)
                            debug["brown_exact_rejected"] += 1
                        else:
                            debug.setdefault("brown_expanded_rejected", 0)
                            debug["brown_expanded_rejected"] += 1
                    continue

                # Per-chip pass counters (mirror the legacy debug keys).
                if brown_pass:
                    if base_color_strict:
                        debug.setdefault("brown_exact_passed", 0)
                        debug["brown_exact_passed"] += 1
                        if base_color_source == "manual":
                            debug.setdefault("brown_exact_passed_via_manual", 0)
                            debug["brown_exact_passed_via_manual"] += 1
                    else:
                        debug.setdefault("brown_expanded_passed", 0)
                        debug["brown_expanded_passed"] += 1
                        if brown_pass_via_brown:
                            debug.setdefault("brown_expanded_passed_via_brown", 0)
                            debug["brown_expanded_passed_via_brown"] += 1
                            if base_color_source == "manual":
                                debug.setdefault("brown_expanded_passed_via_manual", 0)
                                debug["brown_expanded_passed_via_manual"] += 1
                        elif brown_beige_trigger_hue:
                            debug.setdefault("brown_expanded_passed_via_beige_warm", 0)
                            debug["brown_expanded_passed_via_beige_warm"] += 1
                            key = (
                                f"brown_expanded_beige_trigger_"
                                f"{brown_beige_trigger_hue}"
                            )
                            debug.setdefault(key, 0)
                            debug[key] += 1
        
        if filters.stone_types:
            family = stone.get("canonical_family", "").lower()
            if family and family not in [t.lower() for t in filters.stone_types]:
                debug["removed_by_stone_type"] += 1
                continue
        
        # Pattern P4 (J.6): the strict-mode-only `filters.patterns` gate
        # was removed here. The `filters.pattern_families` block below is
        # the always-hard pattern gate; the SearchFilters validator folds
        # any legacy `patterns` payload into `pattern_families`, so this
        # branch is fully subsumed.

        if filters.pattern_families:
            pf_val, pf_source = _get_effective_pattern_family_for_filter(stone)
            pf_lower = pf_val if pf_val else ""
            # Split the request into labelled values and the unlabeled sentinel.
            # Labelled values are normalised through PATTERN_FAMILY_ALIASES so
            # legacy inputs like "brecciated" match the canonical "breccia".
            pf_filter_labels = []
            pf_filter_includes_unlabeled = False
            try:
                aliases = taxonomy_client.get_pattern_family_aliases()
            except Exception:
                aliases = {
                    "brecciated": "breccia",
                    "onyx-like": "linear",
                    "dramatic-mix": "breccia",
                    "web": "webbed",
                    "bold veined": "bold_veined",
                    "bold-veined": "bold_veined",
                }
            for p in filters.pattern_families:
                p_low = p.lower().strip() if isinstance(p, str) else ""
                if p_low == PATTERN_FAMILY_UNLABELED_SENTINEL:
                    pf_filter_includes_unlabeled = True
                else:
                    pf_filter_labels.append(
                        aliases.get(p_low, p_low)
                    )
            label_match = bool(pf_lower) and pf_lower in pf_filter_labels
            unlabeled_match = pf_filter_includes_unlabeled and pf_source == "missing"
            # Task #391 — Bold Veined widening.  When the buyer-facing
            # "bold_veined" chip is in the filter, also admit stones
            # whose primary classification is something else but whose
            # accent_vein/network signals mark them as bold-veined.
            # OR of three signals: pattern_family=bold_veined (already
            # covered by label_match) OR has_bold_accent_vein=true OR
            # network_thickness=bold.
            bold_widen_match = (
                "bold_veined" in pf_filter_labels
                and _stone_matches_bold_veined_widening(stone)
            )
            if not (label_match or unlabeled_match or bold_widen_match):
                debug["removed_by_pattern_family"] = debug.get("removed_by_pattern_family", 0) + 1
                continue
        
        accent_strict = False
        if accent_strict and filters.accent_colors:
            accents, _, _ = _get_display_tag(stone, "accent_colors", [])
            if isinstance(accents, str):
                try:
                    accents = json.loads(accents)
                except:
                    accents = []
            if accents:
                accent_names = [a.get("color", "").lower() for a in accents if isinstance(a, dict)]
                if not any(c.lower() in accent_names for c in filters.accent_colors):
                    debug["removed_by_accent"] += 1
                    continue
            else:
                debug["removed_by_accent"] += 1
                continue
        
        cloudiness_hard = filters.cloudiness_strict or global_strict
        if cloudiness_hard and (filters.cloudiness_min is not None or filters.cloudiness_max is not None):
            cloudiness_score, _, _ = _get_display_tag(stone, "cloudiness_score")
            if cloudiness_score is not None:
                try:
                    cloudiness_val = float(cloudiness_score)
                    if filters.cloudiness_min is not None and cloudiness_val < filters.cloudiness_min:
                        debug["removed_by_cloudiness"] += 1
                        continue
                    if filters.cloudiness_max is not None and cloudiness_val > filters.cloudiness_max:
                        debug["removed_by_cloudiness"] += 1
                        continue
                except (ValueError, TypeError):
                    pass
        
        tone_strict = filters.dominant_tone_strict or global_strict
        if tone_strict and filters.dominant_tones:
            tone, _, _ = _get_display_tag(stone, "dominant_tone")
            if tone:
                if tone.lower() not in [t.lower() for t in filters.dominant_tones]:
                    debug["removed_by_tone"] += 1
                    continue
            else:
                if "neutral" not in [t.lower() for t in filters.dominant_tones]:
                    debug["removed_by_tone"] += 1
                    continue
        if filters.dominant_hues:
            # Generic smart soft-matching logic for all colors (green, red, blue_grey, etc.)
            requested_hues = [h.lower() for h in filters.dominant_hues]
            effective_tone, _ = _get_effective_tone_for_filter(stone)
            hue, _, _ = _get_display_tag(stone, "dominant_hue")
            dominant_tone, _, _ = _get_display_tag(stone, "dominant_tone")
            
            passes = False
            for req_h in requested_hues:
                if (effective_tone and effective_tone.lower() == req_h) or \
                   (hue and hue.lower() == req_h) or \
                   (dominant_tone and dominant_tone.lower() == req_h):
                    passes = True
                    break
                    
            if passes:
                debug.setdefault("smart_hue_passed", 0)
                debug["smart_hue_passed"] += 1
            else:
                debug["removed_by_hue"] += 1
                continue
        
        filtered.append(stone)
    
    debug["output_count"] = len(filtered)
    
    return filtered, debug


def _load_embedding(company_stone_id: str) -> Optional[np.ndarray]:
    """Load embedding for a stone.

    Uses the shared in-memory foundation.db embedding cache.  Falls back to
    per-stone .npy files in EMBEDDINGS_CACHE_DIR if the foundation.db cache
    has no entry for this stone (legacy path, kept for safety).

    Fallback rule: if no embedding is found by either path, returns None and
    the caller produces no visual similarity for this stone — identical to
    pre-cache behaviour.
    """
    emb = ml_svc.get_embedding(company_stone_id)
    if emb is not None:
        return emb
    emb_path = EMBEDDINGS_CACHE_DIR / f"{company_stone_id}.npy"
    if emb_path.exists():
        try:
            return np.load(emb_path)
        except Exception:
            pass
    return None


def _compute_visual_similarities(
    query_embedding: np.ndarray,
    stones: List[Dict]
) -> Dict[str, float]:
    """Compute cosine similarity between query embedding and all stones.

    Uses the shared in-memory embedding cache (single DB load per TTL window)
    instead of per-stone .npy disk reads.  Stones with no embedding are skipped
    — their similarity is 0.0, identical to the previous per-.npy behaviour.
    """
    all_embeddings = ml_svc.load_all_embeddings()
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    similarities: Dict[str, float] = {}
    for stone in stones:
        stone_id = stone["company_stone_id"]
        emb = all_embeddings.get(stone_id)
        if emb is not None:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_norm.flatten(), emb_norm.flatten()))
            similarities[stone_id] = max(0.0, min(1.0, (sim + 1) / 2))
    return similarities




TONE_SYNONYMS = {
    "green": ["green", "verde", "emerald", "olive", "forest", "jade", "mint", "teal"],
    "red": ["red", "rojo", "scarlet", "crimson", "burgundy", "maroon", "ruby", "rose"],
    "warm": ["warm", "gold", "golden", "orange", "amber", "honey", "caramel", "copper", "bronze", "yellow"],
    "cool": ["cool", "blue", "steel", "silver", "slate", "icy", "cold", "navy"],
}


def _detect_tone_from_query(query: str, existing_tones: List[str]) -> List[str]:
    """
    Detect tone keywords from search query.
    Returns inferred tones if no explicit tones are set.
    """
    if existing_tones:
        return existing_tones
    
    if not query:
        return []
    
    query_lower = query.lower()
    detected = set()
    
    for tone, keywords in TONE_SYNONYMS.items():
        for kw in keywords:
            if kw in query_lower:
                detected.add(tone)
                break
    
    return list(detected)





@router.get("/health")
async def api_health():
    """
    Public health / readiness probe for the search module.

    Returns:
      - status: "ok" when the stone cache is warm and all sub-systems are ready.
      - internal_secret_configured: whether REPLIT_INTERNAL_SECRET is set (bool).
        A False value means all /internal/* endpoints are currently rejecting requests.
      - cache: live snapshot of the stone cache state.
      - search_log_db: whether the durable search log DB is reachable.
    """
    secret_configured = bool(os.environ.get("REPLIT_INTERNAL_SECRET"))
    if not secret_configured:
        logger.warning(
            "[health] REPLIT_INTERNAL_SECRET is not set — "
            "all /internal/* endpoints are rejecting requests."
        )

    with _stone_cache_lock:
        cache_warm = _stone_cache_payload is not None
        cache_age_s = round(time.time() - _stone_cache_ts, 1) if cache_warm else None
        cache_stones = len(_stone_cache_payload) if cache_warm else 0
        cache_dataset = _stone_cache_dataset_id

    log_db_ok = _slog.ping()

    return {
        "status": "ok",
        "internal_secret_configured": secret_configured,
        "cache": {
            "warm": cache_warm,
            "age_s": cache_age_s,
            "stones": cache_stones,
            "dataset_id": cache_dataset,
            "hits": _stone_cache_hits,
            "misses": _stone_cache_misses,
        },
        "search_log_db": {
            "ok": log_db_ok,
        },
    }


async def search_with_embedding(
    query_embedding: np.ndarray,
    q: str = "",
    filters_json: str = "{}",
    limit: int = 60,
    offset: int = 0
):
    """
    Search stones using a precomputed visual embedding plus optional text query and filters.
    """
    start = time.perf_counter()
    
    try:
        filters_dict = json.loads(filters_json) if filters_json else {}
        filters = SearchFilters(**filters_dict)
    except Exception:
        filters = SearchFilters()
    
    inferred_tones = _detect_tone_from_query(q, filters.dominant_tones)
    if inferred_tones and not filters.dominant_tones:
        filters.dominant_tones = inferred_tones
    
    dataset_id, stones, *_ = _load_all_stone_data_batched()
    stones, _ = _apply_hard_filters_v2(stones, filters)
    
    similarities = _compute_visual_similarities(query_embedding, stones)
    
    if q:
        name_search.log_parsed_query(q)

    for stone in stones:
        stone_id = stone["company_stone_id"]
        visual_sim = similarities.get(stone_id)
        score = _compute_search_score_v2(stone, q, filters, visual_similarity=visual_sim)
        stone["final_score"] = score
        stone["similarity_score"] = visual_sim
    
    stones.sort(key=lambda x: x["final_score"], reverse=True)
    
    total = len(stones)
    paginated = stones[offset:offset + limit]
    
    results = [_stone_to_result_v2(s, s.get("similarity_score")) for s in paginated]
    
    latency = (time.perf_counter() - start) * 1000
    search_request_id, _log_ts = _log_search(
        q,
        filters.model_dump(),
        True,
        total,
        latency,
        mode="image",
        dataset_id=dataset_id,
        query_offset=offset,
        query_limit=limit,
    )
    
    return SearchResponse(
        results=results,
        total=total,
        query=q,
        search_request_id=search_request_id,
        backend_time_ms=latency,
    )


async def search_with_embedding_v3(
    query_embedding: np.ndarray,
    q: str = "",
    filters_json: str = "{}",
    limit: int = 60,
    offset: int = 0
):
    """
    Search stones using a precomputed visual embedding and enrich results with V3 classifiers.
    """

    response = await search_with_embedding(
        query_embedding=query_embedding,
        q=q,
        filters_json=filters_json,
        limit=limit,
        offset=offset
    )
    result_dict = response.model_dump()
    enrich_with_v3(result_dict["results"])
    return result_dict


async def search_stones(request: SearchRequest):
    """
    Search stones with text query and filters.
    
    - Empty query + no filters = newest 60 stones (never empty UI)
    - Smart mode = soft ranking only (no exclusions)
    - Strict mode = hard filter by user-selected filters
    
    Hue Family Rescue (smart mode only):
    - When strict_hue=false and dominant_hues is set, rescued blocks from green families
      are included but ranked lower than direct matches.
    """
    global _last_db_substage_timing

    start = time.perf_counter()
    t_db_start = time.perf_counter()

    inferred_tones = _detect_tone_from_query(request.q, request.filters.dominant_tones)
    effective_filters = request.filters.copy()
    if inferred_tones and not request.filters.dominant_tones:
        effective_filters.dominant_tones = inferred_tones

    needs_tags_for_filter = _needs_tags_for_search(effective_filters)

    # -----------------------------------------------------------------------
    # Candidate-first: two-tier cache architecture.
    #
    # TEXT-ONLY PATH (no hard tag filters):
    #   Phase A: get lightweight candidates from tier-1 cache (~0ms hit, ~17ms miss).
    #   Text-score all candidates on lightweight data, sort, paginate.
    #   Phase B: _hydrate_stones_for_page() ONLY the paginated page (5-60 stones).
    #   Cost: 0ms SQL (tier-1 hit) + ~5ms Phase B. Full cold miss: ~25ms total.
    #
    # TAG-FILTERED PATH (hard tag filters active):
    #   Use tier-2 cache (fully hydrated stones, ~0ms SQL hit).
    #   Cache miss: Phase A (~17ms) + Phase B all (~100-300ms) → stored in tier-2.
    #   Filter → score (with full tags) → paginate.
    #
    # Both caches are invalidated by invalidate_stone_cache() on any write or
    # dataset switch, ensuring fresh data on the next request.
    # -----------------------------------------------------------------------
    if needs_tags_for_filter:
        dataset_id, candidates = _load_all_stone_data_batched()
        phase_b_mode = "broad"
        phase_b_stones_hydrated = 0
        _cache_status = "BROAD"
        _cache_age_ms = 0.0
    else:
        dataset_id, _lw_candidates, _phase_a_timing = _get_lightweight_candidates()
        _last_db_substage_timing = _phase_a_timing
        phase_b_mode = "deferred"
        phase_b_stones_hydrated = 0
        _cache_status = "LW_HIT" if _phase_a_timing.get("mode") == "lw_cache_hit" else "LW_MISS"
        _cache_age_ms = 0.0
        candidates = [{**s, "display_tags": {}} for s in _lw_candidates]

    total_candidates_before_filters = len(candidates)

    t_db_end = time.perf_counter()



    stones = candidates

    # Task #390 — stone dicts come from the shared cache and persist
    # across requests; clear any stale `_match_tier` from a prior
    # Smart-mode sectioning request before this request runs so non-
    # sectioning callers don't surface a leftover tier in match_tier.
    for _s in stones:
        if "_match_tier" in _s:
            _s["_match_tier"] = None

    use_rescue = (
        effective_filters.dominant_hues and
        not effective_filters.dominant_hue_strict and
        len(effective_filters.dominant_hues) == 1
    )

    family_stats = {}
    block_stats = {}
    rescue_count = 0

    if use_rescue:
        target_hue = effective_filters.dominant_hues[0]
        family_stats = taxonomy_client.get_family_hue_stats(target_hue)
        block_stats = taxonomy_client.get_block_hue_stats(target_hue)

    # Task #390 — Smart-mode tier sections activation.  Active only when
    # both a colour filter and a pattern_families filter are present and
    # neither global strict_exclude nor base_color_mode='strict' is set.
    # Explicitly excluded: Color Browse (no query + base_colors set AND
    # no pattern chips — Task #419 narrowed this so Color Browse owns
    # only the colour-only-no-query case and yields to Smart sectioning
    # whenever pattern chips join the filter set) and Hue Rescue
    # (single non-strict dominant_hue) — both modes have their own
    # `_rank_bucket` ordering and must not be preempted by the tier
    # sort.  When active, suppress the hard pattern_families gate so
    # colour-match-only candidates (tier 2) survive _apply_hard_filters_v2;
    # the post-scoring pass below classifies each survivor into tier
    # 1/2/3 and drops stones that match neither.
    _color_browse_mode_for_activation = bool(
        not (request.q or "").strip()
        and effective_filters.base_colors
        and not effective_filters.pattern_families
    )
    smart_section_active = (
        bool(effective_filters.base_colors)
        and bool(effective_filters.pattern_families)
        and not _color_browse_mode_for_activation
        and not use_rescue
    )
    section_colour_set: set = set()
    section_pattern_set: set = set()
    if smart_section_active:
        section_colour_set = {
            _normalize_chromatic(c.lower())
            for c in effective_filters.base_colors
            if isinstance(c, str) and c
        }
        try:
            aliases = taxonomy_client.get_pattern_family_aliases()
        except Exception:
            aliases = {
                "brecciated": "breccia",
                "onyx-like": "linear",
                "dramatic-mix": "breccia",
                "web": "webbed",
                "bold veined": "bold_veined",
                "bold-veined": "bold_veined",
            }
        for p in effective_filters.pattern_families:
            p_low = p.lower().strip() if isinstance(p, str) else ""
            if p_low and p_low != PATTERN_FAMILY_UNLABELED_SENTINEL:
                section_pattern_set.add(
                    aliases.get(p_low, p_low)
                )
        filters_for_hard_filter = effective_filters.copy()
        filters_for_hard_filter.pattern_families = []
    else:
        filters_for_hard_filter = effective_filters

    t_filter_start = time.perf_counter()
    stones, filter_debug = _apply_hard_filters_v2(stones, filters_for_hard_filter)
    t_filter_end = time.perf_counter()
    
    suppressed_count = 0
    matched_before_suppression = len(stones)
    suppression_applied = False
    shown_after_suppression = len(stones)
    
    color_browse_mode = bool(not (request.q or "").strip() and effective_filters.base_colors)
    # Task #442 — OR-expand the "yellow" chip into the GOLD family so
    # Color Browse `_rank_bucket` treats orange/gold/amber/… stones as
    # tier 1 (direct match) instead of demoting them to tier 3.
    filter_colors_lower_bc = sorted(_expand_color_filter_set(effective_filters.base_colors))

    t_score_start = time.perf_counter()
    if request.q:
        name_search.log_parsed_query(request.q)

    for stone in stones:
        score = _compute_search_score_v2(
            stone, request.q, effective_filters,
            debug=request.debug_scores,
            color_browse_mode=color_browse_mode
        )

        if color_browse_mode:
            eff_bc, _ = _get_effective_base_color_for_filter(stone)
            if eff_bc and eff_bc.lower() in filter_colors_lower_bc:
                stone["_rank_bucket"] = 1
            elif stone.get("_color_browse_rescue"):
                stone["_rank_bucket"] = 2
                rescue_count += 1
            else:
                stone["_rank_bucket"] = 3
        elif use_rescue:
            target_hue = effective_filters.dominant_hues[0]
            rescue_info = _evaluate_hue_rescue(stone, target_hue, family_stats, block_stats)
            stone["_rescue_info"] = rescue_info
            
            if rescue_info["direct_match"]:
                stone["_rank_bucket"] = 1
            elif rescue_info["rescue_applied"]:
                stone["_rank_bucket"] = 2
                rescue_count += 1
            else:
                stone["_rank_bucket"] = 3
        else:
            stone["_rank_bucket"] = 1
        
        stone["final_score"] = score
    t_score_end = time.perf_counter()

    # Task #390 — Tier classification + drop "matches neither" stones.
    smart_tier_counts: Optional[Dict[str, int]] = None
    if smart_section_active:
        tier_1 = tier_2 = tier_3 = 0
        classified: List[Dict] = []
        for stone in stones:
            colour_hit = _stone_matches_section_colour(stone, section_colour_set)
            pattern_hit = _stone_matches_section_pattern(stone, section_pattern_set)
            if colour_hit and pattern_hit:
                stone["_match_tier"] = 1
                tier_1 += 1
            elif colour_hit:
                stone["_match_tier"] = 2
                tier_2 += 1
            elif pattern_hit:
                stone["_match_tier"] = 3
                tier_3 += 1
            else:
                # Matched neither — would have been removed by the
                # pattern_families hard filter; the colour gate is soft
                # in Smart mode so misses leak through.  Drop here.
                continue
            classified.append(stone)
        stones = classified
        smart_tier_counts = {
            "tier_1_both": tier_1,
            "tier_2_colour_only": tier_2,
            "tier_3_pattern_only": tier_3,
        }

    # Task #404 — tiered ranking for the default best_match branch.
    # Compute query intent from the alias-normalised filter sets that the
    # hard filter already used.  Empty sets degrade cleanly to
    # ``(1, 1, -final_score, -completeness)`` so unfiltered searches keep
    # score-then-completeness ordering.
    tiered_default_active = (
        request.sort == "best_match"
        and not smart_section_active
        and not color_browse_mode
        and not use_rescue
    )
    if tiered_default_active:
        query_colour_set = {
            c.lower() for c in (request.filters.base_colors or []) if c
        }
        try:
            aliases = taxonomy_client.get_pattern_family_aliases()
        except Exception:
            aliases = {
                "brecciated": "breccia",
                "onyx-like": "linear",
                "dramatic-mix": "breccia",
                "web": "webbed",
                "bold veined": "bold_veined",
                "bold-veined": "bold_veined",
            }
        query_pattern_set = {
            aliases.get(p.lower(), p.lower())
            for p in (request.filters.pattern_families or []) if p
        }
        for stone in stones:
            stone["_main_color_tier"] = _compute_main_color_tier(stone, query_colour_set)
            stone["_main_pattern_tier"] = _compute_main_pattern_tier(stone, query_pattern_set)
            stone["_main_completeness"] = _compute_completeness_score(stone)
            # STOCK_AGE_LAYER deferred — no per-stone supplier timestamp exists.
            stone["_ghost_stock_candidate"] = False

    t_sort_start = time.perf_counter()
    if request.sort == "best_match":
        if smart_section_active:
            stones.sort(key=lambda x: (x.get("_match_tier", 3), -x["final_score"]))
        elif color_browse_mode or use_rescue:
            stones.sort(key=lambda x: (x.get("_rank_bucket", 3), -x["final_score"]))
        else:
            stones.sort(key=lambda x: (
                x.get("_main_color_tier", 3),
                x.get("_main_pattern_tier", 3),
                -x.get("final_score", 0.0),
                -x.get("_main_completeness", 0),
            ))
    t_sort_end = time.perf_counter()

    # Task #404 — vendor cap on the display layer (post-sort, pre-pagination).
    vendor_cap_meta: Dict[str, Any] = {
        "vendor_cap_applied": False,
        "vendor_cap_relaxed": False,
        "distinct_vendor_count": 0,
    }
    if request.sort == "best_match":
        stones, vendor_cap_meta = _apply_vendor_cap(stones)
        ghost_count = sum(1 for s in stones if s.get("_ghost_stock_candidate"))
        if ghost_count:
            logger.info("main_search ghost_stock_candidate count=%d", ghost_count)

    total = len(stones)

    if phase_b_mode == "deferred":
        _hydrate_facet_tags_for_candidates(stones)

    t_facet_start = time.perf_counter()
    pf_facets = {}
    pf_unlabeled = 0
    bc_facets = {}
    for stone in stones:
        pf_val, pf_source = _get_effective_pattern_family_for_filter(stone)
        if pf_val:
            pf_facets[pf_val] = pf_facets.get(pf_val, 0) + 1
        else:
            pf_unlabeled += 1

        eff_color, _ = _get_effective_base_color_for_filter(stone)
        if eff_color:
            eff_color_norm = _normalize_chromatic(eff_color)
            bc_facets[eff_color_norm] = bc_facets.get(eff_color_norm, 0) + 1
    t_facet_end = time.perf_counter()

    paginated = stones[request.offset:request.offset + request.limit]

    # -----------------------------------------------------------------------
    # Candidate-first: Phase B (deferred) — text-only search path.
    # Hydrates display_tags for ONLY the final-page stones (5-60),
    # not the full 2,137-stone candidate set.
    #
    # NOTE: candidates in the deferred path are per-request shallow copies
    # (created above via [s.copy() for s in _lw_candidates]), so hydrating
    # paginated stones here is safe — it does not mutate the shared
    # _lw_cache_payload objects in the tier-1 cache.
    # -----------------------------------------------------------------------
    t_phase_b_start = time.perf_counter()
    if phase_b_mode == "deferred":
        _hydrate_stones_for_page(paginated)
        phase_b_stones_hydrated = len(paginated)
    t_phase_b_ms = (time.perf_counter() - t_phase_b_start) * 1000

    if request.filters.cloudiness_pref is not None or request.filters.cloudiness_strict:
        dataset_id_for_lazy = dataset_id

    t_hydrate_start = time.perf_counter()
    results = [_stone_to_result_v2(s) for s in paginated]
    t_hydrate_end = time.perf_counter()

    t_serialize_start = time.perf_counter()
    payload_bytes = len(json.dumps([r.model_dump() for r in results]).encode("utf-8"))
    t_serialize_end = time.perf_counter()

    latency = (time.perf_counter() - start) * 1000
    search_request_id, _log_ts = _log_search(
        request.q,
        request.filters.model_dump(),
        False,
        total,
        latency,
        mode="text",
        dataset_id=dataset_id,
        query_offset=request.offset,
        query_limit=request.limit,
    )
    _slog.log_results(
        search_request_id=search_request_id,
        timestamp=_log_ts,
        dataset_id=dataset_id,
        search_mode="text",
        query_offset=request.offset,
        page_size=request.limit,
        stones=paginated,
    )

    db_ms       = (t_db_end       - t_db_start)       * 1000
    filter_ms   = (t_filter_end   - t_filter_start)   * 1000
    score_ms    = (t_score_end    - t_score_start)     * 1000
    sort_ms     = (t_sort_end     - t_sort_start)      * 1000
    facet_ms    = (t_facet_end    - t_facet_start)     * 1000
    hydrate_ms  = (t_hydrate_end  - t_hydrate_start)   * 1000
    serialize_ms = (t_serialize_end - t_serialize_start) * 1000
    logger.info(
        "[Search timing] db=%.1fms filter=%.1fms score=%.1fms sort=%.1fms "
        "facet=%.1fms hydrate=%.1fms serialize=%.1fms total=%.1fms payload=%dB n=%d "
        "cache=%s cache_age_ms=%.0f",
        db_ms, filter_ms, score_ms, sort_ms,
        facet_ms, hydrate_ms, serialize_ms, latency, payload_bytes, len(results),
        _cache_status, _cache_age_ms,
    )
    if latency > 300:
        logger.warning(
            "[Search budget] total=%.1fms > 300ms q='%s' n=%d", latency, request.q, len(results)
        )
    if hydrate_ms > 50:
        logger.warning(
            "[Search budget] hydrate=%.1fms > 50ms n=%d", hydrate_ms, len(results)
        )
    if serialize_ms > 50:
        logger.warning(
            "[Search budget] serialize=%.1fms > 50ms n=%d payload=%dB",
            serialize_ms, len(results), payload_bytes,
        )
    if payload_bytes > 200_000:
        logger.warning(
            "[Search budget] payload=%dB > 200KB n=%d", payload_bytes, len(results)
        )
    
    bucket_counts = {"direct": 0, "rescued": 0, "other": 0}
    for stone in stones:
        bucket = stone.get("_rank_bucket", 3)
        if bucket == 1:
            bucket_counts["direct"] += 1
        elif bucket == 2:
            bucket_counts["rescued"] += 1
        else:
            bucket_counts["other"] += 1
    
    green_exact_applied = (
        request.filters.dominant_hue_strict and 
        "green" in [h.lower() for h in request.filters.dominant_hues]
    )
    
    red_exact_applied = (
        request.filters.dominant_hue_strict and 
        "red" in [h.lower() for h in request.filters.dominant_hues]
    )
    
    red_expanded_applied = (
        not request.filters.dominant_hue_strict and 
        len(request.filters.dominant_hues) == 1 and
        request.filters.dominant_hues[0].lower() == "red"
    )
    
    debug_filters = {
        "strict_exclude_received": request.filters.strict_exclude,
        "dominant_hue_strict_received": request.filters.dominant_hue_strict,
        "dominant_hues_received": request.filters.dominant_hues,
        "total_candidates_before_filters": total_candidates_before_filters,
        "total_after_hue_gate": total,
        "removed_by_hue": filter_debug.get("removed_by_hue", 0),
        "admin_suppressed": suppressed_count,
        "total_returned": len(results),
        "rescue_enabled": use_rescue,
        "rescue_count": rescue_count,
        "bucket_counts": bucket_counts,
        "matched_before_suppression": matched_before_suppression,
        "shown_after_suppression": shown_after_suppression,
        "suppression_applied": suppression_applied,
        "include_suppressed": request.include_suppressed,
        "timing_ms": {
            "db_total": round(db_ms, 1),
            "phase_b_ms": round(t_phase_b_ms, 1),
            "filter": round(filter_ms, 1),
            "score": round(score_ms, 1),
            "sort": round(sort_ms, 1),
            "facet": round(facet_ms, 1),
            "hydrate": round(hydrate_ms, 1),
            "serialize": round(serialize_ms, 1),
            "total": round(latency, 1),
            "db_substages": dict(
                **(_last_db_substage_timing or {}),
                phase_b_mode=phase_b_mode,
                phase_b_stones_hydrated=phase_b_stones_hydrated,
                phase_b_hydrate_ms=round(t_phase_b_ms, 2),
            ),
        },
   }
    
    return SearchResponse(
        results=results,
        total=total,
        query=request.q,
        image_used=False,
        latency_ms=latency,
        search_request_id=search_request_id,
        debug_filters=debug_filters,
        pattern_family_facets=pf_facets,
        pattern_family_unlabeled=pf_unlabeled,
        base_color_facets=bc_facets,
        meta=_build_search_meta(smart_tier_counts, vendor_cap_meta),
    )






















@router.post("/search-v3")
async def search_stones_v3(request: SearchRequest):
    """
    Search v3 — same as /api/search but enriches each result with v3 pattern
    family predictions (prototype pool M4b + selective U↔L skeleton reranking).

    Extra fields per stone:
      pattern_family_v3        -- predicted family (or None if no embedding)
      pattern_family_v3_source -- "prototype" | "ul_rerank" | None
    """

    response = await search_stones(request)
    result_dict = response.model_dump()
    enrich_with_v3(result_dict["results"])
    return result_dict




# Similar explorer endpoints and helpers removed - moved to similar module
