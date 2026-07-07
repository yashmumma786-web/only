"""
Active Stock Correction API Router
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
from src.modules.taxonomy import repository as tag_storage
from src.utils.cache_manager import invalidate_stone_cache

router = APIRouter()
log = logging.getLogger("tag_router")


def _parse_bool_field(value) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"has_bold_accent_vein must be a boolean (true/false), got {type(value).__name__}: {value!r}"
    )


def _coerce_corrected_value(field_name: str, raw_value):
    if raw_value is None:
        return None
    if field_name == "has_bold_accent_vein":
        return _parse_bool_field(raw_value)
    if isinstance(raw_value, str):
        return raw_value
    return str(raw_value)


PATTERN_SOURCE_TIER_VALUES = frozenset({"override", "manual", "aggregated", "ai", "missing"})


def _resolve_pattern_source_tier(conn, company_stone_id: str):
    """Pattern Correction Phase 1 — server-authoritative cascade snapshot.

    Mirrors search_router._get_effective_pattern_family_for_filter precedence
    (override → manual stock_tags → aggregated → ai → missing).
    Returns (resolved_value, source_tier) where source_tier is one of
    PATTERN_SOURCE_TIER_VALUES. Any deviation from the allowed set is logged
    as a warning and coerced to 'missing' before being returned/persisted.
    """
    field = "pattern_family"
    tier = "missing"
    value = None
    row = conn.execute(
        "SELECT override_value FROM stock_overrides "
        "WHERE company_stone_id = ? AND field_name = ?",
        (company_stone_id, field),
    ).fetchone()
    if row and row["override_value"]:
        value, tier = row["override_value"], "override"
    if value is None:
        row = conn.execute(
            "SELECT aggregated_value FROM stock_aggregated "
            "WHERE company_stone_id = ? AND field_name = ?",
            (company_stone_id, field),
        ).fetchone()
        if row and row["aggregated_value"]:
            value, tier = row["aggregated_value"], "aggregated"
    if value is None:
        row = conn.execute(
            "SELECT tag_value FROM stock_tags_ai "
            "WHERE company_stone_id = ? AND tag_name = ?",
            (company_stone_id, field),
        ).fetchone()
        if row and row["tag_value"]:
            value, tier = row["tag_value"], "ai"
    if tier not in PATTERN_SOURCE_TIER_VALUES:
        log.warning(
            "pattern_source_tier_out_of_contract stone=%s tier=%r — coercing to 'missing'",
            company_stone_id, tier,
        )
        return None, "missing"
    return value, tier


@router.get("/api/admin/correction/stock/{company_stone_id}/pattern-evidence")
async def correction_pattern_evidence(company_stone_id: str):
    """Pattern Correction Phase 1 — evidence payload for the popover.

    Returns the AI value, aggregated value, effective resolved value, and
    source-tier label so the admin popover can render the required evidence
    rows server-authoritatively (no client-side guessing).
    """
    with tag_storage.get_connection() as conn:
        ai_row = conn.execute(
            "SELECT tag_value FROM stock_tags_ai "
            "WHERE company_stone_id = ? AND tag_name = 'pattern_family'",
            (company_stone_id,),
        ).fetchone()
        agg_row = conn.execute(
            "SELECT aggregated_value FROM stock_aggregated "
            "WHERE company_stone_id = ? AND field_name = 'pattern_family'",
            (company_stone_id,),
        ).fetchone()
        effective, source_tier = _resolve_pattern_source_tier(conn, company_stone_id)
    return JSONResponse({
        "company_stone_id": company_stone_id,
        "ai_pattern_family": ai_row["tag_value"] if ai_row else None,
        "agg_pattern_family": agg_row["aggregated_value"] if agg_row else None,
        "effective_pattern_family": effective,
        "source_tier": source_tier,
    })




@router.post("/api/admin/correction/stock")
async def correction_stock(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    company_stone_id = data.get("company_stone_id")
    field_name = data.get("field_name")
    override_value = data.get("override_value")
    session_user = request.session.get("user") if hasattr(request, "session") else None
    reviewer_id = session_user or data.get("reviewer_id", "admin")
    action_type = data.get("action_type", "CORRECTION")

    # Pattern Correction Phase 1 — additive payload fields (Task #261).
    reviewer_note = data.get("reviewer_note")
    session_id = data.get("session_id")
    surface = data.get("surface")

    if not company_stone_id or not field_name:
        raise HTTPException(status_code=400, detail="company_stone_id and field_name are required")

    if action_type != "CORRECTION":
        raise HTTPException(status_code=422, detail=f"Invalid action_type: {action_type}")

    if surface is not None and surface not in tag_storage.PATTERN_CORRECTION_SURFACES:
        raise HTTPException(status_code=422, detail=f"Invalid surface: {surface}")

    try:
        tag_storage._validate_taxonomy_field(field_name)
        value = _coerce_corrected_value(field_name, override_value)
        if value is not None:
            tag_storage._validate_taxonomy_value(field_name, value)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    existing_agg = None
    ai_value_at_correction = None
    source_tier_before = None
    with tag_storage.get_connection() as conn:
        agg_row = conn.execute(
            "SELECT aggregated_value FROM stock_aggregated WHERE company_stone_id = ? AND field_name = ?",
            (company_stone_id, field_name)
        ).fetchone()
        if agg_row:
            existing_agg = agg_row["aggregated_value"]

        if field_name == "pattern_family":
            ai_row = conn.execute(
                "SELECT tag_value FROM stock_tags_ai "
                "WHERE company_stone_id = ? AND tag_name = 'pattern_family'",
                (company_stone_id,),
            ).fetchone()
            if ai_row and ai_row["tag_value"]:
                ai_value_at_correction = ai_row["tag_value"]
            _, source_tier_before = _resolve_pattern_source_tier(conn, company_stone_id)

    tag_storage.write_stock_override(
        company_stone_id=company_stone_id,
        field_name=field_name,
        value=value,
        reviewer_id=reviewer_id
    )

    if value is None:
        stored_value_str = None
    elif field_name in tag_storage.TAXONOMY_BOOL_FIELDS and isinstance(value, bool):
        stored_value_str = tag_storage._serialize_taxonomy_bool(value)
    else:
        stored_value_str = str(value)

    # Materialise the override into stock_aggregated immediately so search results
    # reflect the change on the next query without waiting for a full taxonomy backfill.
    # IMPORTANT: stock_aggregated is a search-visibility cache only.
    # The authoritative human override truth lives in:
    #   stock_overrides  (this write, above)
    #   correction_log   (written below)
    #   correction_candidates  (written below when action_type == CORRECTION)
    tag_storage.write_stock_aggregated(
        company_stone_id=company_stone_id,
        field_name=field_name,
        aggregated_value=stored_value_str,
    )

    # Pattern Correction Phase 1 — only snapshot AI/agg/source_tier for
    # pattern_family corrections; other field corrections persist NULL.
    if field_name != "pattern_family":
        ai_value_at_correction = None
        source_tier_before = None

    tag_storage.log_correction(
        stock_id=company_stone_id,
        image_id=None,
        field_name=field_name,
        computed_value=existing_agg,
        computed_confidence=None,
        corrected_value=stored_value_str,
        action_type=action_type,
        scope_type="STOCK",
        reviewer_id=reviewer_id,
        model_version=None,
        reason_code=None,
        reviewer_note=(str(reviewer_note).strip() if reviewer_note else None),
        source_tier=source_tier_before,
        ai_value_at_correction=ai_value_at_correction,
        agg_value_at_correction=(existing_agg if field_name == "pattern_family" else None),
        session_id=session_id,
        surface=surface,
    )


    session_correction_count = None
    if session_id:
        with tag_storage.get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM correction_log "
                "WHERE session_id = ? AND DATE(timestamp) = DATE('now')",
                (session_id,),
            ).fetchone()
            session_correction_count = int(row["c"]) if row else 0

    invalidate_stone_cache()
    return JSONResponse({
        "success": True,
        "company_stone_id": company_stone_id,
        "field_name": field_name,
        "action_type": action_type,
        "corrected_value": stored_value_str,
        "source_tier_before": source_tier_before,
        "session_correction_count": session_correction_count,
    })





