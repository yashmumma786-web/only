"""
Similar Stone Explorer — business logic module.

Lives in ``src.modules.similar`` — standalone domain module.
Do NOT import from ``src.modules.search``.

Public entry points (consumed via similar.services):
  resolve()              — 3-section similar-stones resolver
  log_event()            — write to similar_explorer.db event log

Dependency rules:
  ✅ src.modules.taxonomy.services  (color info, sibling votes)
  ✅ src.modules.ml_inference.services  (embeddings)
  ✅ src.utils.cache_manager  (color-cache invalidation hook)
  ❌ src.modules.search.*  — never import from here

Resolves an anchor stone plus exactly three sections of similar candidates:

  Section A — Cross-vendor inferred (same colour family STRICT, fuzzy name).
  Section B — Same commercial profile (same colour + same pattern, visually
              confirmed).
  Section C — AI visual similarity (loose colour gate, no pattern gate).

Mode A: the public output shape is ``{anchor, sections, meta}``.
"""

import logging
import sqlite3
import threading
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.utils.cache_manager import register_color_cache_invalidator

import numpy as np

from src.modules.taxonomy import services as taxonomy_client
from src.modules.ml_inference import services as ml_svc
# get_retrieval_color_family lives in taxonomy — pure color math.
# Both search/name_search and this module import it from there.
from src.utils.color_rules import (
    get_retrieval_color_family,
    _COLOR_VETO_BUCKET,
    _VETO_PAIRS,
    _COLOR_NORMALIZE,
    _ADJACENT_FAMILIES,
    _RETRIEVAL_FAMILY_VETO_PAIRS,
    LIGHT_COOL_DISTANCE_PAIRS,
    LIGHT_COOL_DISTANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

def get_similar_explorer_db_path() -> Path:
    """Return path to similar_explorer.db — owned by similar."""
    return Path(os.environ.get("SIMILAR_EXPLORER_DB_PATH"))
EXPLORER_DB = get_similar_explorer_db_path()

VISUAL_LIMIT = 10


# ---------------------------------------------------------------------------
# DB-backed color caches — similar module owns these.
# They cache calls to taxonomy.services.get_stone_color_info() made by
# _resolve_override_base_color() and _resolve_ai_color_fallback() below.
# These are NOT re-exported; they are internal to the resolver.
# ---------------------------------------------------------------------------

# Task #369 — defensive override → aggregated DB fallback for the
# anchor/candidate ``base_color`` reader.  In-process cache keyed by
# ``company_stone_id`` keeps the per-call SQLite cost O(1) amortised
# inside the resolver loop.

_STONE_OVERRIDE_BASE_COLOR_CACHE: Dict[str, Optional[str]] = {}
_STONE_OVERRIDE_BASE_COLOR_LOCK = threading.RLock()


def _resolve_override_base_color(cid: Optional[str]) -> Optional[str]:
    """Return stock_overrides.base_color then stock_aggregated.base_color for ``cid``.
    Calls taxonomy.services only — no direct DB access.
    """
    if not cid or not isinstance(cid, str):
        return None
    with _STONE_OVERRIDE_BASE_COLOR_LOCK:
        if cid in _STONE_OVERRIDE_BASE_COLOR_CACHE:
            return _STONE_OVERRIDE_BASE_COLOR_CACHE[cid]
    value: Optional[str] = None
    try:
        info = taxonomy_client.get_stone_color_info(cid)
        value = info.get("base_color")
    except Exception:
        logger.exception("override base_color lookup failed for %s", cid)
        value = None
    with _STONE_OVERRIDE_BASE_COLOR_LOCK:
        _STONE_OVERRIDE_BASE_COLOR_CACHE[cid] = value
    return value


_STONE_AI_COLOR_CACHE: Dict[str, Optional[str]] = {}
_STONE_AI_COLOR_LOCK = threading.RLock()


def _resolve_ai_color_fallback(cid: Optional[str]) -> Optional[str]:
    """Final fallback — reads ai_color from taxonomy when display_tags lacks color."""
    if not cid or not isinstance(cid, str):
        return None
    with _STONE_AI_COLOR_LOCK:
        if cid in _STONE_AI_COLOR_CACHE:
            return _STONE_AI_COLOR_CACHE[cid]
    value: Optional[str] = None
    try:
        info = taxonomy_client.get_stone_color_info(cid)
        value = info.get("ai_color")
    except Exception:
        logger.exception("ai colour fallback lookup failed for %s", cid)
        value = None
    with _STONE_AI_COLOR_LOCK:
        _STONE_AI_COLOR_CACHE[cid] = value
    return value


def _invalidate_color_caches(cid: str) -> None:
    """Drop ``cid`` from both colour-resolution caches.

    Registered with cache_manager at module import time so every
    write_stock_override for base_color or vein_colour evicts the
    stale entry before the next Similar Stone Explorer request runs.
    Single-process only — if multiple workers are used, a pubsub
    broadcast is required.
    """
    if not cid or not isinstance(cid, str):
        return
    with _STONE_OVERRIDE_BASE_COLOR_LOCK:
        _STONE_OVERRIDE_BASE_COLOR_CACHE.pop(cid, None)
    with _STONE_AI_COLOR_LOCK:
        _STONE_AI_COLOR_CACHE.pop(cid, None)


def _get_stone_veto_color(stone: Dict) -> Optional[str]:
    """Canonical color reader for anchor and candidate stones.

    Fallback chain: stock_overrides.base_color → stock_aggregated.base_color
    → display_tags[base_color] → main_color → ai fallback via taxonomy.
    """
    color = (
        _resolve_override_base_color(stone.get("company_stone_id"))
        or _get_tag_value(stone, "base_color")
        or _get_tag_value(stone, "main_color")
        or _resolve_ai_color_fallback(stone.get("company_stone_id"))
    )
    if not color or not isinstance(color, str):
        return None
    color = color.strip().lower()
    color = _COLOR_NORMALIZE.get(color, color)
    if color in ("multi", "multicolor"):
        return None
    return color if color else None


def _get_veto_bucket(color: Optional[str]) -> Optional[str]:
    if not color:
        return None
    return _COLOR_VETO_BUCKET.get(color)


def _is_hard_color_veto(anchor_color: Optional[str], candidate_color: Optional[str]) -> bool:
    a_bucket = _get_veto_bucket(anchor_color)
    c_bucket = _get_veto_bucket(candidate_color)
    if not a_bucket or not c_bucket:
        return False
    if a_bucket == c_bucket:
        return False
    return frozenset({a_bucket, c_bucket}) in _VETO_PAIRS




def _get_color_tier(
    anchor_family: Optional[str],
    candidate_family: Optional[str],
    anchor_stone: Optional[Dict] = None,
    cand_stone: Optional[Dict] = None,
) -> int:
    """Return 1 (same), 2 (adjacent), or 3 (allowed cross-family / unknown).


    LIGHT_COOL lightness-distance safety gate: within the adjacency pairs
    ``{LIGHT_COOL, DARK_COOL}`` and ``{LIGHT_COOL, DARK_COOL_CHROMATIC}``
    (collected in ``LIGHT_COOL_DISTANCE_PAIRS``), if both stones expose a
    numeric ``stone_lightness_mean`` and the L* gap exceeds
    ``LIGHT_COOL_DISTANCE_THRESHOLD``, demote to tier 3.  Missing on
    either side keeps the adjacency tier 2 (safe default, no regression).
    All other adjacency pairs are unaffected.
    """
    if not anchor_family or not candidate_family:
        return 3
    if anchor_family == candidate_family:
        return 1
    pair = frozenset({anchor_family, candidate_family})
    if pair in _ADJACENT_FAMILIES:
        if pair in LIGHT_COOL_DISTANCE_PAIRS and anchor_stone is not None and cand_stone is not None:
            a_L = _get_tag_value(anchor_stone, "stone_lightness_mean")
            c_L = _get_tag_value(cand_stone, "stone_lightness_mean")
            try:
                if a_L is not None and c_L is not None and abs(float(a_L) - float(c_L)) > LIGHT_COOL_DISTANCE_THRESHOLD:
                    return 3
            except (TypeError, ValueError):
                pass
        return 2
    return 3


def _is_anchor_relaxed(anchor_stone: Dict) -> bool:
    """Return True when the anchor has no clear single color family (multi, unknown).
    In relaxed mode, color-tier binning is skipped and pure similarity ranking applies."""
    return get_retrieval_color_family(anchor_stone) is None


# get_retrieval_color_family is imported from taxonomy.color_rules.
# It is the canonical implementation — do NOT add a local version here.



def _is_retrieval_family_veto(a_family: Optional[str], c_family: Optional[str]) -> bool:
    """Family-level veto using retrieval families. Applied after bucket veto."""
    if not a_family or not c_family:
        return False
    return frozenset({a_family, c_family}) in _RETRIEVAL_FAMILY_VETO_PAIRS




_explorer_lock = threading.Lock()
_explorer_conn: Optional[sqlite3.Connection] = None

_CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS similar_explorer_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    anchor_id TEXT,
    candidate_id TEXT,
    section TEXT,
    tier TEXT,
    role TEXT,
    search_version TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
)
"""


def _get_explorer_conn() -> sqlite3.Connection:
    """Return (or create) a persistent WAL-mode connection for event logging.

    Must be called inside _explorer_lock.
    If the connection is found to be closed/broken, it is recreated.
    """
    global _explorer_conn
    if _explorer_conn is not None:
        try:
            _explorer_conn.execute("SELECT 1")
            return _explorer_conn
        except Exception:
            try:
                _explorer_conn.close()
            except Exception:
                pass
            _explorer_conn = None

    EXPLORER_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(EXPLORER_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_CREATE_EVENTS_TABLE)
    conn.commit()
    _explorer_conn = conn
    return conn


def log_event(
    event_type: str,
    anchor_id: str = "",
    candidate_id: str = "",
    section: str = "",
    tier: str = "",
    role: str = "",
    search_version: str = "",
    detail: str = "",
):
    try:
        with _explorer_lock:
            conn = _get_explorer_conn()
            conn.execute(
                "INSERT INTO similar_explorer_events "
                "(event_type, anchor_id, candidate_id, section, tier, role, search_version, detail, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (event_type, anchor_id, candidate_id, section, tier, role, search_version, detail,
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("similar_explorer log_event failed: %s", exc)


def _get_tag_value(stone: Dict, key: str, default=None):
    tags = stone.get("display_tags", {})
    entry = tags.get(key)
    if entry is None:
        return default
    if isinstance(entry, dict):
        return entry.get("value", default)
    return entry


def _make_taxonomy(stone: Dict) -> Dict[str, Any]:
    return {
        # Task #312 — colour chip fallback chain extended to ``main_color`` so
        # cards never silently render a null colour chip when only the AI
        # ``main_color`` tag is populated.
        # Task #369 — chip reader prefixed with the same override → aggregated
        # DB fallback used by ``_get_stone_veto_color`` so the chip and the
        # retrieval-family gate share one precedence and cannot drift even if
        # the upstream stone-loader hydration changes.
        "base_color": (
            _resolve_override_base_color(stone.get("company_stone_id"))
            or _get_tag_value(stone, "base_color")
            or _get_tag_value(stone, "main_color")
            or _resolve_ai_color_fallback(stone.get("company_stone_id"))
        ),
        "pattern_family": _get_tag_value(stone, "pattern_family"),
        "network_thickness": _get_tag_value(stone, "network_thickness"),
        "has_bold_accent_vein": _get_tag_value(stone, "has_bold_accent_vein"),
    }


def _stone_to_anchor(stone: Dict) -> Dict[str, Any]:
    # Task #366 — expose the per-anchor refinement field values so the
    # frontend can compute disabled-state for buttons whose tag is absent
    # on this anchor.  Read raw ``display_tags[key]["value"]`` only —
    # no normalisation, no transformation.  Missing keys are emitted
    # as ``None`` (never omitted) so the frontend can rely on the shape.
    _tags = stone.get("display_tags", {}) or {}

    def _raw(key: str):
        entry = _tags.get(key)
        if entry is None:
            return None
        if isinstance(entry, dict):
            v = entry.get("value")
            if v in (None, ""):
                return None
            return v
        if entry in ("", None):
            return None
        return entry

    return {
        "id": stone.get("company_stone_id", ""),
        "batch_id": stone.get("batch_id", ""),
        "image_url": stone.get("thumbnail_url", ""),
        "display_name": stone.get("stone_name", "") or stone.get("batch_id", ""),
        "source": stone.get("vendor_name", ""),
        "vendor": stone.get("vendor_name", ""),
        "status": "active",
        "taxonomy": _make_taxonomy(stone),
        "buyer_reason_chips": ["Selected Stone"],
        "admin_reason_chips": ["Selected Stone"],
        # Task #366 — buyer-refinement field exposure.  Additive only.
        "refinement_fields": {
            "stone_lightness_mean": _raw("stone_lightness_mean"),
            "dominant_tone":        _raw("dominant_tone"),
            "plain":                _raw("plain"),
            "blotchy":              _raw("blotchy"),
            "exotic":               _raw("exotic"),
            "vein_colour":          _raw("vein_colour"),
            "network_thickness":    _raw("network_thickness"),
            "pattern_family":       _raw("pattern_family"),
            # Task #361/#372 — LAB a/b/chroma channels for precise tone &
            # chromaticity refinement buttons.  Missing → null.
            "lab_a":                _raw("lab_a"),
            "lab_b":                _raw("lab_b"),
            "lab_chroma":           _raw("lab_chroma"),
            # Task #384 — new slider / chip controls need second_share
            # (Two-tone slider), drama_score + drama_band (Drama chip).
            # drama_score is also surfaced on StoneResult for the search
            # API but the explorer reads it through display_tags.
            "second_share":         _raw("second_share"),
            "drama_score":          _raw("drama_score"),
            "drama_band":           _raw("drama_band"),
        },
    }


def _make_candidate(
    stone: Dict,
    tier: str,
    buyer_chips: List[str],
    admin_chips: List[str],
    debug: Optional[Dict] = None,
    similarity: Optional[float] = None,
) -> Dict[str, Any]:
    candidate = {
        "id": stone.get("company_stone_id", ""),
        "batch_id": stone.get("batch_id", ""),
        "image_url": stone.get("thumbnail_url", ""),
        "display_name": stone.get("stone_name", "") or stone.get("batch_id", ""),
        "source": stone.get("vendor_name", ""),
        "status": "active",
        "tier": tier,
        "taxonomy": _make_taxonomy(stone),
        "buyer_reason_chips": buyer_chips,
        "admin_reason_chips": admin_chips,
    }
    if debug:
        candidate["debug"] = debug
    if similarity is not None:
        candidate["similarity"] = round(similarity, 4)
    return candidate


# ---------------------------------------------------------------------------
# Commercial-profile scoring helpers
# ---------------------------------------------------------------------------

def _section_c_pattern_tier(anchor_pf: Optional[str], cand_pf: Optional[str]) -> int:
    """Task #335 — Section C pattern-aware rerank tier.

    Used by ``_resolve_ai_suggestions`` to soft-rerank visual candidates
    after the colour gate has passed them.  Composes with the existing
    ``color_tier`` (colour ordering invariant) — pattern is secondary to
    colour in the non-relaxed sort, primary in the relaxed sort (where
    ``color_tier`` is absent by design).

    Tier 1: candidate has the same ``pattern_family`` as the anchor (best
            — promoted to the top of the colour-ordered group).
    Tier 2: candidate has a ``pattern_family`` that differs from the
            anchor's (mid — kept eligible, ranked below tier 1).
    Tier 3: candidate has no ``pattern_family`` at all (soft penalty —
            sorted last among tied colour but **not** dropped, since
            Section C is the broad safety-net section and pattern
            coverage is only ~65 %).

    Anchor-pattern-missing edge case: when the anchor itself has no
    ``pattern_family``, no candidate can be tier 1.  Candidates that
    have any ``pattern_family`` become tier 2; candidates without
    become tier 3.  This preserves a meaningful ordering on
    sparse-anchor anchors (the alternative — collapsing every candidate
    to tier 2 — would silently disable the rerank for those anchors).
    """
    if not cand_pf:
        return 3
    if not anchor_pf:
        return 2
    if anchor_pf == cand_pf:
        return 1
    return 2




def _section_c_tone_tier(anchor: Dict, cand: Dict) -> int:
    """Task #348 — Section C tone tier (1=match, 2=mismatch, 3=missing-either)."""
    a_raw = _get_tag_value(anchor, "dominant_tone")
    c_raw = _get_tag_value(cand, "dominant_tone")
    a = a_raw.strip().lower() if isinstance(a_raw, str) and a_raw.strip() else None
    c = c_raw.strip().lower() if isinstance(c_raw, str) and c_raw.strip() else None
    if a is None or c is None:
        return 3
    return 1 if a == c else 2


def _section_c_network_tier(anchor: Dict, cand: Dict) -> int:
    """Task #348 — Section C network tier (1=match incl. both na, 2=mismatch, 3=missing)."""
    a_raw = _get_tag_value(anchor, "network_thickness")
    c_raw = _get_tag_value(cand, "network_thickness")
    a = a_raw.strip().lower() if isinstance(a_raw, str) and a_raw.strip() else None
    c = c_raw.strip().lower() if isinstance(c_raw, str) and c_raw.strip() else None
    if a is None or c is None:
        return 3
    return 1 if a == c else 2




def _section_c_lightness_tier(anchor: Dict, cand: Dict) -> int:
    """Task #355 — Section C lightness tier (1=match, 2=mismatch, 3=missing-either)."""
    a_raw = _get_tag_value(anchor, "stone_lightness")
    c_raw = _get_tag_value(cand, "stone_lightness")
    a = a_raw.strip().lower() if isinstance(a_raw, str) and a_raw.strip() else None
    c = c_raw.strip().lower() if isinstance(c_raw, str) and c_raw.strip() else None
    if a is None or c is None:
        return 3
    return 1 if a == c else 2




def _section_c_vein_colour_tier(anchor: Dict, cand: Dict) -> int:
    """Task #358 — Section C vein-colour tier (1=match, 2=mismatch, 3=missing-either)."""
    a_raw = _get_tag_value(anchor, "vein_colour")
    c_raw = _get_tag_value(cand, "vein_colour")
    a = a_raw.strip().lower() if isinstance(a_raw, str) and a_raw.strip() else None
    c = c_raw.strip().lower() if isinstance(c_raw, str) and c_raw.strip() else None
    if a is None or c is None:
        return 3
    return 1 if a == c else 2


_DRAMA_BAND_VALUES = ("plain", "moderate", "dramatic")

def _drama_band_for_stone(stone: Dict) -> Optional[str]:
    raw = _get_tag_value(stone, "drama_band")
    if not isinstance(raw, str):
        return None
    v = raw.strip().lower()
    return v if v in _DRAMA_BAND_VALUES else None




def _section_c_drama_tier(anchor: Dict, cand: Dict) -> int:
    """Section C drama-band tier.

    tier 1 = same band
    tier 2 = adjacent bands (plain↔moderate or moderate↔dramatic)
    tier 3 = plain↔dramatic OR missing on either side
    """
    a = _drama_band_for_stone(anchor)
    c = _drama_band_for_stone(cand)
    if a is None or c is None:
        return 3
    if a == c:
        return 1
    pair = frozenset({a, c})
    if pair == frozenset({"plain", "dramatic"}):
        return 3
    return 2




def _compute_similarities_for_anchor(
    anchor_id: str,
    anchor_stone: Dict,
    stones: List[Dict],
) -> Dict[str, float]:
    """Load anchor embedding and compute cosine similarities against all stones.

    Uses the shared in-memory embedding cache (embedding_cache.py) backed by
    foundation.db.  The .npy fallback path is intentionally removed — all
    embeddings live in foundation.db.

    Returns dict {company_stone_id: similarity_score} with scores in [0, 1].
    Returns empty dict if anchor embedding not found (fallback: no visual matches).
    """
    embeddings = ml_svc.load_all_embeddings()
    query_emb = embeddings.get(anchor_id)
    if query_emb is None:
        return {}

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    similarities: Dict[str, float] = {}
    for s in stones:
        sid = s.get("company_stone_id", "")
        s_emb = embeddings.get(sid)
        if s_emb is not None:
            s_norm = s_emb / (np.linalg.norm(s_emb) + 1e-8)
            sim = float(np.dot(query_norm.flatten(), s_norm.flatten()))
            similarities[sid] = max(0.0, min(1.0, (sim + 1) / 2))
    return similarities


def _resolve_ai_suggestions(
    anchor_id: str,
    anchor_stone: Dict,
    stones: List[Dict],
    exclude_ids: set,
    similarities: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], int]:
    """Resolve Section C — broad visual / CLIP candidates.

    Loose colour gate (hard veto + retrieval-family veto) but no pattern
    requirement — pattern-missing candidates that were excluded from
    Section B remain eligible here.  Returns ``(visual_results, vetoed_count)``.
    """
    anchor_color = _get_stone_veto_color(anchor_stone)
    anchor_relaxed = _is_anchor_relaxed(anchor_stone)
    anchor_family = get_retrieval_color_family(anchor_stone) if not anchor_relaxed else None
    # Task #335 — pattern-aware rerank.  Anchor pattern_family may be None;
    # _section_c_pattern_tier handles that case explicitly.
    anchor_pf = _get_tag_value(anchor_stone, "pattern_family")

    if similarities is None:
        similarities = {}

    # Task #335/#348/#355/#358 + drama-band — tuple is (stone, sim,
    # color_tier, pattern_tier, tone_tier, lightness_tier,
    # vein_colour_tier, drama_tier, network_tier).  drama_tier is
    # inserted between vein_colour_tier and network_tier per the
    # Two-Tone Drama Score brief.  The soft tiers (1=match, 2=mismatch,
    # 3=missing) compose with pattern_tier in the sort key below as
    # soft tiebreakers.
    visual_candidates: List[Tuple[Dict, float, int, int, int, int, int, int, int]] = []
    visual_vetoed = 0
    if similarities:
        for s in stones:
            sid = s.get("company_stone_id", "")
            if sid == anchor_id or sid in exclude_ids:
                continue
            sim = similarities.get(sid, 0.0)
            if sim <= 0:
                continue
            cand_color = _get_stone_veto_color(s)
            if _is_hard_color_veto(anchor_color, cand_color):
                visual_vetoed += 1
                logger.debug(
                    "color_veto anchor=%s cand=%s anchor_color=%s cand_color=%s bucket_pair=%s/%s",
                    anchor_id, sid, anchor_color, cand_color,
                    _get_veto_bucket(anchor_color), _get_veto_bucket(cand_color),
                )
                continue
            cand_family = get_retrieval_color_family(s)
            if _is_retrieval_family_veto(anchor_family, cand_family):
                visual_vetoed += 1
                logger.debug(
                    "retrieval_family_veto anchor=%s cand=%s anchor_family=%s cand_family=%s",
                    anchor_id, sid, anchor_family, cand_family,
                )
                continue
            color_tier = _get_color_tier(anchor_family, cand_family, anchor_stone, s) if not anchor_relaxed else 1
            cand_pf = _get_tag_value(s, "pattern_family")
            pattern_tier = _section_c_pattern_tier(anchor_pf, cand_pf)
            # Task #348 — soft tone / network tiers (additive).  Missing
            # on either side maps to tier 3 (never penalises).
            tone_tier = _section_c_tone_tier(anchor_stone, s)
            network_tier = _section_c_network_tier(anchor_stone, s)
            # Task #355 — soft lightness tier (1=match, 2=mismatch,
            # 3=missing-either).  Ordered AFTER tone_tier and BEFORE
            # network_tier in the Section C sort key so a same-band
            # candidate outranks a wrong-band one when tone matches both.
            lightness_tier = _section_c_lightness_tier(anchor_stone, s)
            # Task #358 — soft vein_colour tier inserted between
            # lightness_tier and network_tier in the Section C sort key
            # so a same-vein-colour candidate outranks a wrong-vein
            # candidate when colour, pattern, tone and lightness all tie.
            vein_colour_tier = _section_c_vein_colour_tier(anchor_stone, s)
            # Two-Tone Drama Score — soft drama-band tier inserted
            # between vein_colour_tier and network_tier in the Section
            # C sort key so a same-drama-band candidate outranks a
            # plain-vs-dramatic candidate when colour, pattern, tone,
            # lightness and vein_colour all tie.
            drama_tier = _section_c_drama_tier(anchor_stone, s)
            visual_candidates.append(
                (s, sim, color_tier, pattern_tier, tone_tier, lightness_tier, vein_colour_tier, drama_tier, network_tier)
            )

        # Task #335 — pattern-aware rerank.  In non-relaxed mode the existing
        # colour ordering invariant is preserved as the primary key
        # (color_tier ASC), pattern_tier is inserted as the secondary key,
        # and -similarity remains the tertiary tiebreaker.  In relaxed mode
        # color_tier is absent from the sort by design, so pattern_tier
        # becomes the primary key with -similarity as the tiebreaker.
        # Pattern-missing candidates (tier 3) are NOT dropped — Section C is
        # the broad safety-net section and pattern coverage is incomplete.
        # Task #348/#355/#358 — sort keys are color_tier (x[2]),
        # pattern_tier (x[3]), tone_tier (x[4]), lightness_tier (x[5]),
        # vein_colour_tier (x[6]), network_tier (x[7]) — vein_colour is
        # inserted between lightness and network per the #358 brief —
        # with -similarity as the final tiebreaker.  Relaxed mode omits
        # color_tier by design.
        if anchor_relaxed:
            visual_candidates.sort(key=lambda x: (x[3], x[4], x[5], x[6], x[7], x[8], -x[1]))
        else:
            visual_candidates.sort(key=lambda x: (x[2], x[3], x[4], x[5], x[6], x[7], x[8], -x[1]))

        # Cap AFTER the pattern-aware rerank, per the #335 brief.
        visual_candidates = visual_candidates[:VISUAL_LIMIT]

    visual_results: List[Dict] = []
    for s, sim, color_tier, pattern_tier, tone_tier, lightness_tier, vein_colour_tier, drama_tier, network_tier in visual_candidates:
        buyer_chips: List[str] = []
        admin_chips = ["CLIP Match"]

        s_tax = _make_taxonomy(s)
        if s_tax.get("pattern_family"):
            buyer_chips.append("Similar Pattern")
        if s_tax.get("base_color"):
            buyer_chips.append("Similar Colors")
        if not buyer_chips:
            buyer_chips.append("Close Visual Match")

        # Task #335 — surface pattern_tier in debug so the validation
        # harness and downstream tooling can read the rerank decision
        # directly from the payload (mirrors color_tier).
        debug = {
            "clip_similarity": round(sim, 4),
            "color_tier": color_tier,
            "pattern_tier": pattern_tier,
            # Task #348 — tone / network rerank tiers surfaced on each card.
            "tone_tier": tone_tier,
            "network_tier": network_tier,
            # Task #355 — lightness rerank tier surfaced on each card.
            "lightness_tier": lightness_tier,
            # Task #358 — vein_colour rerank tier surfaced on each card.
            "vein_colour_tier": vein_colour_tier,
            # Two-Tone Drama Score — drama-band rerank tier surfaced
            # on each card (1=same band, 2=adjacent, 3=plain↔dramatic
            # OR missing on either side).
            "drama_tier": drama_tier,
        }
        visual_results.append(_make_candidate(s, "ai_visual", buyer_chips, admin_chips, debug=debug, similarity=sim))

    return visual_results, visual_vetoed


def _empty_result(anchor_id: str, mode: str, search_version: str) -> Dict[str, Any]:
    """Return the not-found stub in the new 3-section shape."""
    return {
        "anchor": {
            "id": anchor_id, "display_name": "[Not Found]", "status": "unknown",
            "image_url": "", "source": "", "vendor": "", "batch_id": "",
            "taxonomy": {}, "buyer_reason_chips": [], "admin_reason_chips": [],
        },
        "sections": {
            "section_a": [],
            "section_b": [],
            "section_c": [],
        },
        "meta": {
            "mode": mode,
            "search_version": search_version,
            "anchor_color": None,
            "anchor_color_family": None,
            "anchor_dominant_hue": None,
            "anchor_relaxed": True,
            "section_a_count": 0,
            "section_b_count": 0,
            "section_c_count": 0,
            "section_a_vetoed_by_color": 0,
            "section_b_vetoed_by_color": 0,
            "section_c_vetoed_by_color": 0,
            "section_b_gate": "strict",
        },
    }


def resolve(
    anchor_id: str,
    stones: List[Dict],
    mode: str = "buyer",
    search_version: str = "v2",
) -> Dict[str, Any]:
    """Resolve the 3-section similar-stones contract for an anchor.

    Returns ``{anchor, sections: {section_a, section_b, section_c}, meta}``.
    See module docstring for section semantics.  Mode A — the legacy
    ``linked`` and ``ai_suggestions`` keys are NOT emitted.
    """
    anchor_stone = None
    for s in stones:
        if s.get("company_stone_id") == anchor_id or s.get("batch_id") == anchor_id:
            anchor_stone = s
            break

    if anchor_stone is None:
        return _empty_result(anchor_id, mode, search_version)

    logger.info(
        "similar_explorer anchor=%s main_color=%s dominant_hue=%s retrieval_family=%s relaxed=%s",
        anchor_id,
        _get_stone_veto_color(anchor_stone),
        _get_tag_value(anchor_stone, "dominant_hue"),
        get_retrieval_color_family(anchor_stone),
        _is_anchor_relaxed(anchor_stone),
    )

    anchor_dict = _stone_to_anchor(anchor_stone)

    # Section A is deprecated and returns empty.
    section_a = []
    section_a_vetoed = 0

    # Compute embeddings similarities once — shared by Sections B and C.
    similarities = _compute_similarities_for_anchor(anchor_id, anchor_stone, stones)

    # Section B is deprecated and returns empty.
    section_b = []
    section_b_vetoed = 0
    section_b_gate = "strict"

    # Fetch any suppressed sibling IDs (marked not related) for this anchor stone
    try:
        suppressed_ids = set(taxonomy_client.get_suppressed_siblings(anchor_id))
    except Exception:
        suppressed_ids = set()

    # Section C — broad CLIP visual.
    section_c, section_c_vetoed = _resolve_ai_suggestions(
        anchor_id, anchor_stone, stones,
        exclude_ids=suppressed_ids,
        similarities=similarities,
    )

    anchor_color = _get_stone_veto_color(anchor_stone)
    anchor_relaxed = _is_anchor_relaxed(anchor_stone)

    return {
        "anchor": anchor_dict,
        "sections": {
            "section_a": section_a,
            "section_b": section_b,
            "section_c": section_c,
        },
        "meta": {
            "mode": mode,
            "search_version": search_version,
            "anchor_color": anchor_color,
            "anchor_color_family": get_retrieval_color_family(anchor_stone),
            "anchor_dominant_hue": _get_tag_value(anchor_stone, "dominant_hue"),
            "anchor_relaxed": anchor_relaxed,
            "section_a_count": len(section_a),
            "section_b_count": len(section_b),
            "section_c_count": len(section_c),
            "section_a_vetoed_by_color": section_a_vetoed,
            "section_b_vetoed_by_color": section_b_vetoed,
            "section_c_vetoed_by_color": section_c_vetoed,
            "section_b_gate": section_b_gate,
        },
    }


# Task #388 — Register the colour-cache invalidator with tag_storage so admin
# saves to ``base_color`` / ``vein_colour`` evict the stale cache entry
# before the next Similar Stone Explorer request runs.  Guarded so test
# environments that stub tag_storage do not break import.
try:
    register_color_cache_invalidator(_invalidate_color_caches)
except Exception:
    logger.exception("failed to register colour cache invalidator with cache_manager")


# ---------------------------------------------------------------------------
# Visual Similarity (extracted from search/api.py)
# ---------------------------------------------------------------------------

_CHROMATIC_BG_COLORS = {"green", "red", "pink", "rose"}
_CHROMATIC_NORMALIZE = {"rose": "pink", "multicolor": "multi"}


def _normalize_chromatic(color: str) -> str:
    """Normalize chromatic color values (e.g. rose -> pink)."""
    return _CHROMATIC_NORMALIZE.get(color, color)


def _get_display_tag(stone: Dict, tag_name: str, default: Any = None) -> Tuple[Any, float, str]:
    """Get tag value, confidence, and source from display_tags."""
    tags = stone.get("display_tags", {})
    tag_info = tags.get(tag_name)
    if tag_info and isinstance(tag_info, dict):
        return tag_info.get("value", default), tag_info.get("confidence", 0.0), tag_info.get("source", "unknown")
    return default, 0.0, "missing"


def _stone_to_result_dict(stone: Dict, similarity: Optional[float] = None) -> Dict[str, Any]:
    """Map stone dict into a dict with the exact same schema as search's StoneResult pydantic model."""
    accents_raw, _, acc_source = _get_display_tag(stone, "accent_colors", [])
    if isinstance(accents_raw, str):
        try:
            accents_raw = json.loads(accents_raw)
        except:
            accents_raw = []
    if not accents_raw:
        accents_raw = []
    
    accent_list = [
        {"color": a.get("color", ""), "strength": a.get("strength", 0.0)}
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

    return {
        "company_stone_id": stone.get("company_stone_id", ""),
        "batch_id": stone.get("batch_id"),
        "stone_name": stone.get("stone_name"),
        "vendor_name": stone.get("vendor_name"),
        "thumbnail_url": stone.get("thumbnail_url"),
        "base_color": base_color,
        "base_color_confidence": base_conf,
        "vein_intensity": vein_intensity,
        "pattern_family": pattern_family,
        "accent_colors": accent_list,
        "visual_busyness": visual_busyness,
        "drama_score": drama_score,
        "dominant_tone": dominant_tone,
        "dominant_tone_confidence": tone_conf,
        "dominant_hue": dominant_hue,
        "dominant_hue_confidence": hue_conf,
        "has_manual_override": stone.get("has_manual_override", False),
        "similarity_score": similarity,
        "final_score": similarity or 0.0,
        "tags": tags,
        "match_tier": stone.get("_match_tier")
    }


def _load_embedding(company_stone_id: str) -> Optional[np.ndarray]:
    """Helper to load embedding for a stone, keeping backwards compatibility for pre-cache."""
    emb = ml_svc.get_embedding(company_stone_id)
    if emb is not None:
        return emb
    emb_path = Path("cache/embeddings_v3") / f"{company_stone_id}.npy"
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
    """Compute cosine similarity between query embedding and all stones."""
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


def compute_visual_similar(company_stone_id: str, stones: List[Dict], limit: int = 50) -> Optional[Dict[str, Any]]:
    """Compute the visual similar stones for a query stone."""
    query_embedding = _load_embedding(company_stone_id)
    if query_embedding is None:
        return None
    
    similarities = _compute_visual_similarities(query_embedding, stones)
    
    sorted_stones = sorted(
        [(s, similarities.get(s["company_stone_id"], 0.0)) for s in stones if s["company_stone_id"] != company_stone_id],
        key=lambda x: x[1],
        reverse=True
    )[:limit]
    
    results = []
    for stone, sim in sorted_stones:
        stone["similarity_score"] = sim
        stone["final_score"] = sim
        results.append(_stone_to_result_dict(stone, sim))
        
    return {
        "source_stone_id": company_stone_id,
        "similar_stones": results,
        "count": len(results)
    }


