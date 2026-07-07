"""
Tag-based storage for stock attributes.

Three tables:
- stock_tags: Manual/human overrides (source="manual")
- stock_tags_ai: AI-computed tags (source="ai")
- tag_schema: Extensible tag definitions

Merge rule: Manual tags always override AI tags.
"""

import sqlite3
import json
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)



def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_stock_tags_db_path() -> Path:
    """Return path to stock_tags.db — owned by taxonomy."""
    return Path(os.environ.get("STOCK_TAGS_DB_PATH"))
DB_PATH = get_stock_tags_db_path()

COLOR_OPTIONS = [
    "grey", "white", "blue", "black", "pink", "red", "rose",
    "brown", "yellow", "orange", "green", "beige", "charcoal", "gold", "purple"
]

BASE_COLOR_OPTIONS = ["white", "grey", "beige", "brown", "black", "red", "green", "pink", "yellow", "multi"]

ACCENT_COLOR_OPTIONS = ["green", "gold", "red", "pink", "yellow", "blue", "purple", "orange"]

VEIN_INTENSITY_OPTIONS = ["none", "soft", "medium", "strong"]
VEIN_STYLE_OPTIONS = ["linear", "branching", "web", "cloudy", "brecciated", "speckled"]
VEIN_DIRECTION_OPTIONS = ["none", "horizontal", "vertical", "diagonal", "multi"]
VEIN_SCALE_OPTIONS = ["fine", "medium", "bold"]

PATTERN_FAMILY_OPTIONS = ["uniform", "cloudy", "linear", "breccia", "webbed", "bold_veined"]
# Alias map: legacy UI / import values → canonical taxonomy vocabulary.
# Source: TAXONOMY_PATTERN_FAMILY_VALUES is the authoritative set.
# Kept here so any module doing normalization can import it without touching search_router.
PATTERN_FAMILY_ALIASES: dict = {
    "brecciated": "breccia",
    # Pattern P4 (J.3): canonical mapping reconciled with
    # taxonomy_backfill.PATTERN_MAP (write side). Both raw agent labels
    # below previously mapped to "bold_veined" on the read side, which
    # disagreed with the write side; the user-locked canonical decision
    # is onyx-like → linear, dramatic-mix → breccia.
    "onyx-like": "linear",
    "dramatic-mix": "breccia",
    "web": "webbed",
    "bold veined": "bold_veined",
    "bold-veined": "bold_veined",
}

TAXONOMY_FIELDS = {"pattern_family", "network_thickness", "has_bold_accent_vein", "base_color", "vein_colour"}

# Task #388 — callback registration slot so ``similar_explorer`` can invalidate
# its in-process colour caches when ``write_stock_override`` persists a
# ``base_color`` or ``vein_colour`` change.  Wired via a setter (no top-level
# import of similar_explorer) to avoid a circular import.  Single global slot
# is sufficient — there is only ever one similar_explorer module instance.
from src.utils.cache_manager import register_color_cache_invalidator, notify_color_cache_invalidator

_COLOR_CACHE_INVALIDATOR_FIELDS = {"base_color", "vein_colour"}

def _notify_color_cache_invalidator(field_name: str, company_stone_id: str) -> None:
    """Best-effort notification — never let a callback exception break the
    underlying write.  Only fires for fields whose retrieval-family
    resolution is colour-cached in similar_explorer."""
    if field_name not in _COLOR_CACHE_INVALIDATOR_FIELDS:
        return
    try:
        notify_color_cache_invalidator(company_stone_id)
    except Exception:
        logging.getLogger(__name__).exception(
            "color cache invalidator failed for cid=%s field=%s",
            company_stone_id, field_name,
        )


TAXONOMY_PATTERN_FAMILY_VALUES = {"uniform", "linear", "webbed", "breccia", "cloudy", "bold_veined"}
TAXONOMY_NETWORK_THICKNESS_VALUES = {"hairline", "medium", "bold", "na"}
TAXONOMY_BOOL_FIELDS = {"has_bold_accent_vein"}
IMAGE_ROLE_WEIGHTS = {"primary_slab": 1.0, "normal": 0.7, "detail_closeup": 0.2}
AGGREGATION_VERSION = "v1.0"

DEFAULT_TAG_SCHEMA = [
    {"tag_name": "main_color", "tag_type": "single", "options": COLOR_OPTIONS},
    {"tag_name": "second_color", "tag_type": "single", "options": COLOR_OPTIONS},
    {"tag_name": "third_color", "tag_type": "single", "options": COLOR_OPTIONS},
    {"tag_name": "exotic", "tag_type": "bool", "options": None},
    {"tag_name": "plain", "tag_type": "bool", "options": None},
    {"tag_name": "straight_vein", "tag_type": "bool", "options": None},
    {"tag_name": "wavy_vein", "tag_type": "bool", "options": None},
    {"tag_name": "spider_vein", "tag_type": "bool", "options": None},
    {"tag_name": "crazy_vein", "tag_type": "bool", "options": None},
    {"tag_name": "blotchy", "tag_type": "bool", "options": None},
    {"tag_name": "spotty", "tag_type": "bool", "options": None},
    {"tag_name": "cloudy", "tag_type": "bool", "options": None},
    {"tag_name": "dirty_look", "tag_type": "bool", "options": None},
    {"tag_name": "busy_look", "tag_type": "bool", "options": None},
    {"tag_name": "two_shades", "tag_type": "bool", "options": None},
    {"tag_name": "vein_colors", "tag_type": "multi", "options": COLOR_OPTIONS},

    {"tag_name": "vein_intensity", "tag_type": "single", "options": VEIN_INTENSITY_OPTIONS},
    {"tag_name": "vein_style", "tag_type": "multi", "options": VEIN_STYLE_OPTIONS},
    {"tag_name": "vein_direction", "tag_type": "single", "options": VEIN_DIRECTION_OPTIONS},
    {"tag_name": "vein_scale", "tag_type": "single", "options": VEIN_SCALE_OPTIONS},
    {"tag_name": "vein_contrast", "tag_type": "float", "options": None},
    {"tag_name": "pattern_family", "tag_type": "single", "options": PATTERN_FAMILY_OPTIONS},
    {"tag_name": "accent_colors", "tag_type": "json", "options": None},
    {"tag_name": "visual_busyness", "tag_type": "float", "options": None},
    {"tag_name": "drama_score", "tag_type": "float", "options": None},
    {"tag_name": "cloudiness_score", "tag_type": "float", "options": None},
    {"tag_name": "cloudiness_debug", "tag_type": "json", "options": None},
    {"tag_name": "dominant_tone", "tag_type": "single", "options": ["neutral", "green", "red", "warm", "cool"]},
    {"tag_name": "dominant_tone_conf", "tag_type": "float", "options": None},
    {"tag_name": "dominant_tone_debug", "tag_type": "json", "options": None},
    {"tag_name": "dominant_hue", "tag_type": "single", "options": ["neutral", "green", "red", "orange_brown", "yellow_beige", "blue_grey"]},
    {"tag_name": "dominant_hue_conf", "tag_type": "float", "options": None},
    {"tag_name": "dominant_hue_debug", "tag_type": "json", "options": None},
]


@dataclass
class MergedStockTags:
    company_stone_id: str
    tags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    debug_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def get_tag(self, tag_name: str) -> Optional[Any]:
        if tag_name in self.tags:
            return self.tags[tag_name].get("value")
        return None
    
    
    def to_dict(self) -> Dict:
        return {
            "company_stone_id": self.company_stone_id,
            "tags": self.tags,
            "debug_metrics": self.debug_metrics
        }


def ensure_db_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """Return a sqlite3 connection to stock_tags.db (or a custom path for trial mode).

    Existing callers with no argument are unaffected — they still get the production DB.
    Trial-aware callers pass db_path=get_config().get("stock_tags_db", True).
    """
    target = DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _ensure_correction_log_notes_column(conn):
    """Add a free-form `notes` column to correction_log (idempotent).

    Used by Color Fixing Stage 2 to persist the canonical derived-to-curated
    reason and any reviewer-supplied correction_reason alongside the audit row.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(correction_log)").fetchall()}
    if cols and "notes" not in cols:
        conn.execute("ALTER TABLE correction_log ADD COLUMN notes TEXT")


# Pattern Correction Phase 1 — surfaces.
PATTERN_CORRECTION_SURFACES = frozenset({"admin_findstone", "admin_reviewer"})


def _ensure_correction_log_phase1_columns(conn):
    """Add 7 nullable structured-reason columns to correction_log (idempotent).

    Pattern Correction Phase 1 (Task #261). All columns default to NULL so
    pre-existing rows and legacy callers continue to work unchanged.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(correction_log)").fetchall()}
    if not cols:
        return
    for new_col in (
        "reason_code",
        "reviewer_note",
        "source_tier",
        "ai_value_at_correction",
        "agg_value_at_correction",
        "session_id",
        "surface",
    ):
        if new_col not in cols:
            conn.execute(f"ALTER TABLE correction_log ADD COLUMN {new_col} TEXT")


def _migrate_correction_log_scope(conn):
    table_info = conn.execute("PRAGMA table_info(correction_log)").fetchall()
    if not table_info:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS correction_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT,
                image_id TEXT,
                field_name TEXT NOT NULL,
                computed_value TEXT,
                computed_confidence REAL,
                corrected_value TEXT,
                action_type TEXT NOT NULL CHECK(action_type IN ('CORRECTION','CONFIRMATION','CLEAR_OVERRIDE')),
                scope_type TEXT NOT NULL CHECK(scope_type IN ('IMAGE','BATCH','STOCK')),
                reviewer_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_version TEXT
            )
        """)
        return

    has_batch_scope = False
    try:
        conn.execute("INSERT INTO correction_log (stock_id, field_name, computed_value, corrected_value, action_type, scope_type, reviewer_id, timestamp) VALUES ('__test__', '__test__', NULL, NULL, 'CORRECTION', 'BATCH', '__test__', '__test__')")
        conn.execute("DELETE FROM correction_log WHERE stock_id = '__test__' AND field_name = '__test__'")
        has_batch_scope = True
    except Exception:
        has_batch_scope = False

    if not has_batch_scope:
        existing_rows = conn.execute("SELECT * FROM correction_log ORDER BY log_id").fetchall()
        existing_data = [dict(row) for row in existing_rows]

        conn.execute("DROP TABLE IF EXISTS correction_log")
        conn.execute("""
            CREATE TABLE correction_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id TEXT,
                image_id TEXT,
                field_name TEXT NOT NULL,
                computed_value TEXT,
                computed_confidence REAL,
                corrected_value TEXT,
                action_type TEXT NOT NULL CHECK(action_type IN ('CORRECTION','CONFIRMATION','CLEAR_OVERRIDE')),
                scope_type TEXT NOT NULL CHECK(scope_type IN ('IMAGE','BATCH','STOCK')),
                reviewer_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_version TEXT
            )
        """)

        for row in existing_data:
            conn.execute("""
                INSERT INTO correction_log
                (stock_id, image_id, field_name, computed_value, computed_confidence,
                 corrected_value, action_type, scope_type, reviewer_id, timestamp, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row.get("stock_id"), row.get("image_id"), row["field_name"],
                  row.get("computed_value"), row.get("computed_confidence"),
                  row.get("corrected_value"), row["action_type"], row["scope_type"],
                  row["reviewer_id"], row["timestamp"], row.get("model_version")))


def init_db():
    ensure_db_dir()
    
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_tags_ai (
                company_stone_id TEXT NOT NULL,
                tag_name TEXT NOT NULL,
                tag_value TEXT,
                source TEXT NOT NULL DEFAULT 'ai',
                confidence REAL NOT NULL DEFAULT 0.5,
                updated_at TEXT NOT NULL,
                updated_by TEXT NOT NULL DEFAULT 'system',
                PRIMARY KEY (company_stone_id, tag_name)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tag_schema (
                tag_name TEXT PRIMARY KEY,
                tag_type TEXT NOT NULL,
                options_json TEXT,
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_tags_ai_stock ON stock_tags_ai(company_stone_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_tags_ai_tag ON stock_tags_ai(tag_name)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stock_tags_ai_cover ON stock_tags_ai"
            "(company_stone_id, tag_name, tag_value, confidence)"
        )
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_aggregated (
                company_stone_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                aggregated_value TEXT,
                secondary_value TEXT,
                aggregation_version TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (company_stone_id, field_name)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_agg_stock ON stock_aggregated(company_stone_id)")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_overrides (
                company_stone_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                override_value TEXT,
                reviewer_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (company_stone_id, field_name)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_ovr_stock ON stock_overrides(company_stone_id)")

        _migrate_correction_log_scope(conn)
        _ensure_correction_log_notes_column(conn)
        _ensure_correction_log_phase1_columns(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corr_log_stock ON correction_log(stock_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corr_log_image ON correction_log(image_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_corr_log_ts ON correction_log(timestamp)")


        conn.commit()
        
        _init_default_schema(conn)


def _init_default_schema(conn):
    now = datetime.now().isoformat()
    
    for schema in DEFAULT_TAG_SCHEMA:
        existing = conn.execute(
            "SELECT 1 FROM tag_schema WHERE tag_name = ?",
            (schema["tag_name"],)
        ).fetchone()
        
        if not existing:
            conn.execute("""
                INSERT INTO tag_schema (tag_name, tag_type, options_json, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                schema["tag_name"],
                schema["tag_type"],
                json.dumps(schema["options"]) if schema["options"] else None,
                now
            ))
    
    conn.commit()


def _serialize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.bool_):
        return "true" if value else "false"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, np.floating):
        return str(float(value))
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, (list, dict)):
        return json.dumps(convert_numpy_types(value))
    return str(value)






def get_stones_missing_ai_tag(company_stone_ids: List[str], tag_name: str) -> List[str]:
    init_db()
    if not company_stone_ids:
        return []
    with get_connection() as conn:
        placeholders = ",".join(["?"] * len(company_stone_ids))
        rows = conn.execute(f"""
            SELECT company_stone_id FROM stock_tags_ai 
            WHERE tag_name = ? AND company_stone_id IN ({placeholders})
        """, (tag_name, *company_stone_ids)).fetchall()
        found_ids = {row[0] for row in rows}
        return [sid for sid in company_stone_ids if sid not in found_ids]


def get_lightness_pending(company_stone_ids: List[str]) -> List[str]:
    """Retrieve stones that have base_color but no stone_lightness in stock_tags_ai."""
    if not company_stone_ids:
        return []
    init_db()
    with get_connection() as conn:
        placeholders = ",".join(["?"] * len(company_stone_ids))
        rows = conn.execute(f"""
            SELECT company_stone_id, tag_name FROM stock_tags_ai 
            WHERE tag_name IN ('base_color', 'stone_lightness') AND company_stone_id IN ({placeholders})
        """, company_stone_ids).fetchall()
        has_base_color = {r[0] for r in rows if r[1] == "base_color"}
        has_lightness = {r[0] for r in rows if r[1] == "stone_lightness"}
    return sorted(list(has_base_color - has_lightness))


def set_ai_tags_batch(
    company_stone_id: str,
    tags: Dict[str, Tuple[Any, float]]
) -> bool:
    init_db()
    
    now = datetime.now().isoformat()
    
    with get_connection() as conn:
        for tag_name, (tag_value, confidence) in tags.items():
            value_str = _serialize_value(tag_value)
            conf_float = float(confidence) if isinstance(confidence, np.floating) else confidence
            conn.execute("""
                INSERT OR REPLACE INTO stock_tags_ai 
                (company_stone_id, tag_name, tag_value, source, confidence, updated_at, updated_by)
                VALUES (?, ?, ?, 'ai', ?, ?, 'system')
            """, (company_stone_id, tag_name, value_str, conf_float, now))
        conn.commit()
    
    return True




def _validate_taxonomy_field(field_name: str):
    if field_name not in TAXONOMY_FIELDS:
        raise ValueError(f"Invalid taxonomy field: '{field_name}'. Must be one of: {sorted(TAXONOMY_FIELDS)}")


# NOTE: ``rose`` is stored truthfully as ``rose`` in stock_overrides /
# stock_aggregated but is normalised to ``pink`` at retrieval time by
# ``similar_explorer._COLOR_NORMALIZE`` so the retrieval-family gate still
# resolves to ``PINK``.  Do not drop ``rose`` from this whitelist without
# also coordinating with the override → family resolution path.
TAXONOMY_BASE_COLOR_VALUES = {"white", "beige", "grey", "black", "brown", "green", "red", "pink", "rose", "yellow", "multi","blue"}
TAXONOMY_VEIN_COLOUR_VALUES = {"gold", "grey", "white", "pink", "green", "black", "none", "mixed"}


def _validate_taxonomy_value(field_name: str, value):
    if value is None:
        return
    if field_name == "has_bold_accent_vein":
        if not isinstance(value, bool):
            raise ValueError(
                f"has_bold_accent_vein must be a Python bool (True/False), got {type(value).__name__}: {value!r}"
            )
    elif field_name == "pattern_family":
        if value not in TAXONOMY_PATTERN_FAMILY_VALUES:
            raise ValueError(
                f"Invalid pattern_family value: '{value}'. Must be one of: {sorted(TAXONOMY_PATTERN_FAMILY_VALUES)}"
            )
    elif field_name == "network_thickness":
        if value not in TAXONOMY_NETWORK_THICKNESS_VALUES:
            raise ValueError(
                f"Invalid network_thickness value: '{value}'. Must be one of: {sorted(TAXONOMY_NETWORK_THICKNESS_VALUES)}"
            )
    elif field_name == "base_color":
        if value not in TAXONOMY_BASE_COLOR_VALUES:
            raise ValueError(
                f"Invalid base_color value: '{value}'. Must be one of: {sorted(TAXONOMY_BASE_COLOR_VALUES)}"
            )
    elif field_name == "vein_colour":
        if value not in TAXONOMY_VEIN_COLOUR_VALUES:
            raise ValueError(
                f"Invalid vein_colour value: '{value}'. Must be one of: {sorted(TAXONOMY_VEIN_COLOUR_VALUES)}"
            )




def _serialize_taxonomy_bool(value: bool) -> str:
    return "true" if value else "false"

def write_stock_override(
    company_stone_id: str,
    field_name: str,
    value,
    reviewer_id: str,
    conn: Optional[sqlite3.Connection] = None,
):
    _validate_taxonomy_field(field_name)
    if value is not None:
        _validate_taxonomy_value(field_name, value)

    now = datetime.now().isoformat()

    if value is None:
        stored_value = None
    elif field_name in TAXONOMY_BOOL_FIELDS:
        stored_value = _serialize_taxonomy_bool(value)
    else:
        stored_value = str(value)

    sql = """
        INSERT OR REPLACE INTO stock_overrides
        (company_stone_id, field_name, override_value, reviewer_id, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """
    params = (company_stone_id, field_name, stored_value, reviewer_id, now)

    if conn is not None:
        # Caller-supplied connection: the caller owns transaction lifetime
        # and commit timing.  Invalidating the colour cache here would fire
        # BEFORE commit, opening a race where a concurrent read repopulates
        # the cache from the still-pre-commit DB snapshot and the stale
        # value survives the eventual commit.  Callers passing ``conn`` are
        # therefore responsible for calling
        # ``similar_explorer._invalidate_color_caches(company_stone_id)``
        # (or equivalently re-invoking ``write_stock_override`` without
        # ``conn``) AFTER their commit if the write touched
        # ``base_color`` / ``vein_colour``.  All current production
        # callers (tag_router endpoints) use the no-conn path below.
        conn.execute(sql, params)
        return

    init_db()
    with get_connection() as c:
        c.execute(sql, params)
        c.commit()
    # Post-commit notification — safe to fire here because the row is
    # durable and any concurrent read that races against the eviction
    # will repopulate the cache from the new committed value.
    _notify_color_cache_invalidator(field_name, company_stone_id)



def log_correction(
    stock_id: Optional[str],
    image_id: Optional[str],
    field_name: str,
    computed_value: Optional[str],
    computed_confidence: Optional[float],
    corrected_value: Optional[str],
    action_type: str,
    scope_type: str,
    reviewer_id: str,
    model_version: Optional[str],
    conn=None,
    notes: Optional[str] = None,
    *,
    reason_code: Optional[str] = None,
    reviewer_note: Optional[str] = None,
    source_tier: Optional[str] = None,
    ai_value_at_correction: Optional[str] = None,
    agg_value_at_correction: Optional[str] = None,
    session_id: Optional[str] = None,
    surface: Optional[str] = None,
) -> Optional[int]:
    """Insert one row into correction_log. Returns the new log_id when an
    explicit `conn` is supplied (so the caller can chain it inside their
    transaction), otherwise None.

    `notes` is the free-form audit reason (added by Color Truth Phase 2 §11):
    canonical derived-to-curated string when applicable, else the
    reviewer-supplied correction_reason.

    The 7 keyword-only parameters (Pattern Correction Phase 1, Task #261) are
    additive structured-reason metadata. All default to None and are written
    as NULL when omitted, preserving full backwards compatibility.
    """
    now = datetime.now().isoformat()

    def _do_insert(c):
        cur = c.execute("""
            INSERT INTO correction_log
            (stock_id, image_id, field_name, computed_value, computed_confidence,
             corrected_value, action_type, scope_type, reviewer_id, timestamp,
             model_version, notes,
             reason_code, reviewer_note, source_tier,
             ai_value_at_correction, agg_value_at_correction,
             session_id, surface)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stock_id, image_id, field_name, computed_value, computed_confidence,
              corrected_value, action_type, scope_type, reviewer_id, now,
              model_version, notes,
              reason_code, reviewer_note, source_tier,
              ai_value_at_correction, agg_value_at_correction,
              session_id, surface))
        return int(cur.lastrowid)

    if conn:
        return _do_insert(conn)
    init_db()
    with get_connection() as c:
        _do_insert(c)
        c.commit()
    return None



def write_stock_aggregated(
    company_stone_id: str,
    field_name: str,
    aggregated_value: Optional[str],
    secondary_value: Optional[str] = None,
    aggregation_version: str = AGGREGATION_VERSION,
    conn: Optional[sqlite3.Connection] = None,
):
    now = datetime.now().isoformat()
    sql = """
        INSERT OR REPLACE INTO stock_aggregated
        (company_stone_id, field_name, aggregated_value, secondary_value, aggregation_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = (company_stone_id, field_name, aggregated_value, secondary_value, aggregation_version, now)
    if conn is not None:
        conn.execute(sql, params)
        return
    init_db()
    with get_connection() as c:
        c.execute(sql, params)
        c.commit()

def get_family_hue_stats(family_name: str) -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT family_name, mean_ratio, count
            FROM hue_stats_family
            WHERE target_hue = ?
            """,
            (family_name,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_block_hue_stats(block_id: str) -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT block_name, mean_ratio, count
            FROM hue_stats_block
            WHERE target_hue = ?
            """,
            (block_id,)
        ).fetchall()
        return [dict(r) for r in rows]






def get_batch_facet_data(
    company_stone_ids: List[str],
    tag_names: List[str],
    agg_fields: List[str],
    ovr_fields: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    init_db()
    if not company_stone_ids:
        return {
            "ai_rows": [],
            "manual_rows": [],
            "agg_rows": [],
            "ovr_rows": []
        }

    placeholders = ",".join("?" for _ in company_stone_ids)
    tag_placeholders = ",".join("?" for _ in tag_names)
    agg_placeholders = ",".join("?" for _ in agg_fields)
    ovr_placeholders = ",".join("?" for _ in ovr_fields)

    with get_connection() as conn:
        ai_rows = conn.execute(f"""
            SELECT company_stone_id, tag_name, tag_value, confidence
            FROM stock_tags_ai
            WHERE company_stone_id IN ({placeholders})
            AND tag_name IN ({tag_placeholders})
        """, company_stone_ids + tag_names).fetchall()

        manual_rows = []

        agg_rows = conn.execute(f"""
            SELECT company_stone_id, field_name, aggregated_value
            FROM stock_aggregated
            WHERE company_stone_id IN ({placeholders})
            AND field_name IN ({agg_placeholders})
        """, company_stone_ids + agg_fields).fetchall()

        ovr_rows = conn.execute(f"""
            SELECT company_stone_id, field_name, override_value
            FROM stock_overrides
            WHERE company_stone_id IN ({placeholders})
            AND field_name IN ({ovr_placeholders})
            AND override_value IS NOT NULL
        """, company_stone_ids + ovr_fields).fetchall()

    return {
        "ai_rows": [dict(r) for r in ai_rows],
        "manual_rows": [dict(r) for r in manual_rows],
        "agg_rows": [dict(r) for r in agg_rows],
        "ovr_rows": [dict(r) for r in ovr_rows]
    }


def get_stone_tags(company_stone_id: str) -> Dict[str, Any]:
    init_db()
    manual_tags = {}
    ai_tags = {}
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT tag_name, tag_value, confidence FROM stock_tags_ai WHERE company_stone_id = ?",
            (company_stone_id,)
        ).fetchall()
        for row in rows:
            ai_tags[row["tag_name"]] = {
                "value": row["tag_value"],
                "confidence": row["confidence"]
            }
        
    return {
        "manual_tags": manual_tags,
        "ai_tags": ai_tags,
        "debug_metrics": {}
    }


def get_stone_color_info(company_stone_id: str) -> Dict[str, Optional[str]]:
    init_db()
    base_color_val = None
    ai_color_val = None
    
    with get_connection() as conn:
        row = conn.execute(
            "SELECT override_value FROM stock_overrides "
            "WHERE company_stone_id=? AND field_name='base_color' "
            "AND override_value IS NOT NULL",
            (company_stone_id,),
        ).fetchone()
        if row and row["override_value"] is not None:
            base_color_val = row["override_value"]
        else:
            row = conn.execute(
                "SELECT aggregated_value FROM stock_aggregated "
                "WHERE company_stone_id=? AND field_name='base_color' "
                "AND aggregated_value IS NOT NULL",
                (company_stone_id,),
            ).fetchone()
            if row and row["aggregated_value"] is not None:
                base_color_val = row["aggregated_value"]
                
        rows = conn.execute(
            "SELECT tag_name, tag_value FROM stock_tags_ai "
            "WHERE company_stone_id=? AND tag_name IN ('base_color','main_color') "
            "AND tag_value IS NOT NULL AND tag_value != ''",
            (company_stone_id,),
        ).fetchall()
        by_name = {r["tag_name"]: r["tag_value"] for r in rows}
        ai_color_val = by_name.get("base_color") or by_name.get("main_color")
        
    return {
        "base_color": base_color_val,
        "ai_color": ai_color_val
    }


def get_page_hydration_data(company_stone_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    init_db()
    if not company_stone_ids:
        return {
            "ai_rows": [],
            "manual_rows": [],
            "agg_rows": [],
            "ovr_rows": []
        }
        
    placeholders = ",".join("?" for _ in company_stone_ids)
    
    with get_connection() as conn:
        ai_rows = conn.execute(f"""
            SELECT company_stone_id, tag_name, tag_value, confidence
            FROM stock_tags_ai
            WHERE company_stone_id IN ({placeholders})
        """, company_stone_ids).fetchall()

        manual_rows = []

        agg_rows = conn.execute(f"""
            SELECT company_stone_id, field_name, aggregated_value
            FROM stock_aggregated
            WHERE company_stone_id IN ({placeholders})
            AND field_name IN ('pattern_family', 'network_thickness', 'has_bold_accent_vein', 'base_color', 'vein_colour')
        """, company_stone_ids).fetchall()

        ovr_rows = conn.execute(f"""
            SELECT company_stone_id, field_name, override_value
            FROM stock_overrides
            WHERE company_stone_id IN ({placeholders})
            AND field_name IN ('pattern_family', 'network_thickness', 'has_bold_accent_vein', 'base_color', 'vein_colour')
            AND override_value IS NOT NULL
        """, company_stone_ids).fetchall()
        
    return {
        "ai_rows": [dict(r) for r in ai_rows],
        "manual_rows": [dict(r) for r in manual_rows],
        "agg_rows": [dict(r) for r in agg_rows],
        "ovr_rows": [dict(r) for r in ovr_rows]
    }


def check_integrity(db_path: Optional[Path] = None) -> bool:
    path = db_path if db_path is not None else DB_PATH
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return bool(row and str(row[0]).lower() == "ok")
    except Exception:
        return False
    finally:
        conn.close()


def backup_tags_db(backup_path: Path) -> None:
    shutil.copy2(DB_PATH, backup_path)


def snapshot_tag_counts(cids: set) -> Dict[str, Tuple[int, int]]:
    if not cids:
        return {}
    tag_counts = {}
    tag_stones = {}
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            "SELECT company_stone_id, tag_name FROM stock_tags_ai"
        ).fetchall()
        for cid, tag in rows:
            if cid in cids:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                if tag not in tag_stones:
                    tag_stones[tag] = set()
                tag_stones[tag].add(cid)
    finally:
        conn.close()
    return {tag: (tag_counts[tag], len(tag_stones[tag])) for tag in tag_counts}


def has_ai_main_color(stone_id: str) -> bool:
    """True iff ``stock_tags_ai`` has a non-NULL ``main_color`` row."""
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT tag_value FROM stock_tags_ai "
            "WHERE company_stone_id=? AND tag_name='main_color' LIMIT 1",
            (stone_id,),
        ).fetchone()
    if row is None:
        return False
    val = row["tag_value"] if hasattr(row, "keys") else row[0]
    return val is not None and str(val).strip() != ""




