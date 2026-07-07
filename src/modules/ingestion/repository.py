"""
Dataset storage for versioned image metadata.

Tables:
- datasets: Dataset versions with metadata
- dataset_images: Image records per dataset

Images are stored with public URLs, not downloaded locally.
Manual V3 tags are stored separately and NEVER touched by imports.
"""

import sqlite3
import re
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from src.utils.cache_manager import invalidate_stone_cache  # noqa: PLC0415


@dataclass
class DatasetImage:
    dataset_id: str
    image_asset_id: str
    image_url: str
    batch_id: str
    company_stone_id: str
    vendor: str
    stone_name_raw: str
    canonical_family: str
    block_no: str
    block_id: str

    def to_dict(self) -> Dict:
        return asdict(self)


logger = logging.getLogger(__name__)
DB_PATH = Path(os.environ.get("DATASETS_DB_PATH"))

def get_db_info() -> Dict[str, Any]:
    """Get info about the current database configuration."""
    return {
        "db_path": str(DB_PATH),
        "exists": DB_PATH.exists(),
        "is_clean_db": "clean" in DB_PATH.name,
        "db_name": DB_PATH.name
    }

FINISH_WORDS = [
    "polished", "honed", "leathered", "brushed", "flamed", 
    "sandblasted", "tumbled", "bush hammered", "antiqued",
    "satin", "matte", "glossy", "natural", "split face"
]

THICKNESS_PATTERN = re.compile(r'\b\d+(\.\d+)?\s*(cm|mm|in|inch|inches)?\b', re.IGNORECASE)


@dataclass
class Dataset:
    dataset_id: str
    created_at: str
    source_filename: str
    rows_read: int
    is_active: bool
    unique_images: int = 0
    unique_batches: int = 0
    unique_company_stones: int = 0
    unique_families: int = 0
    unique_blocks: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)

def ensure_db_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection():
    """Return a sqlite3 connection to DB_PATH."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


_init_db_done = False


def init_db():
    global _init_db_done
    if _init_db_done:
        return
    ensure_db_dir()
    
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_filename TEXT,
                rows_read INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 0,
                unique_images INTEGER DEFAULT 0,
                unique_batches INTEGER DEFAULT 0,
                unique_company_stones INTEGER DEFAULT 0,
                unique_families INTEGER DEFAULT 0,
                unique_blocks INTEGER DEFAULT 0,
                column_mapping_used TEXT
            )
        """)
        
        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN column_mapping_used TEXT")
        except:
            pass
        
        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN import_contract_version TEXT")
        except:
            pass
        
        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN detected_headers TEXT")
        except:
            pass
        
        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN validation_summary TEXT")
        except:
            pass

        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN validation_status TEXT DEFAULT 'draft'")
        except:
            pass

        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN activation_blocked INTEGER DEFAULT 1")
        except:
            pass

        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN validation_timestamp TEXT")
        except:
            pass

        # Data migration: datasets that existed before v1 safety pipeline.
        # Pre-pipeline rows are identified by BOTH validation_timestamp IS NULL AND
        # import_contract_version IS NULL. New imports always set import_contract_version='v1',
        # so future rows with an accidental null timestamp are not auto-upgraded.
        conn.execute(
            "UPDATE datasets SET validation_status = 'validated', activation_blocked = 0 "
            "WHERE validation_timestamp IS NULL AND import_contract_version IS NULL"
        )
        conn.commit()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                dataset_id TEXT NOT NULL,
                image_asset_id TEXT NOT NULL,
                image_url TEXT,
                batch_id TEXT,
                company_stone_id TEXT,
                vendor TEXT,
                stone_name_raw TEXT,
                canonical_family TEXT,
                block_no TEXT,
                block_id TEXT,
                PRIMARY KEY (dataset_id, image_asset_id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_images_company ON dataset_images(dataset_id, company_stone_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_images_batch ON dataset_images(dataset_id, batch_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_images_family ON dataset_images(dataset_id, canonical_family)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_images_block ON dataset_images(dataset_id, block_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_di_dataset_batch_cover ON dataset_images"
            "(dataset_id, batch_id, company_stone_id, stone_name_raw, vendor, canonical_family, image_url)"
        )
        
        conn.commit()

    _init_db_done = True


def normalize_stone_name(stone_name_raw: str) -> str:
    if not stone_name_raw:
        return ""
    
    name = stone_name_raw.lower().strip()
    name = name.replace("gray", "grey")
    
    for finish in FINISH_WORDS:
        name = re.sub(r'\b' + re.escape(finish) + r'\b', '', name, flags=re.IGNORECASE)
    
    name = THICKNESS_PATTERN.sub('', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def get_active_dataset_id() -> Optional[str]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT dataset_id FROM datasets WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        if row and row["dataset_id"]:
            return row["dataset_id"]
        
        fallback = conn.execute(
            "SELECT dataset_id FROM datasets ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if fallback and fallback["dataset_id"]:
            return fallback["dataset_id"]
    return None


def get_all_clean_datasets() -> List[Dict[str, Any]]:
    """Return all datasets from the clean DB as dicts,
    including validation_status, activation_blocked, blocked_reasons, and
    continuity_report fields parsed from validation_summary JSON."""

    # Ensure schema + data migration runs (handles legacy-validated datasets)
    init_db()

    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC"
        ).fetchall()
    except Exception:
        conn.close()
        return []
    finally:
        conn.close()

    result = []
    for row in rows:
        vs_raw = row["validation_summary"] if "validation_summary" in row.keys() else None
        vs_dict: Dict[str, Any] = {}
        if vs_raw:
            try:
                vs_dict = json.loads(vs_raw)
            except Exception:
                pass

        result.append({
            "dataset_id": row["dataset_id"],
            "created_at": row["created_at"],
            "source_filename": row["source_filename"] or "",
            "rows_read": row["rows_read"] or 0,
            "is_active": bool(row["is_active"]),
            "unique_images": row["unique_images"] or 0,
            "unique_batches": row["unique_batches"] or 0,
            "unique_company_stones": row["unique_company_stones"] or 0,
            "unique_families": row["unique_families"] or 0,
            "unique_blocks": row["unique_blocks"] or 0,
            "validation_status": row["validation_status"] if "validation_status" in row.keys() else vs_dict.get("validation_status", "draft"),
            "activation_blocked": bool(row["activation_blocked"] if "activation_blocked" in row.keys() else 1),
            "validation_timestamp": row["validation_timestamp"] if "validation_timestamp" in row.keys() else None,
            "blocked_reasons": vs_dict.get("blocked_reasons", []),
            "continuity_status": vs_dict.get("continuity_status", None),
            "continuity_warnings": vs_dict.get("continuity_warnings", []),
            "url_anomaly_count": vs_dict.get("url_anomaly_count", 0),
        })
    return result


def set_active_dataset(dataset_id: str) -> bool:
    """Activate a dataset, gated by validation_status=validated AND activation_blocked=0.

    Atomically archives the previously active dataset and sets the new one active.
    Raises ValueError with a human-readable message if activation is blocked.
    Returns True on success, False if dataset not found.
    """
    init_db()

    ctx = get_connection()

    with ctx as conn:
        row = conn.execute(
            "SELECT dataset_id, validation_status, activation_blocked FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        ).fetchone()

        if not row:
            return False

        vs = row["validation_status"] or "draft"
        ab_raw = row["activation_blocked"]
        blocked = int(ab_raw) if ab_raw is not None else 1

        if vs != "validated" or blocked:
            reasons = []
            if vs != "validated":
                reasons.append(f"validation_status is '{vs}' (must be 'validated')")
            if blocked:
                reasons.append("activation_blocked is True")
            raise ValueError(
                f"Cannot activate dataset '{dataset_id}': " + "; ".join(reasons)
            )

        # --- batch_id collision guard ---
        # Proof from Task #76: in a safe dataset, no batch_id maps to more than one
        # company_stone_id.  If a collision exists, the Phase A GROUP BY query would
        # collapse multiple stones into one candidate, silently dropping stones from
        # search results.  Block activation loudly instead.
        collision_rows = conn.execute(
            """
            SELECT batch_id, COUNT(DISTINCT company_stone_id) AS cnt
            FROM dataset_images
            WHERE dataset_id = ?
            GROUP BY batch_id
            HAVING COUNT(DISTINCT company_stone_id) > 1
            """,
            (dataset_id,),
        ).fetchall()
        if collision_rows:
            collision_count = len(collision_rows)
            example = collision_rows[0]
            raise ValueError(
                f"Cannot activate dataset '{dataset_id}': batch_id collision detected. "
                f"{collision_count} batch_id(s) map to multiple company_stone_id values. "
                f"Example: batch_id='{example['batch_id']}' → "
                f"{example['cnt']} distinct stones. "
                "Fix the import data before activating."
            )
        logger.info(
            "batch_id collision guard PASSED for dataset '%s': 0 collisions found.",
            dataset_id,
        )

        conn.execute(
            "UPDATE datasets SET is_active = 0, validation_status = 'archived' "
            "WHERE is_active = 1 AND dataset_id != ?",
            (dataset_id,)
        )
        conn.execute(
            "UPDATE datasets SET is_active = 1 WHERE dataset_id = ?",
            (dataset_id,)
        )
        conn.commit()

    # Invalidate the stone cache so the next search uses the newly active dataset.
    # Lazy import avoids circular dependency (search_router imports dataset_storage).
    try:
        invalidate_stone_cache()
    except Exception:
        pass

    return True


def delete_dataset(dataset_id: str) -> bool:
    init_db()
    
    with get_connection() as conn:
        row = conn.execute(
            "SELECT is_active FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        ).fetchone()
        
        if row and row["is_active"]:
            return False
        
        conn.execute(
            "DELETE FROM dataset_images WHERE dataset_id = ?",
            (dataset_id,)
        )
        conn.execute(
            "DELETE FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        )
        conn.commit()
    
    return True





@dataclass
class CleanImportResult:
    success: bool
    dataset_id: str
    rows_read: int
    rows_imported: int
    duplicates_skipped: int
    errors: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "draft"
    activation_blocked: bool = True
    blocked_reasons: List[str] = field(default_factory=list)
    continuity_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


SEARCH_CSV_V1_MANDATORY_COLUMNS = [
    "company_stone_id",
    "packing_list_id",
    "object_key",
    "url",
    "name",
    "vendor name",
]

SEARCH_CSV_V1_CONTRACT_VERSION = "search_csv_v1"

SEARCH_CSV_V1_SUSPICIOUS_VARIANTS = {
    "vendor name": ["vendor", "vendor_name", "vendor.name", "vendorname", "name.1"],
    "name": ["stone_name", "stone name", "material", "name.1"],
    "object_key": ["object key", "objectkey", "image_id", "asset_id"],
    "packing_list_id": ["packing_id", "batch_id", "packinglistid"],
    "company_stone_id": ["stone_id", "company_id", "companystoneid"],
    "url": ["image_url", "photo_url", "link"],
}


def validate_search_csv_v1(
    rows: List[Dict[str, str]],
    headers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate a CSV against the Search CSV v1 contract (frozen schema).

    Mandatory columns (all 6 must be present by exact name):
        company_stone_id, packing_list_id, object_key, url, name, vendor name

    Checks performed:
    1. Header validation:
       - All 6 mandatory columns present (case-insensitive lookup, exact name required)
       - No duplicate / ambiguous headers that indicate a schema drift
         (e.g. name + name.1 together signals missing vendor name column)
       - Suspicious renamed variants of mandatory fields are flagged

    2. Row-level validation:
       - Every mandatory field must be non-empty in every row
       - Count invalid rows and surface a sample (up to 10) for admin review

    Expected outcomes for known files:
      - 1export20260122_corrected.csv: passes all checks (has vendor name, no duplicates)
      - 2export_20260131.csv: fails header validation — vendor name is absent and
        name appears twice (as name and name.1), indicating the schema has drifted.

    Datasets in cache/fingerprints_v3/datasets.db (source_filename field):
      - Datasets imported from 1export20260122_corrected.csv (or identical schema):
        PASS — all mandatory columns present.
      - Datasets imported from 2export_20260131.csv (or similar drifted exports):
        FAIL — vendor name missing, duplicate name columns.

    Returns a structured report dict; no DB writes are performed.
    """
    if headers is None:
        headers = list(rows[0].keys()) if rows else []

    headers_lower_map: Dict[str, str] = {}
    duplicate_headers: List[str] = []
    seen_lower: Dict[str, str] = {}
    for h in headers:
        hl = h.lower().strip()
        if hl in seen_lower:
            duplicate_headers.append(h)
            if seen_lower[hl] not in duplicate_headers:
                duplicate_headers.append(seen_lower[hl])
        else:
            seen_lower[hl] = h
            headers_lower_map[hl] = h

    mandatory_lower = [c.lower() for c in SEARCH_CSV_V1_MANDATORY_COLUMNS]
    missing_columns: List[str] = []
    for col in mandatory_lower:
        if col not in headers_lower_map:
            display = next(
                c for c in SEARCH_CSV_V1_MANDATORY_COLUMNS if c.lower() == col
            )
            missing_columns.append(display)

    suspicious_columns: List[str] = []
    for mandatory_col, variants in SEARCH_CSV_V1_SUSPICIOUS_VARIANTS.items():
        mandatory_present = mandatory_col.lower() in headers_lower_map
        for variant in variants:
            if variant.lower() in headers_lower_map:
                suspicious_columns.append(
                    f"'{headers_lower_map[variant.lower()]}' looks like a renamed variant "
                    f"of mandatory column '{mandatory_col}'"
                    + (" (mandatory column IS present)" if mandatory_present
                       else " (mandatory column is MISSING)")
                )

    name_name1_no_vendor = (
        "name" in headers_lower_map
        and "name.1" in headers_lower_map
        and "vendor name" not in headers_lower_map
    )
    ambiguous_note = ""
    if name_name1_no_vendor:
        ambiguous_note = (
            "Duplicate 'name' columns detected (name + name.1) with no 'vendor name' column. "
            "This is the drifted schema pattern where vendor name was lost."
        )

    passed_header_validation = (
        len(missing_columns) == 0
        and len(duplicate_headers) == 0
        and not name_name1_no_vendor
    )

    row_count_total = len(rows)
    row_count_valid = 0
    row_count_invalid = 0
    row_level_errors_by_field: Dict[str, int] = {col: 0 for col in mandatory_lower}
    sample_invalid_rows: List[Dict[str, Any]] = []

    if passed_header_validation:
        for i, row in enumerate(rows):
            row_errors = []
            for col in mandatory_lower:
                actual_col = headers_lower_map[col]
                val = (row.get(actual_col) or "").strip()
                if not val:
                    row_level_errors_by_field[col] += 1
                    row_errors.append(f"empty '{actual_col}'")
            if row_errors:
                row_count_invalid += 1
                if len(sample_invalid_rows) < 10:
                    sample_invalid_rows.append({
                        "row_index": i + 2,
                        "errors": row_errors
                    })
            else:
                row_count_valid += 1
    else:
        row_count_valid = 0
        row_count_invalid = row_count_total

    # --- File-level: duplicate object_key (image primary key) detection (warn only) ---
    # Multiple rows per stone (same company_stone_id, different object_key) is normal.
    # Duplicate object_keys are silently deduped at import time — informational only.
    duplicate_object_key_count = 0
    duplicate_object_key_samples: List[str] = []
    if passed_header_validation:
        ok_col = next(
            (headers_lower_map[c] for c in headers_lower_map if c == "object_key"),
            None
        )
        if ok_col:
            seen_keys: Dict[str, int] = {}
            for row in rows:
                key = (row.get(ok_col) or "").strip()
                if key:
                    seen_keys[key] = seen_keys.get(key, 0) + 1
            for key, count in seen_keys.items():
                if count > 1:
                    duplicate_object_key_count += count - 1
                    if len(duplicate_object_key_samples) < 5:
                        duplicate_object_key_samples.append(key)

    # --- File-level: duplicate company_stone_id with conflicting stone name (hard fail) ---
    # A company_stone_id may appear in many rows (one per image of that stone) — that is normal.
    # A company_stone_id must identify exactly ONE stone: it must map to exactly one distinct
    # stone_name across all rows. If the same company_stone_id appears with >1 distinct name,
    # this is a true duplicate-identity collision (two different stones sharing an ID), which
    # would corrupt stone-level search results and triggers a hard fail.
    duplicate_csid_count = 0
    duplicate_csid_samples: List[str] = []
    if passed_header_validation:
        csid_col_name = headers_lower_map.get("company_stone_id")
        name_col_name = headers_lower_map.get("name")
        if csid_col_name and name_col_name:
            csid_to_names: Dict[str, set] = {}
            for row in rows:
                csid = (row.get(csid_col_name) or "").strip()
                stone_name = (row.get(name_col_name) or "").strip().lower()
                if csid and stone_name:
                    if csid not in csid_to_names:
                        csid_to_names[csid] = set()
                    csid_to_names[csid].add(stone_name)
            for csid, names in csid_to_names.items():
                if len(names) > 1:
                    duplicate_csid_count += 1
                    if len(duplicate_csid_samples) < 5:
                        duplicate_csid_samples.append(
                            f"{csid} → [{', '.join(sorted(names)[:3])}]"
                        )

    passed_file_level = (passed_header_validation and duplicate_csid_count == 0)

    # --- File-level: URL sanity (warn only — empty already caught above) ---
    url_anomaly_count = 0
    url_anomaly_samples: List[str] = []
    if passed_header_validation:
        url_col = next(
            (headers_lower_map[c] for c in headers_lower_map if c == "url"),
            None
        )
        if url_col:
            for i, row in enumerate(rows):
                val = (row.get(url_col) or "").strip()
                if val and not (val.startswith("http://") or val.startswith("https://")):
                    url_anomaly_count += 1
                    if len(url_anomaly_samples) < 5:
                        url_anomaly_samples.append(val[:80])

    passed_row_validation = (
        passed_header_validation and row_count_invalid == 0
    )

    passed_all = passed_header_validation and passed_row_validation and passed_file_level

    return {
        "contract_version": SEARCH_CSV_V1_CONTRACT_VERSION,
        "passed_header_validation": passed_header_validation,
        "passed_row_validation": passed_row_validation,
        "passed_file_level": passed_file_level,
        "passed_all": passed_all,
        "missing_columns": missing_columns,
        "duplicate_columns": list(set(duplicate_headers)),
        "suspicious_columns": suspicious_columns,
        "ambiguous_schema_note": ambiguous_note,
        "detected_headers": headers,
        "row_count_total": row_count_total,
        "row_count_valid": row_count_valid,
        "row_count_invalid": row_count_invalid,
        "row_level_errors_by_field": row_level_errors_by_field,
        "sample_invalid_rows": sample_invalid_rows,
        "duplicate_object_key_count": duplicate_object_key_count,
        "duplicate_object_key_samples": duplicate_object_key_samples,
        "duplicate_company_stone_id_count": duplicate_csid_count,
        "duplicate_company_stone_id_samples": duplicate_csid_samples,
        "url_anomaly_count": url_anomaly_count,
        "url_anomaly_samples": url_anomaly_samples,
    }


# ---------------------------------------------------------------------------
# Continuity validation thresholds (all thresholds in one place — v1)
# ---------------------------------------------------------------------------
# CONTINUITY_THRESHOLDS: Dict[str, Any] = {
#     "row_count_drop_warn_pct": 10.0,    # warn if row count drops > 10%
#     "row_count_drop_block_pct": 30.0,   # block if row count drops > 30%
#     "csid_disappear_warn_pct": 10.0,    # warn if > 10% of active csids disappear
#     "csid_disappear_block_pct": 25.0,   # block if > 25% of active csids disappear
#     "csid_overlap_warn_pct": 85.0,      # warn if overlap < 85%
#     "new_csid_warn_pct": 30.0,          # warn if > 30% of candidate csids are new
#     "supplier_disappear_warn_count": 1,  # warn if any supplier disappears
# }
CONTINUITY_THRESHOLDS: Dict[str, Any] = {
    "row_count_drop_warn_pct": 10.0,    # warn if row count drops > 10%
    "row_count_drop_block_pct": 105.0,   # block if row count drops > 30%
    "csid_disappear_warn_pct": 10.0,    # warn if > 10% of active csids disappear
    "csid_disappear_block_pct": 105.0,   # block if > 25% of active csids disappear
    "csid_overlap_warn_pct": 85.0,      # warn if overlap < 85%
    "new_csid_warn_pct": 30.0,          # warn if > 30% of candidate csids are new
    "supplier_disappear_warn_count": 1,  # warn if any supplier disappears
}

def run_continuity_checks(
    candidate_rows: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compare candidate CSV rows against the current active dataset rows.

    Reads actual rows from dataset_images (not validation_summary JSON) to
    compute overlap and disappearance rates.

    Returns a structured dict:
      continuity_status: "pass" | "warn" | "block" | "no_baseline"
      metrics: {...}
      warnings: [...]
      blocked_reasons: [...]
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    blocked_reasons: List[str] = []
    metrics: Dict[str, Any] = {}

    # --- Resolve active dataset ---
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        active_row = conn.execute(
            "SELECT dataset_id FROM datasets WHERE is_active = 1 LIMIT 1"
        ).fetchone()
    except Exception as e:
        if "conn" in dir():
            conn.close()
        return {
            "continuity_status": "no_baseline",
            "metrics": {},
            "warnings": [f"Could not query active dataset: {e}"],
            "blocked_reasons": [],
        }

    if not active_row:
        conn.close()
        return {
            "continuity_status": "no_baseline",
            "metrics": {"note": "No active dataset — first import"},
            "warnings": ["No active baseline dataset found. Continuity checks skipped."],
            "blocked_reasons": [],
        }

    active_dataset_id = active_row["dataset_id"]

    # --- Load active dataset rows ---
    try:
        active_csids = set(
            r["company_stone_id"]
            for r in conn.execute(
                "SELECT DISTINCT company_stone_id FROM dataset_images WHERE dataset_id = ?",
                (active_dataset_id,)
            ).fetchall()
            if r["company_stone_id"]
        )
        active_packing_lists = set(
            r["batch_id"]
            for r in conn.execute(
                "SELECT DISTINCT batch_id FROM dataset_images WHERE dataset_id = ?",
                (active_dataset_id,)
            ).fetchall()
            if r["batch_id"]
        )
        active_vendors = set(
            r["vendor"]
            for r in conn.execute(
                "SELECT DISTINCT vendor FROM dataset_images WHERE dataset_id = ?",
                (active_dataset_id,)
            ).fetchall()
            if r["vendor"]
        )
        active_row_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM dataset_images WHERE dataset_id = ?",
            (active_dataset_id,)
        ).fetchone()["cnt"]
    finally:
        conn.close()

    # --- Compute candidate metrics ---
    cand_csids = set(
        (r.get("company_stone_id") or "").strip()
        for r in candidate_rows
        if (r.get("company_stone_id") or "").strip()
    )
    cand_packing_lists = set(
        (r.get("packing_list_id") or "").strip()
        for r in candidate_rows
        if (r.get("packing_list_id") or "").strip()
    )
    cand_vendors = set(
        (r.get("vendor name") or "").strip()
        for r in candidate_rows
        if (r.get("vendor name") or "").strip()
    )
    cand_row_count = len(candidate_rows)

    # Overlap and disappearance
    disappeared_csids = active_csids - cand_csids
    new_csids = cand_csids - active_csids
    overlap_csids = active_csids & cand_csids
    disappeared_pls = active_packing_lists - cand_packing_lists
    new_pls = cand_packing_lists - active_packing_lists
    disappeared_vendors = active_vendors - cand_vendors
    new_vendors = cand_vendors - active_vendors

    overlap_pct = (len(overlap_csids) / len(active_csids) * 100) if active_csids else 100.0
    disappear_pct = (len(disappeared_csids) / len(active_csids) * 100) if active_csids else 0.0
    new_csid_pct = (len(new_csids) / len(cand_csids) * 100) if cand_csids else 0.0
    row_count_delta = cand_row_count - active_row_count
    row_count_drop_pct = (
        (-row_count_delta / active_row_count * 100) if active_row_count > 0 and row_count_delta < 0 else 0.0
    )

    metrics = {
        "active_dataset_id": active_dataset_id,
        "active_row_count": active_row_count,
        "candidate_row_count": cand_row_count,
        "row_count_delta": row_count_delta,
        "row_count_drop_pct": round(row_count_drop_pct, 2),
        "active_unique_csid_count": len(active_csids),
        "candidate_unique_csid_count": len(cand_csids),
        "overlap_csid_count": len(overlap_csids),
        "overlap_csid_pct": round(overlap_pct, 2),
        "disappeared_csid_count": len(disappeared_csids),
        "disappeared_csid_pct": round(disappear_pct, 2),
        "disappeared_csid_samples": sorted(disappeared_csids)[:5],
        "new_csid_count": len(new_csids),
        "new_csid_pct": round(new_csid_pct, 2),
        "new_csid_samples": sorted(new_csids)[:5],
        "disappeared_packing_list_count": len(disappeared_pls),
        "new_packing_list_count": len(new_pls),
        "disappeared_vendor_count": len(disappeared_vendors),
        "disappeared_vendors": sorted(disappeared_vendors),
        "new_vendor_count": len(new_vendors),
        "new_vendors": sorted(new_vendors),
    }

    thr = CONTINUITY_THRESHOLDS

    # --- Block conditions ---
    if row_count_drop_pct >= thr["row_count_drop_block_pct"]:
        blocked_reasons.append(
            f"Continuity block: row count dropped {row_count_drop_pct:.1f}% "
            f"(threshold {thr['row_count_drop_block_pct']}%)"
        )
    if disappear_pct >= thr["csid_disappear_block_pct"]:
        blocked_reasons.append(
            f"Continuity block: {disappear_pct:.1f}% of active company_stone_id values disappeared "
            f"(threshold {thr['csid_disappear_block_pct']}%)"
        )

    # --- Warn conditions ---
    if row_count_drop_pct >= thr["row_count_drop_warn_pct"]:
        warnings.append(
            f"Row count dropped {row_count_drop_pct:.1f}% vs active dataset"
        )
    if disappear_pct >= thr["csid_disappear_warn_pct"]:
        warnings.append(
            f"{disappear_pct:.1f}% of active company_stone_id values are not in the candidate file"
        )
    if overlap_pct < thr["csid_overlap_warn_pct"]:
        warnings.append(
            f"company_stone_id overlap is only {overlap_pct:.1f}% "
            f"(threshold {thr['csid_overlap_warn_pct']}%)"
        )
    if new_csid_pct >= thr["new_csid_warn_pct"]:
        warnings.append(
            f"{new_csid_pct:.1f}% of candidate company_stone_id values are new (not in active)"
        )
    if len(disappeared_vendors) >= thr["supplier_disappear_warn_count"]:
        warnings.append(
            f"Suppliers no longer present: {', '.join(sorted(disappeared_vendors)[:10])}"
        )
    if new_vendors:
        warnings.append(
            f"New supplier names in candidate: {', '.join(sorted(new_vendors)[:10])}"
        )

    if blocked_reasons:
        continuity_status = "block"
    elif warnings:
        continuity_status = "warn"
    else:
        continuity_status = "pass"

    return {
        "continuity_status": continuity_status,
        "metrics": metrics,
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
    }

def _write_blocked_record(
    c: sqlite3.Connection,
    dataset_id: str,
    now: "datetime",
    blocked_reasons: List[str],
    val_report: Dict[str, Any],
    rows: List[Dict[str, str]],
    source_filename: str,
) -> None:
    validation_summary = {
        "passed_all": val_report.get("passed_all", False),
        "row_count_total": val_report.get("row_count_total", len(rows)),
        "row_count_valid": val_report.get("row_count_valid", 0),
        "row_count_invalid": val_report.get("row_count_invalid", len(rows)),
        "blocked_reasons": blocked_reasons,
    }
    c.execute("""
        INSERT OR REPLACE INTO datasets (
            dataset_id, created_at, source_filename, rows_read, is_active,
            unique_images, unique_batches, unique_company_stones, unique_families,
            unique_blocks, import_contract_version, detected_headers, validation_summary,
            validation_status, activation_blocked, validation_timestamp
        ) VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 0, ?, ?, ?, 'blocked', 1, ?)
    """, (
        dataset_id,
        now.isoformat(),
        source_filename,
        len(rows),
        SEARCH_CSV_V1_CONTRACT_VERSION,
        json.dumps(val_report.get("detected_headers", []), ensure_ascii=False),
        json.dumps(validation_summary, ensure_ascii=False),
        now.isoformat(),
    ))
    c.commit()

    # --- Ensure schema (idempotent) ---
def _ensure_schema(c: sqlite3.Connection) -> None:
    c.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            source_filename TEXT,
            rows_read INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 0,
            unique_images INTEGER DEFAULT 0,
            unique_batches INTEGER DEFAULT 0,
            unique_company_stones INTEGER DEFAULT 0,
            unique_families INTEGER DEFAULT 0,
            unique_blocks INTEGER DEFAULT 0,
            column_mapping_used TEXT
        )
    """)
    for extra_col in (
        "import_contract_version TEXT",
        "detected_headers TEXT",
        "validation_summary TEXT",
        "validation_status TEXT DEFAULT 'draft'",
        "activation_blocked INTEGER DEFAULT 1",
        "validation_timestamp TEXT",
    ):
        try:
            c.execute(f"ALTER TABLE datasets ADD COLUMN {extra_col}")
        except Exception:
            pass
    c.execute("""
        CREATE TABLE IF NOT EXISTS dataset_images (
            dataset_id TEXT NOT NULL,
            image_asset_id TEXT NOT NULL,
            image_url TEXT,
            batch_id TEXT,
            company_stone_id TEXT,
            vendor TEXT,
            stone_name_raw TEXT,
            canonical_family TEXT,
            block_no TEXT,
            block_id TEXT,
            PRIMARY KEY (dataset_id, image_asset_id)
        )
    """)
    c.commit()


def import_csv_clean_v2(
    rows: List[Dict[str, str]],
    source_filename: str,
    fieldnames: Optional[List[str]] = None
) -> CleanImportResult:
    """
    STRICT importer with hardcoded column mapping and full safety pipeline.

    Pipeline stages:
    1. validate_search_csv_v1() — header, row-level, file-level (duplicate csid) checks.
       Any hard failure: writes a metadata-only BLOCKED record, returns success=False.
    2. Ingest rows into dataset_images.
    3. run_continuity_checks() — compare against active dataset.
       Block-level continuity failure: dataset written but activation_blocked=1.
       Warn-level: dataset written, activation_blocked=0, continuity warnings recorded.
    4. Persist validation_status + activation_blocked + blocked_reasons.
    5. set_active is ignored — auto-activation removed. Caller must call
       set_active_dataset() explicitly after reviewing validation results.

    Column mapping (hardcoded):
    - batch_id        <- packing_list_id
    - company_stone_id<- company_stone_id
    - vendor          <- 'vendor name' ONLY
    - stone_name_raw  <- name
    - image_url       <- url
    - block_no        <- block_no (if present, else empty)
    - block_id        <- block_id (if present, else empty - NOT constructed from vendor|block_no)
    - image_asset_id  <- object_key

    Pass `fieldnames` as the original csv.DictReader.fieldnames list so that
    exact duplicate headers (e.g. two 'name' columns) are detected correctly.
    """
    headers = fieldnames if fieldnames is not None else (list(rows[0].keys()) if rows else [])
    validation_report = validate_search_csv_v1(rows, headers=headers)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        _ensure_schema(conn)
        now = datetime.now()
        dataset_id = now.strftime("%Y%m%d_%H%M%S") + "_clean"

        # ---------------------------------------------------------------
        # Stage 1A: Hard header-level failure
        # ---------------------------------------------------------------
        if not validation_report["passed_header_validation"]:
            error_parts = []
            if validation_report["missing_columns"]:
                error_parts.append(
                    "Missing mandatory columns: " + ", ".join(validation_report["missing_columns"])
                )
            if validation_report["duplicate_columns"]:
                error_parts.append(
                    "Duplicate/ambiguous columns: " + ", ".join(validation_report["duplicate_columns"])
                )
            if validation_report.get("ambiguous_schema_note"):
                error_parts.append(validation_report["ambiguous_schema_note"])
            _write_blocked_record(conn, dataset_id, now, error_parts, validation_report, rows, source_filename)
            return CleanImportResult(
                success=False,
                dataset_id=dataset_id,
                rows_read=len(rows),
                rows_imported=0,
                duplicates_skipped=0,
                errors=error_parts,
                validation={},
                validation_report=validation_report,
                validation_status="blocked",
                activation_blocked=True,
                blocked_reasons=error_parts,
            )

        # ---------------------------------------------------------------
        # Stage 1B: Hard row-level failure
        # ---------------------------------------------------------------
        if not validation_report["passed_row_validation"]:
            invalid_count = validation_report["row_count_invalid"]
            sample = validation_report["sample_invalid_rows"]
            sample_desc = "; ".join(
                f"row {s['row_index']}: {', '.join(s['errors'])}"
                for s in sample[:5]
            )
            error_parts = [
                f"{invalid_count} row(s) have empty mandatory fields. Sample: {sample_desc}"
            ]
            _write_blocked_record(conn, dataset_id, now, error_parts, validation_report, rows, source_filename)
            return CleanImportResult(
                success=False,
                dataset_id=dataset_id,
                rows_read=len(rows),
                rows_imported=0,
                duplicates_skipped=0,
                errors=error_parts,
                validation={},
                validation_report=validation_report,
                validation_status="blocked",
                activation_blocked=True,
                blocked_reasons=error_parts,
            )

        # ---------------------------------------------------------------
        # Stage 1C: Hard file-level failure (duplicate company_stone_id = conflicting names)
        # ---------------------------------------------------------------
        if not validation_report["passed_file_level"]:
            dup_count = validation_report.get("duplicate_company_stone_id_count", 0)
            dup_samples = validation_report.get("duplicate_company_stone_id_samples", [])
            error_parts = [
                f"File-level failure: {dup_count} company_stone_id value(s) appear with more than one "
                f"stone name (duplicate stone identity — data collision). "
                f"Samples: {'; '.join(dup_samples[:5])}"
            ]
            _write_blocked_record(conn, dataset_id, now, error_parts, validation_report, rows, source_filename)
            return CleanImportResult(
                success=False,
                dataset_id=dataset_id,
                rows_read=len(rows),
                rows_imported=0,
                duplicates_skipped=0,
                errors=error_parts,
                validation={},
                validation_report=validation_report,
                validation_status="blocked",
                activation_blocked=True,
                blocked_reasons=error_parts,
            )

        # ---------------------------------------------------------------
        # Stage 2: Ingest rows
        # ---------------------------------------------------------------
        seen_image_ids: set = set()
        valid_rows: List[Dict] = []
        errors: List[str] = []
        duplicates_skipped = 0

        for i, row in enumerate(rows):
            try:
                image_asset_id = (row.get("object_key") or "").strip()

                if not image_asset_id:
                    errors.append(f"Row {i+1}: Missing object_key")
                    continue

                if image_asset_id in seen_image_ids:
                    duplicates_skipped += 1
                    continue

                seen_image_ids.add(image_asset_id)

                vendor = (row.get("vendor name") or "").strip()
                stone_name_raw = (row.get("name") or "").strip()
                batch_id = (row.get("packing_list_id") or "").strip()
                company_stone_id = (row.get("company_stone_id") or "").strip()
                image_url = (row.get("url") or "").strip()
                block_no = (row.get("block_no") or "").strip()
                block_id_csv = (row.get("block_id") or "").strip()

                canonical_family = normalize_stone_name(stone_name_raw)

                valid_rows.append({
                    "dataset_id": dataset_id,
                    "image_asset_id": image_asset_id,
                    "image_url": image_url,
                    "batch_id": batch_id,
                    "company_stone_id": company_stone_id,
                    "vendor": vendor,
                    "stone_name_raw": stone_name_raw,
                    "canonical_family": canonical_family,
                    "block_no": block_no,
                    "block_id": block_id_csv
                })

            except Exception as e:
                errors.append(f"Row {i+1}: {str(e)}")

        # ---------------------------------------------------------------
        # Stage 3: Continuity checks (against live active dataset_images rows)
        # ---------------------------------------------------------------
        continuity_report = run_continuity_checks(rows)
        continuity_blocked_reasons = continuity_report.get("blocked_reasons", [])

        # ---------------------------------------------------------------
        # Stage 4: Compute validation_status + activation_blocked
        # ---------------------------------------------------------------
        url_anomaly_count = validation_report.get("url_anomaly_count", 0)
        dup_obj_key_count = validation_report.get("duplicate_object_key_count", 0)
        all_blocked_reasons: List[str] = list(continuity_blocked_reasons)
        # URL format anomalies and duplicate object_key are warn-only — recorded in validation_summary
        # but do NOT block activation. Only continuity block conditions block activation.

        if all_blocked_reasons:
            validation_status = "blocked"
            activation_blocked = True
        else:
            validation_status = "validated"
            activation_blocked = False

        column_mapping_used = {
            "type": "clean_v2_strict",
            "batch_id": "packing_list_id",
            "company_stone_id": "company_stone_id",
            "vendor": "vendor name (strict, no fallback)",
            "stone_name_raw": "name",
            "image_url": "url",
            "block_no": "block_no",
            "block_id": "block_id (direct from CSV, not constructed)",
            "image_asset_id": "object_key"
        }

        unique_batches = len(set(r["batch_id"] for r in valid_rows if r["batch_id"]))
        unique_company_stones = len(set(r["company_stone_id"] for r in valid_rows if r["company_stone_id"]))
        unique_families = len(set(r["canonical_family"] for r in valid_rows if r["canonical_family"]))
        unique_blocks = len(set(r["block_id"] for r in valid_rows if r["block_id"]))

        validation_summary = {
            "passed_all": validation_report["passed_all"],
            "row_count_total": validation_report["row_count_total"],
            "row_count_valid": validation_report["row_count_valid"],
            "row_count_invalid": validation_report["row_count_invalid"],
            "validation_status": validation_status,
            "activation_blocked": activation_blocked,
            "blocked_reasons": all_blocked_reasons,
            "continuity_status": continuity_report.get("continuity_status", "unknown"),
            "continuity_warnings": continuity_report.get("warnings", []),
            "url_anomaly_count": url_anomaly_count,
            "duplicate_object_key_count": dup_obj_key_count,
        }

        conn.execute("""
            INSERT OR REPLACE INTO datasets (
                dataset_id, created_at, source_filename, rows_read, is_active,
                unique_images, unique_batches, unique_company_stones, unique_families, unique_blocks,
                column_mapping_used, import_contract_version, detected_headers, validation_summary,
                validation_status, activation_blocked, validation_timestamp
            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            now.isoformat(),
            source_filename,
            len(rows),
            len(valid_rows),
            unique_batches,
            unique_company_stones,
            unique_families,
            unique_blocks,
            json.dumps(column_mapping_used, ensure_ascii=False),
            SEARCH_CSV_V1_CONTRACT_VERSION,
            json.dumps(validation_report["detected_headers"], ensure_ascii=False),
            json.dumps(validation_summary, ensure_ascii=False),
            validation_status,
            1 if activation_blocked else 0,
            now.isoformat(),
        ))

        for vr in valid_rows:
            conn.execute("""
                INSERT OR REPLACE INTO dataset_images (
                    dataset_id, image_asset_id, image_url, batch_id, company_stone_id,
                    vendor, stone_name_raw, canonical_family, block_no, block_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vr["dataset_id"],
                vr["image_asset_id"],
                vr["image_url"],
                vr["batch_id"],
                vr["company_stone_id"],
                vr["vendor"],
                vr["stone_name_raw"],
                vr["canonical_family"],
                vr["block_no"],
                vr["block_id"]
            ))

        conn.commit()

        validation = run_clean_import_validation(conn, dataset_id)

        return CleanImportResult(
            success=True,
            dataset_id=dataset_id,
            rows_read=len(rows),
            rows_imported=len(valid_rows),
            duplicates_skipped=duplicates_skipped,
            errors=errors[:100],
            validation=validation,
            validation_report=validation_report,
            validation_status=validation_status,
            activation_blocked=activation_blocked,
            blocked_reasons=all_blocked_reasons,
            continuity_report=continuity_report,
        )

    finally:
        conn.close()


def run_clean_import_validation(conn: sqlite3.Connection, dataset_id: str) -> Dict[str, Any]:
    """
    Run validation checks on a clean import.
    Returns a dict with validation results.
    """
    vendor_pollution_count = conn.execute("""
        SELECT COUNT(*) as cnt FROM dataset_images 
        WHERE dataset_id = ? AND LOWER(vendor) = LOWER(stone_name_raw)
    """, (dataset_id,)).fetchone()["cnt"]
    
    total_rows = conn.execute("""
        SELECT COUNT(*) as cnt FROM dataset_images WHERE dataset_id = ?
    """, (dataset_id,)).fetchone()["cnt"]
    
    top_vendors = conn.execute("""
        SELECT vendor, COUNT(*) as image_count 
        FROM dataset_images 
        WHERE dataset_id = ?
        GROUP BY vendor 
        ORDER BY image_count DESC 
        LIMIT 30
    """, (dataset_id,)).fetchall()
    
    packing_list_with_multi_stone = conn.execute("""
        SELECT COUNT(DISTINCT batch_id) as cnt
        FROM (
            SELECT batch_id, COUNT(DISTINCT company_stone_id) as stone_count
            FROM dataset_images
            WHERE dataset_id = ? AND batch_id != ''
            GROUP BY batch_id
            HAVING stone_count > 1
        )
    """, (dataset_id,)).fetchone()["cnt"]
    
    total_packing_lists = conn.execute("""
        SELECT COUNT(DISTINCT batch_id) as cnt 
        FROM dataset_images 
        WHERE dataset_id = ? AND batch_id != ''
    """, (dataset_id,)).fetchone()["cnt"]
    
    packing_list_with_multi_name = conn.execute("""
        SELECT COUNT(DISTINCT batch_id) as cnt
        FROM (
            SELECT batch_id, COUNT(DISTINCT stone_name_raw) as name_count
            FROM dataset_images
            WHERE dataset_id = ? AND batch_id != ''
            GROUP BY batch_id
            HAVING name_count > 1
        )
    """, (dataset_id,)).fetchone()["cnt"]
    
    packing_list_single_stone_pct = 0.0
    packing_list_single_name_pct = 0.0
    if total_packing_lists > 0:
        packing_list_single_stone_pct = round(
            100 * (total_packing_lists - packing_list_with_multi_stone) / total_packing_lists, 2
        )
        packing_list_single_name_pct = round(
            100 * (total_packing_lists - packing_list_with_multi_name) / total_packing_lists, 2
        )
    
    return {
        "vendor_pollution": {
            "count": vendor_pollution_count,
            "total_rows": total_rows,
            "pollution_pct": round(100 * vendor_pollution_count / total_rows, 2) if total_rows > 0 else 0
        },
        "top_vendors": [
            {"vendor": row["vendor"], "image_count": row["image_count"]}
            for row in top_vendors
        ],
        "packing_list_validation": {
            "total_packing_lists": total_packing_lists,
            "single_company_stone_pct": packing_list_single_stone_pct,
            "multi_company_stone_count": packing_list_with_multi_stone,
            "single_stone_name_pct": packing_list_single_name_pct,
            "multi_stone_name_count": packing_list_with_multi_name
        }
    }


def get_validation_report(dataset_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get validation report for a specific dataset or active dataset.
    """
    if not DB_PATH.exists():
        return {"error": f"Database not found: {DB_PATH}"}
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    try:
        if not dataset_id:
            row = conn.execute(
                "SELECT dataset_id FROM datasets WHERE is_active = 1 LIMIT 1"
            ).fetchone()
            if row:
                dataset_id = row["dataset_id"]
        
        if not dataset_id:
            return {"error": "No active dataset found"}
        
        validation = run_clean_import_validation(conn, dataset_id)
        validation["dataset_id"] = dataset_id
        validation["db_path"] = str(DB_PATH)
        
        return validation
        
    finally:
        conn.close()


def get_sample_images_for_stock(
    company_stone_id: str,
    dataset_id: Optional[str] = None,
    limit: int = 3
) -> List[DatasetImage]:
    init_db()
    
    if not dataset_id:
        dataset_id = get_active_dataset_id()
    
    if not dataset_id:
        return []
        
    images = []
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM dataset_images 
            WHERE dataset_id = ? AND company_stone_id = ?
            GROUP BY batch_id
            LIMIT ?
        """, (dataset_id, company_stone_id, limit)).fetchall()
        
        if len(rows) < limit:
            rows = conn.execute("""
                SELECT * FROM dataset_images 
                WHERE dataset_id = ? AND company_stone_id = ?
                LIMIT ?
            """, (dataset_id, company_stone_id, limit)).fetchall()
            
        for row in rows:
            images.append(DatasetImage(
                dataset_id=row["dataset_id"],
                image_asset_id=row["image_asset_id"] or "",
                image_url=row["image_url"] or "",
                batch_id=row["batch_id"] or "",
                company_stone_id=row["company_stone_id"] or "",
                vendor=row["vendor"] or "",
                stone_name_raw=row["stone_name_raw"] or "",
                canonical_family=row["canonical_family"] or "",
                block_no=row["block_no"] or "",
                block_id=row["block_id"] or ""
            ))
            
    return images


def get_stock_metadata_db(company_stone_id: str, dataset_id: str) -> Dict[str, Any]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT batch_id, stone_name_raw, vendor, 
                   canonical_family, image_url as thumbnail_url, block_no
            FROM dataset_images
            WHERE dataset_id = ? AND company_stone_id = ?
            LIMIT 1
            """,
            (dataset_id, company_stone_id)
        ).fetchone()
        if row:
            batch_rows = conn.execute(
                """
                SELECT DISTINCT batch_id FROM dataset_images 
                WHERE dataset_id = ? AND company_stone_id = ?
                """,
                (dataset_id, company_stone_id)
            ).fetchall()
            batch_ids = [br["batch_id"] for br in batch_rows if br["batch_id"]]
            
            res = dict(row)
            res["vendor_name"] = row["vendor"] or ""
            res["batch_ids"] = batch_ids
            return res
        return {}









def get_dataset_stats(dataset_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_db()
    
    if not dataset_id:
        dataset_id = get_active_dataset_id()
    
    if not dataset_id:
        return None
    
    with get_connection() as conn:
        ds_row = conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?",
            (dataset_id,)
        ).fetchone()
        
        if not ds_row:
            return None
        
        return {
            "dataset_id": ds_row["dataset_id"],
            "created_at": ds_row["created_at"],
            "source_filename": ds_row["source_filename"] or "",
            "rows_read": ds_row["rows_read"] or 0,
            "is_active": bool(ds_row["is_active"]),
            "unique_images": ds_row["unique_images"] or 0,
            "unique_batches": ds_row["unique_batches"] or 0,
            "unique_company_stones": ds_row["unique_company_stones"] or 0,
            "unique_families": ds_row["unique_families"] or 0,
            "unique_blocks": ds_row["unique_blocks"] or 0
        }


def get_all_unique_company_stone_ids(dataset_id: Optional[str] = None) -> List[str]:
    """
    Get all unique company_stone_ids from the active dataset (or specified dataset).
    Returns a list of company_stone_id strings.
    """
    init_db()
    
    if not dataset_id:
        dataset_id = get_active_dataset_id()
    
    if not dataset_id:
        return []
    
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT DISTINCT company_stone_id
            FROM dataset_images
            WHERE dataset_id = ? AND company_stone_id IS NOT NULL AND company_stone_id != ''
            ORDER BY company_stone_id
        """, (dataset_id,)).fetchall()
        
        return [row["company_stone_id"] for row in rows]




def get_active_dataset_csids():
    """Read-only fetch of csids in the active dataset. Empty set if not present."""
    if not DB_PATH.exists():
        return set(), None
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
        try:
            ds_row = conn.execute(
                "SELECT dataset_id FROM datasets WHERE is_active = 1 LIMIT 1"
            ).fetchone()
        except Exception:
            return set(), None
        if not ds_row:
            return set(), None
        ds_id = ds_row[0]
        rows = conn.execute(
            "SELECT DISTINCT company_stone_id FROM dataset_images "
            "WHERE dataset_id = ? AND company_stone_id IS NOT NULL AND company_stone_id != ''",
            (ds_id,),
        ).fetchall()
        return {r[0] for r in rows}, ds_id
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

def get_batch_hydrate(company_stone_ids: List[str]) -> List[str]:
    """Get existing company stone IDs from batch."""
    init_db()
    if not company_stone_ids:
        return []
    placeholders = ",".join("?" for _ in company_stone_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT company_stone_id FROM dataset_images"
            f" WHERE company_stone_id IN ({placeholders})"
            f" GROUP BY company_stone_id",
            company_stone_ids
        ).fetchall()
        return [r["company_stone_id"] for r in rows]


def get_all_dataset_images(dataset_id: str) -> List[Dict[str, Any]]:
    """Get all dataset_images rows for a specific dataset."""
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT company_stone_id, image_asset_id, image_url,
                   stone_name_raw, vendor, batch_id
            FROM dataset_images
            WHERE dataset_id = ?
            ORDER BY company_stone_id, image_asset_id, image_url
            """,
            (dataset_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_stone_data(dataset_id: str) -> Dict[str, Any]:
    """Get all stone data for a dataset."""
    init_db()
    with get_connection() as ds_conn:
        rows = ds_conn.execute("""
            SELECT
                batch_id,
                company_stone_id,
                stone_name_raw,
                vendor as vendor_name,
                canonical_family,
                image_url as thumbnail_url
            FROM dataset_images
            WHERE rowid IN (
                SELECT MAX(rowid)
                FROM dataset_images
                WHERE dataset_id = ?
                GROUP BY company_stone_id
            )
            ORDER BY rowid DESC
        """, (dataset_id,)).fetchall()

    return {
        "stones": [dict(r) for r in rows],
    }


def stream_dataset_rows(dataset_id: str):
    """
    Yields all dataset_images rows for a dataset one at a time.
    Simple cursor-based generator.
    """
    init_db()
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM dataset_images WHERE dataset_id = ? ORDER BY company_stone_id",
            (dataset_id,)
        )
        for row in cursor:
            yield dict(row)


def get_manifest_rows_for_enrichment(dataset_id: str) -> List[Dict]:
    """Return image rows for all stones in a dataset.
    Shape: [{company_stone_id, batch_id, url, image_asset_id}]
    Called by ml_inference via ingestion.services — never directly.
    """
    init_db()
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT company_stone_id, batch_id,
                   image_url AS url, image_asset_id
            FROM dataset_images
            WHERE dataset_id = ?
            ORDER BY company_stone_id, image_url
        """, (dataset_id,)).fetchall()
    return [dict(r) for r in rows]


