"""
SQLite storage for Fingerprints V3 - NEW VERSION

Handles:
- fingerprints_v3 table for computed values
- fingerprint_overrides table for manual edits
- Merge logic for returning final values
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from .schema_v3 import FingerprintV3New

def get_fingerprints_db_path() -> Path:
    """Return path to fingerprints_v3.db — owned by taxonomy."""
    return Path(os.environ.get("FINGERPRINTS_DB_PATH"))
DB_PATH = get_fingerprints_db_path()

def ensure_db() -> "sqlite3.Connection":
    """Create database and tables if they don't exist, return open connection."""
    target = DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fingerprints_v3 (
            company_stone_id TEXT PRIMARY KEY,
            base_color TEXT,
            base_color_conf REAL DEFAULT 0,
            colors_in_stone_json TEXT DEFAULT '[]',
            colors_in_stone_conf REAL DEFAULT 0,
            tonality TEXT,
            tonality_conf REAL DEFAULT 0,
            design TEXT,
            design_conf REAL DEFAULT 0,
            vein_color_json TEXT DEFAULT '[]',
            vein_color_conf REAL DEFAULT 0,
            vein_direction TEXT,
            vein_direction_conf REAL DEFAULT 0,
            vein_distribution TEXT,
            vein_distribution_conf REAL DEFAULT 0,
            vein_thickness TEXT,
            vein_thickness_conf REAL DEFAULT 0,
            spot_color_json TEXT DEFAULT '[]',
            spot_color_conf REAL DEFAULT 0,
            spot_distribution TEXT,
            spot_distribution_conf REAL DEFAULT 0,
            representative_images_json TEXT DEFAULT '{}',
            debug_metrics_json TEXT DEFAULT '{}',
            image_count INTEGER DEFAULT 0,
            usable_image_count INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        );
        
    """)
    
    conn.commit()
    return conn


@contextmanager
def get_db():
    """Context manager for database connections.

    Existing callers with no argument use the production fingerprints_v3.db.
    Trial pipeline callers pass db_path=Path("cache/fingerprints_v3/trial_fingerprints_v3.db").
    """
    conn = ensure_db()
    try:
        yield conn
    finally:
        conn.close()


def save_fingerprint(fp: FingerprintV3New):
    """Save or update a computed fingerprint."""
    with get_db() as conn:
        now = datetime.now().isoformat()
        
        existing = conn.execute(
            "SELECT created_at FROM fingerprints_v3 WHERE company_stone_id = ?",
            (fp.company_stone_id,)
        ).fetchone()
        
        created_at = existing["created_at"] if existing else now
        
        conn.execute("""
            INSERT OR REPLACE INTO fingerprints_v3 (
                company_stone_id,
                base_color, base_color_conf,
                colors_in_stone_json, colors_in_stone_conf,
                tonality, tonality_conf,
                design, design_conf,
                vein_color_json, vein_color_conf,
                vein_direction, vein_direction_conf,
                vein_distribution, vein_distribution_conf,
                vein_thickness, vein_thickness_conf,
                spot_color_json, spot_color_conf,
                spot_distribution, spot_distribution_conf,
                representative_images_json, debug_metrics_json,
                image_count, usable_image_count,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fp.company_stone_id,
            fp.base_color, fp.base_color_conf,
            json.dumps(fp.colors_in_stone), fp.colors_in_stone_conf,
            fp.tonality, fp.tonality_conf,
            fp.design, fp.design_conf,
            json.dumps(fp.vein_color), fp.vein_color_conf,
            fp.vein_direction, fp.vein_direction_conf,
            fp.vein_distribution, fp.vein_distribution_conf,
            fp.vein_thickness, fp.vein_thickness_conf,
            json.dumps(fp.spot_color), fp.spot_color_conf,
            fp.spot_distribution, fp.spot_distribution_conf,
            json.dumps(fp.representative_images.to_dict()),
            json.dumps(fp.debug_metrics.to_dict()),
            fp.image_count, fp.usable_image_count,
            created_at, now
        ))
        conn.commit()



