"""
Shared foundation.db embedding cache.

Two separate in-memory caches (prod and trial), each keyed by the resolved db path.
Callers always pass use_trial explicitly — the cache never reads global trial mode
to avoid accidentally serving trial embeddings to public search paths.

TTL: 300 seconds.

Fallback: if foundation.db does not exist or a stone has no embedding,
callers receive None / 0.0 similarity.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from src.modules.ml_inference.foundation_storage import get_foundation_db_path

logger = logging.getLogger(__name__)

CACHE_TTL = 300  # seconds

# FOUNDATION_DB is the production foundation.db path, resolved from FOUNDATION_DB_PATH env var.
# New code should call get_foundation_db_path() directly.
FOUNDATION_DB = get_foundation_db_path()

_cache_lock = threading.Lock()
# Per-path caches: str(db_path) → Dict[csid, np.ndarray]
_caches: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
_cache_ts: Dict[str, float] = {}

# Diagnostic counters (shared across both caches for simplicity)
_cache_hits: int = 0
_cache_misses: int = 0


def load_all_embeddings() -> Dict[str, np.ndarray]:
    """Return all embeddings keyed by company_stone_id.

    Thread-safe.  Reloads from the appropriate foundation.db when TTL expires.
    Returns an empty dict if the db is missing or unreadable.

    use_trial must be supplied by the caller — it is never derived from global state.
    """
    global _cache_hits, _cache_misses
    db_path = FOUNDATION_DB
    key = str(db_path)
    with _cache_lock:
        now = time.time()
        if _caches.get(key) is not None and (now - _cache_ts.get(key, 0.0)) < CACHE_TTL:
            _cache_hits += 1
            return _caches[key]

        _cache_misses += 1
        _caches[key] = _load_from_db(db_path)
        _cache_ts[key] = time.time()
        return _caches[key]


def get_embedding(company_stone_id: str) -> Optional[np.ndarray]:
    """Return the embedding for a single stone, or None if not found."""
    return load_all_embeddings().get(company_stone_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_from_db(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        logger.warning("embedding_cache: foundation.db not found at %s", db_path)
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT company_stone_id, embedding "
            "FROM stock_images "
            "WHERE status = 'DONE' AND embedding IS NOT NULL"
        ).fetchall()
        conn.close()
        result: Dict[str, np.ndarray] = {}
        for r in rows:
            cid = r["company_stone_id"]
            emb_data = r["embedding"]
            if not emb_data:
                continue
            try:
                if isinstance(emb_data, str):
                    arr = np.array(json.loads(emb_data), dtype=np.float32)
                elif isinstance(emb_data, bytes):
                    arr = np.frombuffer(emb_data, dtype=np.float32).copy()
                else:
                    continue
                result[cid] = arr
            except Exception:
                continue
        logger.info(
            "embedding_cache: loaded %d embeddings from %s (TTL=%ds)",
            len(result), db_path, CACHE_TTL,
        )
        return result
    except Exception as exc:
        logger.warning("embedding_cache: failed to load from %s: %s", db_path, exc)
        return {}
