"""
Durable search query log — SQLite-backed persistence.

Complements the in-memory SEARCH_LOGS ring buffer (fingerprints_v3/search_router.py)
which is kept for the real-time /api/search/logs endpoint.  This module writes every
search event to data/search_logs.db so logs survive process restarts.

Schema (search_logs table):
    id                 INTEGER PRIMARY KEY AUTOINCREMENT
    search_request_id  TEXT NOT NULL          server-issued UUID per request
    timestamp          TEXT NOT NULL          ISO-8601 UTC
    query              TEXT NOT NULL
    filters_json       TEXT NOT NULL          JSON-serialised active filters
    mode               TEXT NOT NULL          text | image | similar | green_hunt
    result_count       INTEGER NOT NULL       total results before pagination
    latency_ms         REAL NOT NULL
    image_used         INTEGER NOT NULL       0 or 1
    dataset_id         TEXT                   active dataset at query time
    query_offset       INTEGER                pagination offset from request
    query_limit        INTEGER                pagination limit from request

Schema (search_log_results table — companion, one row per returned stone):
    id                 INTEGER PRIMARY KEY AUTOINCREMENT
    search_request_id  TEXT NOT NULL          FK → search_logs.search_request_id
    timestamp          TEXT NOT NULL          copied from search_logs row
    dataset_id         TEXT                   active dataset at query time
    company_stone_id   TEXT NOT NULL          stone identifier
    position           INTEGER NOT NULL       0-indexed global position (offset + page_index)
    page_size          INTEGER                query_limit for this request
    final_score        REAL                   ranking score at response time; NULL if unavailable
    rank_bucket        INTEGER                1=direct 2=rescued 3=other; NULL for image/green_hunt paths
    search_mode        TEXT NOT NULL          text | image | green_hunt

    Availability notes:
      final_score  — available in text search and image search (scorer sets stone["final_score"]).
                     NULL in green_hunt because candidates are not scored before pagination.
      rank_bucket  — available in text search only when hue-rescue is active (stone["_rank_bucket"]).
                     Set to 1 (default/all-equal bucket) when rescue is off in text search.
                     NULL in image search (no rescue logic) and green_hunt (not scored).

Feedback position validation (get_result_count_for_request):
    Looks up result_count for a given search_request_id so the feedback
    handler can validate result_position is in [0, result_count).
    Returns None if the request_id is unknown (not yet flushed, or too old).

search_request_id consistency guarantee:
    The same search_request_id is:
      - returned to the client in SearchResponse.search_request_id
      - stored in search_logs (one row per request)
      - stored in search_log_results (one row per returned stone)
      - accepted and validated by the /api/search/feedback endpoint
    This holds for all three covered paths: text search, image search, green_hunt.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def get_search_logs_db_path() -> Path:
    """Return path to search_logs.db — owned by search."""
    return Path(os.environ.get("SEARCH_LOGS_DB_PATH"))
_DB_PATH = get_search_logs_db_path()

_CREATE_SEARCH_LOGS_SQL = """
CREATE TABLE IF NOT EXISTS search_logs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    search_request_id  TEXT    NOT NULL,
    timestamp          TEXT    NOT NULL,
    query              TEXT    NOT NULL,
    filters_json       TEXT    NOT NULL,
    mode               TEXT    NOT NULL,
    result_count       INTEGER NOT NULL,
    latency_ms         REAL    NOT NULL,
    image_used         INTEGER NOT NULL DEFAULT 0,
    dataset_id         TEXT,
    query_offset       INTEGER,
    query_limit        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sl_request_id ON search_logs(search_request_id);
CREATE INDEX IF NOT EXISTS idx_sl_timestamp  ON search_logs(timestamp);
"""

_CREATE_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS search_log_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    search_request_id  TEXT    NOT NULL,
    timestamp          TEXT    NOT NULL,
    dataset_id         TEXT,
    company_stone_id   TEXT    NOT NULL,
    position           INTEGER NOT NULL,
    page_size          INTEGER,
    final_score        REAL,
    rank_bucket        INTEGER,
    search_mode        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slr_request_id  ON search_log_results(search_request_id);
CREATE INDEX IF NOT EXISTS idx_slr_stone       ON search_log_results(company_stone_id);
CREATE INDEX IF NOT EXISTS idx_slr_timestamp   ON search_log_results(timestamp);
"""

_MIGRATE_SEARCH_LOGS_COLUMNS = [
    "ALTER TABLE search_logs ADD COLUMN query_offset INTEGER",
    "ALTER TABLE search_logs ADD COLUMN query_limit  INTEGER",
]


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Create tables/indexes and apply migrations.  Safe to call multiple times."""
    try:
        with _get_conn() as conn:
            for sql_block in [_CREATE_SEARCH_LOGS_SQL, _CREATE_RESULTS_SQL]:
                for stmt in sql_block.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)

            for alter_sql in _MIGRATE_SEARCH_LOGS_COLUMNS:
                try:
                    conn.execute(alter_sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" in str(exc).lower():
                        pass
                    else:
                        logger.warning("search_log_db: migration skipped (%s): %s", alter_sql.split()[5], exc)
    except Exception:
        logger.exception("search_log_db: failed to initialise DB at %s", _DB_PATH)


def log_search(
    *,
    search_request_id: str,
    timestamp: str,
    query: str,
    filters: dict,
    mode: str,
    result_count: int,
    latency_ms: float,
    image_used: bool,
    dataset_id: Optional[str],
    query_offset: Optional[int] = None,
    query_limit: Optional[int] = None,
) -> None:
    """
    Persist one search event row.  Failures are logged and silently swallowed
    so that a DB write error never disrupts the search response.
    """
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO search_logs
                    (search_request_id, timestamp, query, filters_json,
                     mode, result_count, latency_ms, image_used, dataset_id,
                     query_offset, query_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    search_request_id,
                    timestamp,
                    query,
                    json.dumps(filters, default=str),
                    mode,
                    result_count,
                    round(latency_ms, 2),
                    1 if image_used else 0,
                    dataset_id,
                    query_offset,
                    query_limit,
                ),
            )
    except Exception:
        logger.exception(
            "search_log_db: failed to persist log entry for request_id=%s",
            search_request_id,
        )


def log_results(
    *,
    search_request_id: str,
    timestamp: str,
    dataset_id: Optional[str],
    search_mode: str,
    query_offset: int,
    page_size: int,
    stones: List[Dict[str, Any]],
) -> None:
    """
    Persist one row per returned stone, linked to the parent search_logs row.

    stones must be the paginated list in deterministic rank order (already sliced
    to the page that was returned to the client).  Each element is a raw stone dict
    as produced by the search pipeline (before _stone_to_result_v2 conversion).

    Field availability per path:
      final_score  — text search: stone["final_score"] (always set by scorer).
                     image search: stone["final_score"] (set by scorer).
                     green_hunt: NULL (candidates are not scored before pagination).
      rank_bucket  — text search: stone["_rank_bucket"] (1/2/3 when rescue active,
                     always 1 when rescue is off).
                     image search: NULL (no hue-rescue logic in image path).
                     green_hunt: NULL (not scored).

    Failures are logged and silently swallowed so a DB error never disrupts the
    search response.
    """
    if not stones:
        return
    try:
        now = timestamp
        rows = []
        for i, stone in enumerate(stones):
            global_position = query_offset + i
            final_score = stone.get("final_score")
            rank_bucket = stone.get("_rank_bucket")
            rows.append((
                search_request_id,
                now,
                dataset_id,
                stone.get("company_stone_id", ""),
                global_position,
                page_size,
                round(float(final_score), 6) if final_score is not None else None,
                int(rank_bucket) if rank_bucket is not None else None,
                search_mode,
            ))
        with _get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO search_log_results
                    (search_request_id, timestamp, dataset_id, company_stone_id,
                     position, page_size, final_score, rank_bucket, search_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    except Exception:
        logger.exception(
            "search_log_db: failed to persist result snapshot for request_id=%s",
            search_request_id,
        )


def ping() -> bool:
    """
    Lightweight DB readiness probe.  Executes a real SQL query (not just a connection open)
    so the health endpoint can surface actual DB failures rather than just import-time errors.
    Returns True on success, False on any exception.
    """
    try:
        with _get_conn() as conn:
            conn.execute("SELECT 1 FROM search_logs LIMIT 1")
        return True
    except Exception:
        logger.exception("search_log_db: ping failed")
        return False


