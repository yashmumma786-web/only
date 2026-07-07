"""Repository Layer for ML Inference Module

Centralizes all raw SQL/SQLite operations to decouple business logic
and API routes from direct database connections.
"""

from typing import Dict, List, Optional, Set
import json
import datetime
import pandas as pd
import sqlite3
import os
import time
import time as _t2
from src.modules.ingestion import services as ingestion_svc
from pathlib import Path

def get_enrichment_db_path() -> Path:
    """Return path to enrichment.db — owned by ml_inference."""
    return Path(os.environ.get("ENRICHMENT_DB_PATH"))


DB_PATH = get_enrichment_db_path()


def init_progress_table() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_progress (
            dataset_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            stage TEXT,
            stage_done INTEGER DEFAULT 0,
            stage_total INTEGER DEFAULT 0,
            overall_done INTEGER DEFAULT 0,
            overall_total INTEGER DEFAULT 0,
            current_stone TEXT,
            started_at REAL,
            updated_at REAL,
            error TEXT,
            extra TEXT
        )
    """
    )
    conn.commit()
    conn.close()
    try:
        ensure_report_schema()
    except Exception:
        pass


def persist_progress(dataset_id: str, progress: Dict) -> None:

    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        extra = {}
        if "classification" in progress:
            extra["classification"] = progress["classification"]
        if "stages_completed" in progress:
            extra["stages_completed"] = progress["stages_completed"]
        for _k in (
            "run_id",
            "coverage_pct",
            "missing_count",
            "coverage_field",
            "run_report_url",
        ):
            if _k in progress and progress[_k] is not None:
                extra[_k] = progress[_k]

        # Handle any other extra keys if coming from the worker
        for k, v in progress.items():
            if k not in (
                "status",
                "stage",
                "stage_done",
                "stage_total",
                "overall_done",
                "overall_total",
                "current_stone",
                "started_at",
                "error",
                "classification",
                "stages_completed",
                "run_id",
                "coverage_pct",
                "missing_count",
                "coverage_field",
                "run_report_url",
            ):
                extra[k] = v

        conn.execute(
            """
            INSERT INTO enrichment_progress
                (dataset_id, status, stage, stage_done, stage_total,
                 overall_done, overall_total, current_stone,
                 started_at, updated_at, error, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                status=excluded.status, stage=excluded.stage,
                stage_done=excluded.stage_done, stage_total=excluded.stage_total,
                overall_done=excluded.overall_done, overall_total=excluded.overall_total,
                current_stone=excluded.current_stone,
                updated_at=excluded.updated_at, error=excluded.error,
                extra=excluded.extra
        """,
            (
                dataset_id,
                progress.get("status", "running"),
                progress.get("stage"),
                progress.get("stage_done", 0),
                progress.get("stage_total", 0),
                progress.get("overall_done", 0),
                progress.get("overall_total", 0),
                progress.get("current_stone"),
                progress.get("started_at"),
                time.time(),
                progress.get("error"),
                json.dumps(extra) if extra else None,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def load_persisted_progress(dataset_id: str) -> Optional[Dict]:

    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=2)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT * FROM enrichment_progress WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        cols = set(row.keys())
        heartbeat_at = row["heartbeat_at"] if "heartbeat_at" in cols else None
        run_id_val = row["run_id"] if "run_id" in cols else None
        result = {
            "status": row["status"],
            "stage": row["stage"],
            "stage_done": row["stage_done"] or 0,
            "stage_total": row["stage_total"] or 0,
            "overall_done": row["overall_done"] or 0,
            "overall_total": row["overall_total"] or 0,
            "current_stone": row["current_stone"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "error": row["error"],
            "finished_at": (
                row["updated_at"]
                if row["status"] in ("done", "error", "interrupted")
                else None
            ),
            "last_heartbeat_at": heartbeat_at,
            "run_id": run_id_val,
        }
        if row["extra"]:
            try:
                extra = json.loads(row["extra"])
                result.update(extra)
            except Exception:
                pass


        STALE = 180.0
        result["interrupted"] = result["status"] == "interrupted"
        if result["status"] == "running":

            now = _t2.time()
            ref = max(
                float(heartbeat_at or 0),
                float(result.get("updated_at") or 0),
                float(result.get("started_at") or 0),
            )
            if ref and now - ref > STALE:
                result["status"] = "interrupted"
                result["interrupted"] = True
                result["finished_at"] = ref
        if result["status"] == "error":
            result["interrupted"] = False
        result.setdefault("coverage_pct", None)
        result.setdefault("missing_count", None)
        result.setdefault("run_report_url", None)
        result["last_updated"] = result.get("updated_at")
        result["last_heartbeat_at"] = heartbeat_at
        return result
    except Exception:
        return None


def mark_stale_runs_interrupted() -> None:

    stale = 180.0
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enrichment_progress (
                dataset_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT,
                stage_done INTEGER DEFAULT 0,
                stage_total INTEGER DEFAULT 0,
                overall_done INTEGER DEFAULT 0,
                overall_total INTEGER DEFAULT 0,
                current_stone TEXT,
                started_at REAL,
                updated_at REAL,
                error TEXT,
                extra TEXT
            )
        """
        )
        try:
            ensure_report_schema()
        except Exception:
            pass
        now = time.time()
        conn.execute(
            """UPDATE enrichment_progress
               SET status='interrupted', updated_at=?
               WHERE status='running'
                 AND (? - MAX(
                       COALESCE(heartbeat_at, 0),
                       COALESCE(updated_at, 0),
                       COALESCE(started_at, 0)
                     )) > ?
                 AND MAX(
                       COALESCE(heartbeat_at, 0),
                       COALESCE(updated_at, 0),
                       COALESCE(started_at, 0)
                     ) > 0""",
            (now, now, stale),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# --- Manifest / Dataset Queries ---


def load_manifest_df(dataset_id: str) -> "pd.DataFrame":
    rows = ingestion_svc.get_manifest_rows_for_enrichment(dataset_id)
    return pd.DataFrame(rows)



# --- Resume Ledger Queries ---


def init_resume_table() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_resume (
            dataset_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            company_stone_id TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, stage, company_stone_id)
        )
    """
    )
    conn.commit()
    conn.close()


def mark_stage_done(dataset_id: str, stage_name: str, stone_id: str) -> None:

    try:
        rc = sqlite3.connect(str(DB_PATH), timeout=5)
        rc.execute(
            "INSERT OR IGNORE INTO enrichment_resume (dataset_id, stage, company_stone_id, completed_at) VALUES (?,?,?,?)",
            (dataset_id, stage_name, stone_id, time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        rc.commit()
        rc.close()
    except Exception:
        pass


def load_resume_set(dataset_id: str, stage_name: str) -> Set[str]:
    try:
        rc = sqlite3.connect(str(DB_PATH), timeout=5)
        try:
            rows = rc.execute(
                """SELECT company_stone_id FROM enrichment_resume
                   WHERE dataset_id=? AND stage=?
                     AND (status IS NULL OR status='ok')""",
                (dataset_id, stage_name),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = rc.execute(
                "SELECT company_stone_id FROM enrichment_resume WHERE dataset_id=? AND stage=?",
                (dataset_id, stage_name),
            ).fetchall()
        rc.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def clear_resume(dataset_id: str, stage_name: str = None) -> None:
    try:
        rc = sqlite3.connect(str(DB_PATH), timeout=5)
        if stage_name:
            rc.execute(
                "DELETE FROM enrichment_resume WHERE dataset_id=? AND stage=?",
                (dataset_id, stage_name),
            )
        else:
            rc.execute(
                "DELETE FROM enrichment_resume WHERE dataset_id=?",
                (dataset_id,),
            )
        rc.commit()
        rc.close()
    except Exception:
        pass


def mark_stones_resume_status(
    dataset_id: str, stage: str, stone_ids: List[str], status: str
) -> None:

    try:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        rc = sqlite3.connect(str(DB_PATH), timeout=10)
        for sid in stone_ids:
            rc.execute(
                "INSERT OR REPLACE INTO enrichment_resume "
                "(dataset_id, stage, company_stone_id, completed_at, status) "
                "VALUES (?,?,?,?,?)",
                (dataset_id, stage, sid, now_iso, status),
            )
        rc.commit()
        rc.close()
    except Exception:
        pass


# --- Enrichment Report Queries ---


def ensure_report_schema() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            conn.execute(
                "ALTER TABLE enrichment_resume ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'"
            )
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            conn.execute("ALTER TABLE enrichment_progress ADD COLUMN heartbeat_at REAL")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE enrichment_progress ADD COLUMN run_id TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enrichment_failures (
                run_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                company_stone_id TEXT NOT NULL,
                error_class TEXT,
                error_message TEXT,
                image_count_attempted INTEGER DEFAULT 0,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (run_id, dataset_id, stage, company_stone_id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def record_failure(
    *,
    run_id: str,
    dataset_id: str,
    stage: str,
    company_stone_id: str,
    error_class: str,
    error_message: str,
    image_count_attempted: int = 0,
) -> None:

    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT OR REPLACE INTO enrichment_failures
               (run_id, dataset_id, stage, company_stone_id,
                error_class, error_message, image_count_attempted, recorded_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                run_id,
                dataset_id,
                stage,
                company_stone_id,
                error_class[:200],
                (error_message or "")[:2000],
                int(image_count_attempted or 0),
                datetime.datetime.now(datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
                + "Z",
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
