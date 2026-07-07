"""
Foundation Compute Storage - SQLite tables for image embeddings and job tracking.

Tables:
- stock_images: Per-image embedding state with atomic claiming
- jobs: Job tracking for background processing
- stocks: Stock-level aggregates (centroid, rep image, consistency)

Key features:
- Atomic claim mechanism with claim_token to prevent double-processing
- Stale lock recovery (PROCESSING > 5 min → PENDING)
- Idempotent: DONE images are never recomputed
"""

import sqlite3
import json
import uuid
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


def get_foundation_db_path() -> Path:
    """Return path to foundation.db — owned by ml_inference."""
    return Path(os.environ.get("FOUNDATION_DB_PATH"))
DB_PATH = get_foundation_db_path()

STALE_LOCK_MINUTES = 5
OUTLIER_THRESHOLD = 0.78
MAX_RETRIES = 3


class ImageStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class JobType(str, Enum):
    FOUNDATION_COMPUTE = "FOUNDATION_COMPUTE"


@dataclass
class StockImage:
    id: int
    company_stone_id: str
    image_url: str
    embedding: Optional[List[float]]
    embedding_model: Optional[str]
    status: ImageStatus
    retries: int
    error: Optional[str]
    claimed_at: Optional[datetime]
    claim_token: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class Job:
    id: int
    type: JobType
    status: JobStatus
    total: int
    done: int
    failed: int
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class StockAggregate:
    company_stone_id: str
    rep_image_id: Optional[int]
    rep_image_url: Optional[str]
    centroid_embedding: Optional[List[float]]
    consistency_score: Optional[float]
    outlier_count: int
    image_count: int
    updated_at: datetime


class FoundationStorage:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_stone_id TEXT NOT NULL,
                    image_url TEXT NOT NULL UNIQUE,
                    embedding TEXT,
                    embedding_model TEXT,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    retries INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    claimed_at TEXT,
                    claim_token TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_images_company_stone_id 
                ON stock_images(company_stone_id)
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_stock_images_image_url 
                ON stock_images(image_url)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_images_status_claimed 
                ON stock_images(status, claimed_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_images_claim_token 
                ON stock_images(claim_token)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'QUEUED',
                    total INTEGER NOT NULL DEFAULT 0,
                    done INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status 
                ON jobs(status)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    company_stone_id TEXT PRIMARY KEY,
                    rep_image_id INTEGER,
                    centroid_embedding TEXT,
                    consistency_score REAL,
                    outlier_count INTEGER DEFAULT 0,
                    image_count INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (rep_image_id) REFERENCES stock_images(id)
                )
            """)
            
            conn.commit()
    
    def bulk_upsert_images(self, images: List[Tuple[str, str]]) -> int:
        """Bulk insert/update images. images = [(company_stone_id, image_url), ...]"""
        inserted = 0
        with self._get_conn() as conn:
            for company_stone_id, image_url in images:
                try:
                    conn.execute("""
                        INSERT INTO stock_images (company_stone_id, image_url, status)
                        VALUES (?, ?, 'PENDING')
                        ON CONFLICT(image_url) DO UPDATE SET
                            company_stone_id = excluded.company_stone_id,
                            updated_at = datetime('now')
                    """, (company_stone_id, image_url))
                    inserted += 1
                except Exception:
                    pass
            conn.commit()
        return inserted
    
    def claim_batch_pending(self, batch_size: int = 4) -> List[StockImage]:
        """Claim up to batch_size PENDING images atomically."""
        claim_token = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self._get_conn() as conn:
            cursor = conn.execute("""
                UPDATE stock_images
                SET status = 'PROCESSING',
                    claimed_at = ?,
                    claim_token = ?,
                    updated_at = ?
                WHERE id IN (
                    SELECT id FROM stock_images
                    WHERE status = 'PENDING'
                    ORDER BY id
                    LIMIT ?
                )
                RETURNING *
            """, (now, claim_token, now, batch_size))
            
            rows = cursor.fetchall()
            conn.commit()
            
            return [self._row_to_stock_image(row) for row in rows]
    
    def recover_stale_locks(self, stale_minutes: int = STALE_LOCK_MINUTES) -> int:
        """Reset PROCESSING images older than stale_minutes back to PENDING."""
        cutoff = (datetime.utcnow() - timedelta(minutes=stale_minutes)).isoformat()
        
        with self._get_conn() as conn:
            cursor = conn.execute("""
                UPDATE stock_images
                SET status = 'PENDING',
                    claimed_at = NULL,
                    claim_token = NULL,
                    updated_at = datetime('now')
                WHERE status = 'PROCESSING'
                AND claimed_at < ?
            """, (cutoff,))
            recovered = cursor.rowcount
            conn.commit()
            return recovered
    
    def mark_done(self, image_id: int, embedding: List[float], model: str = "open_clip:ViT-B-32"):
        """Mark an image as DONE with its embedding."""
        embedding_json = json.dumps(embedding)
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE stock_images
                SET status = 'DONE',
                    embedding = ?,
                    embedding_model = ?,
                    error = NULL,
                    updated_at = datetime('now')
                WHERE id = ?
            """, (embedding_json, model, image_id))
            conn.commit()
    
    def mark_failed(self, image_id: int, error: str):
        """Increment retries, mark FAILED if max retries exceeded, else PENDING."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT retries FROM stock_images WHERE id = ?", (image_id,)
            ).fetchone()
            
            if not row:
                return
            
            retries = row["retries"] + 1
            new_status = "FAILED" if retries >= MAX_RETRIES else "PENDING"
            
            conn.execute("""
                UPDATE stock_images
                SET status = ?,
                    retries = ?,
                    error = ?,
                    claimed_at = NULL,
                    claim_token = NULL,
                    updated_at = datetime('now')
                WHERE id = ?
            """, (new_status, retries, error, image_id))
            conn.commit()
    
    def get_image_counts(self) -> Dict[str, int]:
        """Get counts by status."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as cnt
                FROM stock_images
                GROUP BY status
            """)
            return {row["status"]: row["cnt"] for row in cursor.fetchall()}
    
    def get_pending_count(self) -> int:
        """Get count of PENDING images."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM stock_images WHERE status = 'PENDING'"
            ).fetchone()
            return row["cnt"] if row else 0
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent failed images."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT id, image_url, error, retries, updated_at
                FROM stock_images
                WHERE status = 'FAILED'
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def reset_all_to_pending(self):
        """Reset all non-DONE images to PENDING. Used when starting a fresh job."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE stock_images
                SET status = 'PENDING',
                    claimed_at = NULL,
                    claim_token = NULL,
                    retries = 0,
                    error = NULL,
                    updated_at = datetime('now')
                WHERE status != 'DONE'
            """)
            conn.commit()
    
    def create_job(self, job_type: JobType = JobType.FOUNDATION_COMPUTE) -> int:
        """Create a new job and return its ID."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs SET status = 'CANCELED', updated_at = datetime('now')
                WHERE status IN ('QUEUED', 'RUNNING') AND type = ?
            """, (job_type.value,))
            
            counts = self.get_image_counts()
            total = sum(counts.get(s, 0) for s in ["PENDING", "PROCESSING", "FAILED"])
            done = counts.get("DONE", 0)
            
            cursor = conn.execute("""
                INSERT INTO jobs (type, status, total, done, failed, started_at)
                VALUES (?, 'RUNNING', ?, ?, 0, datetime('now'))
                RETURNING id
            """, (job_type.value, total + done, done))
            
            job_id = cursor.fetchone()["id"]
            conn.commit()
            return job_id
    
    def get_active_job(self, job_type: JobType = JobType.FOUNDATION_COMPUTE) -> Optional[Job]:
        """Get the currently running job."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM jobs
                WHERE type = ? AND status = 'RUNNING'
                ORDER BY id DESC
                LIMIT 1
            """, (job_type.value,)).fetchone()
            
            if row:
                return self._row_to_job(row)
            return None
    
    def get_latest_job(self, job_type: JobType = JobType.FOUNDATION_COMPUTE) -> Optional[Job]:
        """Get the most recent job regardless of status."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM jobs
                WHERE type = ?
                ORDER BY id DESC
                LIMIT 1
            """, (job_type.value,)).fetchone()
            
            if row:
                return self._row_to_job(row)
            return None
    
    def update_job_progress(self, job_id: int):
        """Update job progress counters from actual image counts."""
        counts = self.get_image_counts()
        done = counts.get("DONE", 0)
        failed = counts.get("FAILED", 0)
        
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs
                SET done = ?, failed = ?, updated_at = datetime('now')
                WHERE id = ?
            """, (done, failed, job_id))
            conn.commit()
    
    def finish_job(self, job_id: int, status: JobStatus = JobStatus.DONE):
        """Mark a job as finished."""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE jobs
                SET status = ?, finished_at = datetime('now'), updated_at = datetime('now')
                WHERE id = ?
            """, (status.value, job_id))
            conn.commit()
    
    def get_done_embeddings_for_stock(self, company_stone_id: str) -> List[Tuple[int, str, List[float]]]:
        """Get all DONE embeddings for a stock. Returns [(id, url, embedding), ...]"""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT id, image_url, embedding
                FROM stock_images
                WHERE company_stone_id = ? AND status = 'DONE' AND embedding IS NOT NULL
            """, (company_stone_id,))
            
            results = []
            for row in cursor.fetchall():
                try:
                    emb = json.loads(row["embedding"])
                    results.append((row["id"], row["image_url"], emb))
                except:
                    pass
            return results
    
    def get_all_stock_ids_with_done_images(self) -> List[str]:
        """Get all company_stone_ids that have at least one DONE image."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT company_stone_id
                FROM stock_images
                WHERE status = 'DONE' AND embedding IS NOT NULL
            """)
            return [row["company_stone_id"] for row in cursor.fetchall()]

    def get_all_known_csids(self) -> set:
        """Get ALL company_stone_ids regardless of status (for overlap analysis).

        Uses a read-only connection so the analyser can never accidentally
        mutate foundation data.
        """
        if not self.db_path.exists():
            return set()
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT DISTINCT company_stone_id FROM stock_images"
            )
            result = {row["company_stone_id"] for row in cursor.fetchall()}
            conn.close()
            return result
        except Exception:
            return set()


    def save_stock_aggregate(
        self,
        company_stone_id: str,
        rep_image_id: Optional[int],
        centroid_embedding: Optional[List[float]],
        consistency_score: Optional[float],
        outlier_count: int,
        image_count: int
    ):
        """Save stock-level aggregate data."""
        centroid_json = json.dumps(centroid_embedding) if centroid_embedding else None
        
        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO stocks (
                    company_stone_id, rep_image_id, centroid_embedding,
                    consistency_score, outlier_count, image_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(company_stone_id) DO UPDATE SET
                    rep_image_id = excluded.rep_image_id,
                    centroid_embedding = excluded.centroid_embedding,
                    consistency_score = excluded.consistency_score,
                    outlier_count = excluded.outlier_count,
                    image_count = excluded.image_count,
                    updated_at = datetime('now')
            """, (company_stone_id, rep_image_id, centroid_json, consistency_score, outlier_count, image_count))
            conn.commit()
    
    def get_stock_aggregate(self, company_stone_id: str) -> Optional[StockAggregate]:
        """Get stock aggregate data."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT s.*, si.image_url as rep_image_url
                FROM stocks s
                LEFT JOIN stock_images si ON s.rep_image_id = si.id
                WHERE s.company_stone_id = ?
            """, (company_stone_id,)).fetchone()
            
            if row:
                return self._row_to_stock_aggregate(row)
            return None
    
    def _row_to_stock_image(self, row: sqlite3.Row) -> StockImage:
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except:
                pass
        
        return StockImage(
            id=row["id"],
            company_stone_id=row["company_stone_id"],
            image_url=row["image_url"],
            embedding=embedding,
            embedding_model=row["embedding_model"],
            status=ImageStatus(row["status"]),
            retries=row["retries"],
            error=row["error"],
            claimed_at=datetime.fromisoformat(row["claimed_at"]) if row["claimed_at"] else None,
            claim_token=row["claim_token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )
    
    def _row_to_job(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            type=JobType(row["type"]),
            status=JobStatus(row["status"]),
            total=row["total"],
            done=row["done"],
            failed=row["failed"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"])
        )
    
    def _row_to_stock_aggregate(self, row: sqlite3.Row) -> StockAggregate:
        centroid = None
        if row["centroid_embedding"]:
            try:
                centroid = json.loads(row["centroid_embedding"])
            except:
                pass
        
        return StockAggregate(
            company_stone_id=row["company_stone_id"],
            rep_image_id=row["rep_image_id"],
            rep_image_url=row["rep_image_url"] if "rep_image_url" in row.keys() else None,
            centroid_embedding=centroid,
            consistency_score=row["consistency_score"],
            outlier_count=row["outlier_count"],
            image_count=row["image_count"],
            updated_at=datetime.fromisoformat(row["updated_at"])
        )
    
    def reset_failed_to_pending(self) -> int:
        """Reset all FAILED images back to PENDING for retry."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                UPDATE stock_images
                SET status = 'PENDING',
                    claimed_at = NULL,
                    claim_token = NULL,
                    retries = 0,
                    error = NULL,
                    updated_at = datetime('now')
                WHERE status = 'FAILED'
            """)
            count = cursor.rowcount
            conn.commit()
            return count
    
    def get_failure_stats(self) -> Dict[str, int]:
        """Get counts of failures grouped by reason (extracted from error field)."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT error FROM stock_images WHERE status = 'FAILED'
            """)
            
            stats = {}
            for row in cursor.fetchall():
                error = row["error"] or "unknown"
                reason = error.split("|")[0] if "|" in error else error[:50]
                stats[reason] = stats.get(reason, 0) + 1
            
            return stats
    
    def get_failed_count(self) -> int:
        """Get count of FAILED images."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM stock_images WHERE status = 'FAILED'"
            ).fetchone()
            return row["cnt"] if row else 0


foundation_storage = FoundationStorage()
