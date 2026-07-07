"""
Sibling Votes Storage - manages voting on Similar Sibling relationships.

Key design:
- pair_key = min(id_a, id_b) | max(id_a, id_b) - ensures symmetry
- A vote on (A,B) automatically applies to (B,A)
- No duplicate directional rows allowed

Vote types:
- UP (1): Verified relationship
- DOWN (-1): Suppressed/rejected relationship
"""

import sqlite3
import os
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timezone


def get_sibling_votes_db_path() -> Path:
    """Return path to sibling_votes.db — owned by taxonomy."""
    return Path(os.environ.get("SIBLING_VOTES_DB_PATH"))
DB_PATH = get_sibling_votes_db_path()

def _ensure_db():
    """Create the database and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sibling_votes (
                pair_key TEXT PRIMARY KEY,
                company_stone_id_a TEXT NOT NULL,
                company_stone_id_b TEXT NOT NULL,
                vote INTEGER NOT NULL,
                admin_user_id TEXT,
                voted_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sibling_votes_a 
            ON sibling_votes(company_stone_id_a)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sibling_votes_b 
            ON sibling_votes(company_stone_id_b)
        """)
        conn.commit()


def make_pair_key(id_a: str, id_b: str) -> Tuple[str, str, str]:
    """
    Create canonical pair_key from two IDs.
    Returns (pair_key, normalized_id_a, normalized_id_b) where a < b.
    """
    if id_a < id_b:
        return f"{id_a}|{id_b}", id_a, id_b
    else:
        return f"{id_b}|{id_a}", id_b, id_a


def set_vote(
    company_stone_id_a: str,
    company_stone_id_b: str,
    vote: int,
    admin_user_id: Optional[str] = None
) -> bool:
    """
    Set a vote on a sibling pair. 
    Vote is automatically normalized to canonical ordering.
    
    Args:
        company_stone_id_a: First stone ID
        company_stone_id_b: Second stone ID  
        vote: 1 for UP (verified), -1 for DOWN (suppressed)
        admin_user_id: Optional admin identifier
    
    Returns:
        True if vote was saved successfully
    """
    if company_stone_id_a == company_stone_id_b:
        return False
    
    if vote not in (1, -1):
        return False
    
    _ensure_db()
    
    pair_key, norm_a, norm_b = make_pair_key(company_stone_id_a, company_stone_id_b)
    now = datetime.now(timezone.utc).isoformat()
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sibling_votes (pair_key, company_stone_id_a, company_stone_id_b, vote, admin_user_id, voted_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pair_key) DO UPDATE SET
                vote = excluded.vote,
                admin_user_id = excluded.admin_user_id,
                updated_at_utc = excluded.updated_at_utc
        """, (pair_key, norm_a, norm_b, vote, admin_user_id, now, now))
        conn.commit()
    
    return True

def get_suppressed_siblings(stone_id: str) -> List[str]:
    """Get all stone IDs that have been marked as 'not related' (vote = -1) to the given stone."""
    _ensure_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT company_stone_id_b FROM sibling_votes WHERE company_stone_id_a = ? AND vote = -1
            UNION
            SELECT company_stone_id_a FROM sibling_votes WHERE company_stone_id_b = ? AND vote = -1
        """, (stone_id, stone_id)).fetchall()
        return [row[0] for row in rows]

