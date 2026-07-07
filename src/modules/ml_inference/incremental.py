"""
Incremental CSV Enrichment — Classification and Stage Invalidation

Compares incoming CSV rows against the currently active dataset via the
Ingestion REST Facade and classifies each stone as:

  NEW             — company_stone_id not present in the active dataset
  CHANGED_IMAGE   — image_asset_id set or image_url values changed
  CHANGED_METADATA — only stone_name_raw / vendor / batch_id changed (image set identical)
  UNCHANGED       — both image hash and metadata hash are identical to the active baseline
  INVALID         — company_stone_id is missing or empty

The active dataset is resolved via the Ingestion HTTP facade — no direct DB access.
No new table or schema change is required.

Normalization contract (applied before hashing):
  - strip() all string fields
  - treat None and "" identically (normalize to "")
  - preserve case on all fields
  - sort image rows by (image_asset_id, image_url) for deterministic ordering
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from src.modules.ingestion import services as ingestion_service
from src.modules.ml_inference import embeddings as v3_embeddings


CHANGE_CLASS_NEW = "NEW"
CHANGE_CLASS_CHANGED_IMAGE = "CHANGED_IMAGE"
CHANGE_CLASS_CHANGED_METADATA = "CHANGED_METADATA"
CHANGE_CLASS_UNCHANGED = "UNCHANGED"
CHANGE_CLASS_INVALID = "INVALID"


def _norm(value: Optional[str]) -> str:
    """Normalize a single field value before hashing.

    Rules:
    - None → ""
    - strip leading/trailing whitespace
    - preserve case (canonical_family handles lowercasing downstream)
    """
    if value is None:
        return ""
    return str(value).strip()


def _compute_image_hash(image_rows: List[Tuple[str, str]]) -> str:
    """Compute a deterministic hash over a stone's image set.

    image_rows: list of (image_asset_id, image_url) tuples (any order).

    Normalization: each value is _norm'd; pairs are sorted lexicographically so
    row ordering in the CSV does not affect the hash.

    Returns a 16-char hex string (MD5 prefix).
    """
    sorted_pairs = sorted((_norm(asset_id), _norm(url)) for asset_id, url in image_rows)
    payload = json.dumps(sorted_pairs, separators=(",", ":"), ensure_ascii=True)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _compute_metadata_hash(stone_name_raw: str, vendor: str, batch_id: str) -> str:
    """Compute a deterministic hash over a stone's non-image metadata.

    Fields included and why:
    - stone_name_raw: feeds canonical_family via normalize_stone_name(), which
      drives Section A (cross-vendor inferred name matches) of similar-stones.
      A name change may change which stones are offered as "Similar" for this
      stone in Section A.
    - vendor: gates Section A by requiring vendor != anchor_vendor (cross-
      vendor only).  A vendor change shifts which name-match candidates are
      eligible to appear in Section A.
    - batch_id: retained in the metadata hash for downstream signature
      stability and audit even though Task #312 removed the legacy
      same_batch / same_stock linked sections from the similar-stones API.

    Returns a 16-char hex string (MD5 prefix).
    """
    payload = json.dumps(
        [_norm(stone_name_raw), _norm(vendor), _norm(batch_id)],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:16]


@dataclass
class StoneSignature:
    """Hashed identity fingerprint for one company_stone_id's row set."""

    company_stone_id: str
    image_hash: str
    metadata_hash: str
    image_count: int


def _build_signature_from_db_rows(
    company_stone_id: str,
    rows: List[Dict],
) -> StoneSignature:
    """Build a StoneSignature from a list of dataset_images rows for one stone."""
    image_pairs = [(r.get("image_asset_id"), r.get("image_url")) for r in rows]

    stone_name_raw = rows[0].get("stone_name_raw") if rows else ""
    vendor = rows[0].get("vendor") if rows else ""
    batch_id = rows[0].get("batch_id") if rows else ""

    return StoneSignature(
        company_stone_id=company_stone_id,
        image_hash=_compute_image_hash(image_pairs),
        metadata_hash=_compute_metadata_hash(stone_name_raw, vendor, batch_id),
        image_count=len(image_pairs),
    )


def load_active_signatures() -> Dict[str, StoneSignature]:
    """Load StoneSignature for every company_stone_id in the currently active dataset.

    Returns an empty dict when:
    - no dataset is marked is_active = 1
    - any API error occurs (logged, not raised)

    Source of truth: Ingestion REST Facade.
    (db_path override is deprecated and ignored)
    """
    try:
        active_dataset_id = ingestion_service.get_active_dataset_id()
        if not active_dataset_id:
            return {}

        rows = ingestion_service.get_all_dataset_images(active_dataset_id)
        if not rows:
            return {}
    except Exception as exc:
        print(f"[incremental] load_active_signatures error: {exc}")
        return {}

    by_stone: Dict[str, List[Dict]] = {}
    for row in rows:
        csid = row.get("company_stone_id") or ""
        if not csid:
            continue
        by_stone.setdefault(csid, []).append(row)

    return {
        csid: _build_signature_from_db_rows(csid, stone_rows)
        for csid, stone_rows in by_stone.items()
    }


@dataclass
class StoneClassification:
    """Classification result for one company_stone_id."""

    company_stone_id: str
    change_class: str
    image_hash_incoming: Optional[str] = None
    image_hash_baseline: Optional[str] = None
    metadata_hash_incoming: Optional[str] = None
    metadata_hash_baseline: Optional[str] = None
    incoming_image_count: int = 0
    baseline_image_count: int = 0
    invalid_reason: Optional[str] = None

    @property
    def is_unchanged(self) -> bool:
        return self.change_class == CHANGE_CLASS_UNCHANGED

    @property
    def requires_cv(self) -> bool:
        return self.change_class in (CHANGE_CLASS_NEW, CHANGE_CLASS_CHANGED_IMAGE)

    @property
    def requires_embeddings(self) -> bool:
        return self.change_class in (CHANGE_CLASS_NEW, CHANGE_CLASS_CHANGED_IMAGE)

    @property
    def requires_aggregation(self) -> bool:
        return self.change_class in (
            CHANGE_CLASS_NEW,
            CHANGE_CLASS_CHANGED_IMAGE,
            CHANGE_CLASS_CHANGED_METADATA,
        )

    def to_dict(self) -> Dict:
        return {
            "company_stone_id": self.company_stone_id,
            "change_class": self.change_class,
            "image_hash_incoming": self.image_hash_incoming,
            "image_hash_baseline": self.image_hash_baseline,
            "metadata_hash_incoming": self.metadata_hash_incoming,
            "metadata_hash_baseline": self.metadata_hash_baseline,
            "incoming_image_count": self.incoming_image_count,
            "baseline_image_count": self.baseline_image_count,
            "invalid_reason": self.invalid_reason,
            "requires_cv": self.requires_cv,
            "requires_embeddings": self.requires_embeddings,
            "requires_aggregation": self.requires_aggregation,
        }


@dataclass
class IncrementalRunSummary:
    """Aggregate counts and per-stone lists for one incremental classification run."""

    classified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    baseline_dataset_id: Optional[str] = None
    total_input_rows: int = 0
    total_stones: int = 0

    new_count: int = 0
    changed_image_count: int = 0
    changed_metadata_count: int = 0
    unchanged_count: int = 0
    invalid_count: int = 0

    new_ids: List[str] = field(default_factory=list)
    changed_image_ids: List[str] = field(default_factory=list)
    changed_metadata_ids: List[str] = field(default_factory=list)
    invalid_ids: List[str] = field(default_factory=list)
    invalid_reasons: Dict[str, str] = field(default_factory=dict)

    stage_cv_required: int = 0
    stage_embeddings_required: int = 0
    stage_aggregation_required: int = 0

    classifications: Dict[str, StoneClassification] = field(default_factory=dict)

    def to_dict(self, include_full_classifications: bool = False) -> Dict:
        result = {
            "classified_at": self.classified_at,
            "baseline_dataset_id": self.baseline_dataset_id,
            "total_input_rows": self.total_input_rows,
            "total_stones": self.total_stones,
            "counts": {
                "new": self.new_count,
                "changed_image": self.changed_image_count,
                "changed_metadata": self.changed_metadata_count,
                "unchanged": self.unchanged_count,
                "invalid": self.invalid_count,
            },
            "ids": {
                "new": self.new_ids,
                "changed_image": self.changed_image_ids,
                "changed_metadata": self.changed_metadata_ids,
                "invalid": self.invalid_ids,
            },
            "invalid_reasons": self.invalid_reasons,
            "stages_required": {
                "cv_analysis": self.stage_cv_required,
                "embeddings": self.stage_embeddings_required,
                "aggregation": self.stage_aggregation_required,
            },
        }
        if include_full_classifications:
            result["classifications"] = {
                csid: c.to_dict() for csid, c in self.classifications.items()
            }
        return result


def classify_csv_rows(
    incoming_rows: List[Dict],
) -> IncrementalRunSummary:
    """Classify each stone in incoming_rows against the active baseline.

    incoming_rows: raw CSV row dicts using Search CSV v1 field names
    (object_key, url, company_stone_id, name, vendor name, packing_list_id, ...).

    Returns an IncrementalRunSummary with per-stone StoneClassification objects
    and aggregate counts.
    """
    baseline = load_active_signatures()
    baseline_dataset_id: Optional[str] = None

    try:
        baseline_dataset_id = ingestion_service.get_active_dataset_id() or None
    except Exception:
        pass

    by_stone: Dict[str, List[Dict]] = {}
    for raw_row in incoming_rows:
        csid = _norm(raw_row.get("company_stone_id") or "")
        by_stone.setdefault(csid, []).append(raw_row)

    summary = IncrementalRunSummary(
        baseline_dataset_id=baseline_dataset_id,
        total_input_rows=len(incoming_rows),
    )

    for csid, rows in by_stone.items():
        if not csid:
            summary.invalid_count += len(rows)
            if "" not in summary.invalid_ids:
                summary.invalid_ids.append("")
            summary.invalid_reasons["<empty>"] = "company_stone_id is missing or empty"
            continue

        incoming_sig = _build_signature_from_db_rows(csid, rows)

        if csid not in baseline:
            cl = StoneClassification(
                company_stone_id=csid,
                change_class=CHANGE_CLASS_NEW,
                image_hash_incoming=incoming_sig.image_hash,
                metadata_hash_incoming=incoming_sig.metadata_hash,
                incoming_image_count=incoming_sig.image_count,
            )
            summary.new_count += 1
            summary.new_ids.append(csid)
        else:
            base_sig = baseline[csid]
            image_changed = incoming_sig.image_hash != base_sig.image_hash
            meta_changed = incoming_sig.metadata_hash != base_sig.metadata_hash

            if image_changed:
                change_class = CHANGE_CLASS_CHANGED_IMAGE
                summary.changed_image_count += 1
                summary.changed_image_ids.append(csid)
            elif meta_changed:
                change_class = CHANGE_CLASS_CHANGED_METADATA
                summary.changed_metadata_count += 1
                summary.changed_metadata_ids.append(csid)
            else:
                change_class = CHANGE_CLASS_UNCHANGED
                summary.unchanged_count += 1

            cl = StoneClassification(
                company_stone_id=csid,
                change_class=change_class,
                image_hash_incoming=incoming_sig.image_hash,
                image_hash_baseline=base_sig.image_hash,
                metadata_hash_incoming=incoming_sig.metadata_hash,
                metadata_hash_baseline=base_sig.metadata_hash,
                incoming_image_count=incoming_sig.image_count,
                baseline_image_count=base_sig.image_count,
            )

        summary.classifications[csid] = cl

    summary.total_stones = len(by_stone)
    summary.stage_cv_required = summary.new_count + summary.changed_image_count
    summary.stage_embeddings_required = summary.new_count + summary.changed_image_count
    summary.stage_aggregation_required = (
        summary.new_count + summary.changed_image_count + summary.changed_metadata_count
    )

    return summary


def get_stones_requiring_stage(
    summary: IncrementalRunSummary,
    stage: str,
) -> List[str]:
    """Return the list of company_stone_ids that need a given stage rerun.

    stage must be one of: "cv", "embeddings", "aggregation"

    Invalid stones are never included.
    """
    if stage == "cv":
        return [csid for csid, cl in summary.classifications.items() if cl.requires_cv]
    if stage == "embeddings":
        return [
            csid
            for csid, cl in summary.classifications.items()
            if cl.requires_embeddings
        ]
    if stage == "aggregation":
        return [
            csid
            for csid, cl in summary.classifications.items()
            if cl.requires_aggregation
        ]
    raise ValueError(
        f"Unknown stage: {stage!r}. Expected 'cv', 'embeddings', or 'aggregation'."
    )


def invalidate_embedding_cache_for_stones(stone_ids: List[str]) -> Dict[str, bool]:
    """Delete cached .npy embedding files for the given stones.

    This forces Stage 3a (v3_embeddings build_all_embeddings) to recompute
    embeddings for these stones even when force_rebuild=False, working around
    the G-11 stale-embedding gap (skip keyed only by file existence).

    Returns {company_stone_id: deleted} for each requested stone.
    """

    results: Dict[str, bool] = {}
    for csid in stone_ids:
        npy_path = v3_embeddings.EMBEDDINGS_CACHE_DIR / f"{csid}.npy"
        try:
            if npy_path.exists():
                npy_path.unlink()
                results[csid] = True
            else:
                results[csid] = False
        except Exception as exc:
            print(f"[incremental] Could not delete .npy for {csid}: {exc}")
            results[csid] = False
    return results
