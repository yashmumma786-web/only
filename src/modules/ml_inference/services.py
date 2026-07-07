import asyncio
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple

# EnrichmentPipelineResult imported dynamically via type hint string to resolve circular dependency
from src.modules.ml_inference import embeddings as ml_embeddings
from src.modules.ml_inference import cv_analyzer as ml_cv_analyzer
from src.modules.ml_inference import color_agent
from src.modules.ml_inference import vein_agent
from src.modules.ml_inference.foundation_storage import (
    foundation_storage,
    JobType,
    JobStatus,
)
from src.modules.ml_inference.foundation_worker import (
    get_worker,
    start_worker_if_needed,
    rebuild_all_aggregates,
)
from src.modules.ml_inference import enrichment_flow
from src.modules.ml_inference import repository as _repo
from src.modules.ml_inference.embedding_cache import (
    get_embedding as _get,
    load_all_embeddings as _load,
)


def compute_image_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """Generates the CLIP embedding of an image locally."""
    return ml_embeddings.compute_image_embedding(image)


def _analyze_image_tracked(
    image_path: str, max_pixels: int = 4000000, mode: str = "default"
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Runs CV analyzer with cache tracking locally."""

    return ml_cv_analyzer._analyze_image_tracked(
        image_path, max_pixels=max_pixels, mode=mode
    )


def analyze_stock_colors_from_arrays(images: List[np.ndarray]) -> Any:
    """Processes color signatures of a list of image pixel arrays locally."""
    return color_agent.analyze_stock_colors_from_arrays(images)


def classify_vein_colour_from_images(images: List[np.ndarray]) -> Optional[str]:
    """Classifies vein color signature from a list of image pixel arrays locally."""
    return vein_agent.classify_vein_colour_from_images(images)


# --- Enrichment Orchestration Constants & State ---


def start_foundation_compute(
    image_pairs: List[Tuple[str, str]], sync_first: bool = True
) -> dict:
    """Syncs image pairs and starts the foundation embedding computation job."""

    images_synced = 0
    if sync_first and image_pairs:
        images_synced = foundation_storage.bulk_upsert_images(image_pairs)

    foundation_storage.reset_all_to_pending()
    job_id = foundation_storage.create_job(JobType.FOUNDATION_COMPUTE)
    get_worker().start()
    job = foundation_storage.get_active_job()

    return {
        "job_id": job_id,
        "message": f"Job started. Processing {job.total if job else 0} images.",
        "total_images": job.total if job else 0,
        "images_synced": images_synced,
    }


def stop_foundation_compute() -> dict:
    """Stops the active embedding worker and cancels the job."""
    worker = get_worker()
    worker.stop()
    job = foundation_storage.get_active_job()
    if job:
        foundation_storage.finish_job(job.id, JobStatus.CANCELED)
    return {"message": "Worker stopped", "job_canceled": job.id if job else None}


def resume_foundation_compute() -> dict:
    """Resumes the pending foundation embedding jobs."""
    started = start_worker_if_needed()
    return {"started": started, "pending": foundation_storage.get_pending_count()}


def get_foundation_status() -> dict:
    """Gets the current status and metrics of foundation compute job."""
    job = foundation_storage.get_latest_job()
    counts = foundation_storage.get_image_counts()

    total = sum(counts.values())
    done = counts.get("DONE", 0)
    failed = counts.get("FAILED", 0)
    pending = counts.get("PENDING", 0) + counts.get("PROCESSING", 0)

    percent = (done / total * 100) if total > 0 else 0.0
    worker = get_worker()
    recent_errors = foundation_storage.get_recent_errors(10)

    return {
        "job_id": job.id if job else None,
        "status": job.status.value if job else None,
        "total": total,
        "done": done,
        "failed": failed,
        "pending": pending,
        "percent": round(percent, 2),
        "worker_running": worker.is_running,
        "recent_errors": recent_errors,
    }


def rebuild_foundation_aggregates() -> dict:
    """Rebuilds stock representation embedding aggregates."""

    return asyncio.run(rebuild_all_aggregates())


def retry_failed_foundation_images() -> dict:
    """Resets failed images to pending and triggers worker."""
    failed_count = foundation_storage.get_failed_count()
    failure_stats = foundation_storage.get_failure_stats()
    reset_count = foundation_storage.reset_failed_to_pending()
    started = start_worker_if_needed()

    return {
        "reset_count": reset_count,
        "previous_failure_stats": failure_stats,
        "worker_started": started,
    }


def start_foundation_worker() -> bool:
    """Facade to start the foundation background worker if there is pending work."""
    return start_worker_if_needed()


def get_foundation_failure_stats() -> dict:
    """Gets the list of failed foundation jobs breakdown."""

    stats = foundation_storage.get_failure_stats()
    total = foundation_storage.get_failed_count()
    return {"total_failed": total, "by_reason": stats}


def get_foundation_stock_aggregate(company_stone_id: str) -> dict:
    """Gets visual stock representations for similar query UI."""
    aggregate = foundation_storage.get_stock_aggregate(company_stone_id)
    if not aggregate:
        return {}

    return {
        "company_stone_id": aggregate.company_stone_id,
        "rep_image_url": aggregate.rep_image_url,
        "consistency_score": aggregate.consistency_score,
        "outlier_count": aggregate.outlier_count,
        "image_count": aggregate.image_count,
        "updated_at": (
            aggregate.updated_at.isoformat() if aggregate.updated_at else None
        ),
    }


def compute_embedding_aggregates(
    embeddings: np.ndarray, outlier_threshold: float = 0.78
) -> Dict[str, Any]:
    """Pure mathematical function to compute centroids and cosine similarities."""
    return ml_embeddings.compute_embedding_aggregates(embeddings, outlier_threshold)


def run_enrichment_pipeline(
    dataset_id: str,
    row_generator,
    mode: str,
    dry_run: bool = False,
    use_model_predictions: bool = False,
    stop_after_cv: bool = False,
) -> "enrichment_flow.EnrichmentPipelineResult":
    """Facade to run dataset enrichment from a row generator."""

    rows = list(row_generator)
    return enrichment_flow.run_enrichment_pipeline(
        dataset_id=dataset_id,
        rows=rows,
        mode=mode,
        dry_run=dry_run,
        use_model_predictions=use_model_predictions,
        stop_after_cv=stop_after_cv,
    )


def enrichment_plan_and_enqueue_after_import(rows: List[dict], dataset_id: str) -> dict:
    """Facade to calculate enrichment plan and enqueue images after CSV import."""
    return enrichment_flow.enrichment_plan_and_enqueue_after_import(rows, dataset_id)


def get_enrichment_progress(dataset_id: str) -> Optional[dict]:
    """Facade to read persisted progress for a dataset."""
    return _repo.load_persisted_progress(dataset_id)


def initialize_enrichment_progress(dataset_id: str) -> None:
    """Facade to initialize persisted progress state as queued."""
    _repo.init_progress_table()
    _repo.persist_progress(dataset_id, {"status": "queued", "stage": "queued"})


def get_foundation_csid_sets() -> tuple:
    """Return (done_csids, all_known_csids) from foundation DB.

    done_csids      — csids with at least one DONE image (embedding complete)
    all_known_csids — all csids in foundation DB regardless of status

    Both reads are read-only. Returns (set(), set()) if DB does not exist.
    Called by orchestrator so that ingestion never opens foundation.db directly.
    """

    if not foundation_storage.db_path.exists():
        return set(), set()
    try:
        done = set(foundation_storage.get_all_stock_ids_with_done_images())
        all_known = foundation_storage.get_all_known_csids()
        return done, all_known
    except Exception:
        return set(), set()


def get_embedding(company_stone_id: str):
    """Return numpy embedding for a single stone, or None if not found.

    Backed by TTL in-memory cache (300s). Called by search/similar modules
    through this public facade — they never import embedding_cache directly.
    """
    return _get(company_stone_id)


def load_all_embeddings() -> dict:
    """Return all embeddings keyed by company_stone_id.

    TTL-cached (300s). Called by search/similar modules through this facade.
    """
    return _load()


def cleanup_stale_runs() -> None:
    """Facade to cleanup and mark stale runs as interrupted."""
    _repo.mark_stale_runs_interrupted()
