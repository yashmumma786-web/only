"""
Orchestrator Service Layer — the ONLY file allowed to import from
multiple domain services simultaneously.

Follows the exact pattern from senior's pseudocode:
  import_csv()     → ingestion.import_new_data  → self.ml_enrichment()
  ml_enrichment()  → ingestion.get_data()       → ml_inference.enrich()
  recompute()      → ml_inference.get_data()    → taxonomy.recompute()
  activate()       → ingestion.activate_dataset()
"""

import asyncio
from typing import Optional
from src.modules.ingestion import services as ingestion_svc
from src.modules.ml_inference import services as ml_svc
from src.modules.taxonomy import services as taxonomy_svc
from src.modules.search import services as search_svc
from src.modules.similar import services as similar_svc
from src.modules.orchestrator.models import (
    FoundationStatusResponse,
    FoundationFailureStatsResponse,
    FoundationStockAggregateResponse,
    StartFoundationComputeResponse,
    StopFoundationComputeResponse,
    ResumeFoundationComputeResponse,
    RetryFailedFoundationResponse,
    ImportResponse,
    EnrichmentResponse,
)
from src.modules.similar.models import RecordSiblingVoteResponse


def import_csv(csv_bytes: bytes, filename: str) -> ImportResponse:
    """
    Parses CSV bytes, saves to Ingestion DB, then triggers ML enrichment.
    """

    result = ingestion_svc.import_csv_from_bytes(csv_bytes, filename)
    if not result.get("success"):
        return ImportResponse(**result)

    dataset_id = result["dataset_id"]
    ml_result = ml_enqueue_after_import(dataset_id)

    # Merge the two domain outputs directly into the composed response
    return ImportResponse(**result, **ml_result)


def ml_enqueue_after_import(dataset_id: str) -> dict:
    """
    Streams dataset images and enqueues them for ML enrichment.
    """

    stone_stream = ingestion_svc.stream_dataset_rows(dataset_id)
    db_rows = list(stone_stream)

    ml_result = ml_svc.enrichment_plan_and_enqueue_after_import(db_rows, dataset_id)

    # Check for hard fail in enqueueing
    if ml_result.get("foundation_enqueue", {}).get("status") == "hard_fail":
        return {
            "success": False,
            "error": (
                f"Import wrote dataset_images rows but failed to enqueue "
                f"to foundation: {ml_result['foundation_enqueue'].get('error', 'Unknown')}. Investigate and re-run."
            ),
            **ml_result,
        }

    return ml_result


def ml_enrichment(
    dataset_id: str,
    mode: str,
    dry_run: bool,
    use_model_predictions: bool,
    stop_after_cv: bool,
) -> EnrichmentResponse:
    """from src.modules.ml_inference import services as ml_svc

    Runs ML enrichment using stream generator and saves predictions to Taxonomy.
    """

    row_generator = ingestion_svc.stream_dataset_rows(dataset_id)  # generator

    pipeline_result = ml_svc.run_enrichment_pipeline(
        dataset_id=dataset_id,
        row_generator=row_generator,
        mode=mode,
        dry_run=dry_run,
        use_model_predictions=use_model_predictions,
        stop_after_cv=stop_after_cv,
    )
    result = pipeline_result.payload
    status_code = pipeline_result.status_code

    if status_code != 200:
        # Instead of failing with a Pydantic ValidationError on return, explicitly raise
        raise RuntimeError(result.get("error", "Unknown ML enrichment error"))

    # ── Only orchestrator crosses domain boundaries ──
    cv_built = result.pop("cv_built", {})
    predictions = result.pop("predictions", {})
    if not dry_run and (cv_built or predictions):

        save_res = taxonomy_svc.save_enrichment_results(cv_built, predictions)
        result["taxonomy_writes"] = {
            "written": len(save_res.get("built", {})),
            "failed_count": 0,
            "failed": {},
        }

    # Start foundation worker in background if there's pending foundation work
    try:
        ml_svc.start_foundation_worker()
    except Exception as e:
        print(f"[Orchestrator] Failed to start foundation worker: {e}")

    return EnrichmentResponse(**result)



def activate_dataset(dataset_id: str) -> bool:
    """Activate dataset in Ingestion DB."""

    return ingestion_svc.set_active_dataset(dataset_id)


def pre_analyse_csv(csv_bytes: bytes, filename: str) -> dict:
    """Pre-import analysis.

    Orchestrator fetches foundation csids from ml_inference and injects them
    into ingestion analyser. Ingestion never opens foundation.db directly.

    Pattern mirrors start_foundation_compute():
      ml    → get foundation sets
      ingestion → analyse with injected context
    """
    # Step 1: ML se foundation csid sets lo (read-only)
    foundation_done, foundation_all = ml_svc.get_foundation_csid_sets()

    # Step 2: ingestion analyser ko inject karke call karo
    # build_analysis_from_bytes returns (analysis, raw_headers) tuple
    analysis, raw_headers = ingestion_svc.build_analysis_from_bytes(
        csv_bytes=csv_bytes,
        foundation_done=foundation_done,
        foundation_all=foundation_all,
    )

    # Step 3: metadata jo ingestion se bahar belong karta hai, yahan attach karo
    analysis["filename"] = filename
    analysis["headers_raw"] = raw_headers
    return analysis



def start_foundation_compute(sync_first: bool = True) -> StartFoundationComputeResponse:
    """
    Syncs images from active Ingestion dataset into Foundation DB, starts worker.
    """

    pairs = []
    if sync_first:
        dataset_id = ingestion_svc.get_active_dataset_id()
        if dataset_id:
            samples = ingestion_svc.get_all_dataset_images(dataset_id)
            seen_urls = set()
            for s in samples:
                url = (
                    s.get("image_url")
                    if isinstance(s, dict)
                    else getattr(s, "image_url", None)
                )
                csid = (
                    s.get("company_stone_id")
                    if isinstance(s, dict)
                    else getattr(s, "company_stone_id", None)
                )
                if url and url not in seen_urls and csid:
                    pairs.append((csid, url))
                    seen_urls.add(url)

    return StartFoundationComputeResponse(**ml_svc.start_foundation_compute(pairs, sync_first=sync_first))


def stop_foundation_compute() -> StopFoundationComputeResponse:

    return StopFoundationComputeResponse(**ml_svc.stop_foundation_compute())


def resume_foundation_compute() -> ResumeFoundationComputeResponse:

    return ResumeFoundationComputeResponse(**ml_svc.resume_foundation_compute())


def get_foundation_status() -> FoundationStatusResponse:

    return FoundationStatusResponse(**ml_svc.get_foundation_status())


def rebuild_foundation_aggregates() -> dict:

    return ml_svc.rebuild_foundation_aggregates()


def retry_failed_foundation_images() -> RetryFailedFoundationResponse:

    return RetryFailedFoundationResponse(**ml_svc.retry_failed_foundation_images())


def get_foundation_failure_stats() -> FoundationFailureStatsResponse:

    return FoundationFailureStatsResponse(**ml_svc.get_foundation_failure_stats())


def get_foundation_stock_aggregate(company_stone_id: str) -> Optional[FoundationStockAggregateResponse]:

    res = ml_svc.get_foundation_stock_aggregate(company_stone_id)
    if not res:
        return None
    return FoundationStockAggregateResponse(**res)


def resolve_similar_stones(
    anchor_id: str,
    mode: str = "buyer",
    search_version: str = "v2",
) -> dict:
    """Orchestrator crosses search → similar boundary.

    search.services gives us the hydrated stone list.
    similar.services does the resolution.
    Neither module knows about the other.
    """

    stones = search_svc.get_hydrated_stone_list()
    return similar_svc.resolve(
        anchor_id, stones, mode=mode, search_version=search_version
    )


def get_hydrated_stone_list() -> list:
    """Return hydrated stones for similar router logic."""
    return search_svc.get_hydrated_stone_list()


def get_visual_similar_stones(company_stone_id: str, limit: int = 50) -> dict:
    """GET /api/stones/{id}/similar flow.

    search.services gives stones + embedding logic.
    """
    stones = search_svc.get_hydrated_stone_list()
    res = similar_svc.get_visual_similar(company_stone_id, stones, limit=limit)
    if res is None:
        return {}
    return res


def record_sibling_vote(
    anchor_id: str,
    candidate_id: str,
    vote: int,
    admin_user_id: Optional[str] = None,
) -> RecordSiblingVoteResponse:
    """Orchestrator crosses similar → taxonomy boundary to record sibling votes."""

    success = taxonomy_svc.set_sibling_vote(
        reference_company_stone_id=anchor_id,
        sibling_company_stone_id=candidate_id,
        vote=vote,
        admin_user_id=admin_user_id,
    )
    message = "Vote recorded successfully" if success else "Failed to record vote"
    return RecordSiblingVoteResponse(ok=success, message=message)


def trigger_ml_enrichment_background(
    dataset_id: str,
    mode: str,
    dry_run: bool,
    use_model_predictions: bool,
    stop_after_cv: bool,
) -> None:
    """Initializes progress state and runs ML enrichment in a background thread."""
    ml_svc.initialize_enrichment_progress(dataset_id)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        ml_enrichment,
        dataset_id,
        mode,
        dry_run,
        use_model_predictions,
        stop_after_cv,
    )
