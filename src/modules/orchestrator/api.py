from typing import Union
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from src.modules.ml_inference.admin_router import router as admin_router
from src.modules.orchestrator.public_router import router as public_router
from src.modules.orchestrator import services as orch_svc
from src.modules.ml_inference import services as ml_svc
import traceback
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.modules.orchestrator.models import (
    FoundationStatusResponse,
    FoundationFailureStatsResponse,
    FoundationStockAggregateResponse,
    StartFoundationComputeResponse,
    StopFoundationComputeResponse,
    ResumeFoundationComputeResponse,
    RetryFailedFoundationResponse,
    StartFoundationComputeRequest,
    ImportResponse,
    EnrichDatasetRequest,
    EnrichmentResponse,
    EnrichmentQueuedResponse,
    GenericSuccessResponse,
)

router = APIRouter(tags=["Orchestrator"])
router.include_router(admin_router)
router.include_router(public_router)


@router.post("/admin/clean-import/pre-analyse")
async def clean_import_pre_analyse(file: UploadFile = File(...)):
    """Pre-import CSV analysis.

    Orchestrator fetches foundation csids from ml_inference and injects them
    into ingestion analyser. Ingestion never opens foundation.db directly.
    """
    try:
        content = await file.read()
        analysis = orch_svc.pre_analyse_csv(content, file.filename)
        return JSONResponse({"success": True, "analysis": analysis})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@router.post("/api/orchestrator/import")
async def import_csv(file: UploadFile = File(...)) -> ImportResponse:
    """
    Import CSV → save to Ingestion DB → trigger ML enrichment.
    Maps to senior's: /import_csv → orchestrator.import_csv(file)
    """
    try:
        content = await file.read()

        return orch_svc.import_csv(content, file.filename)
    except Exception as e:
        traceback.print_exc()
        return ImportResponse(success=False, error=str(e))


@router.post("/api/orchestrator/datasets/{dataset_id}/activate")
async def activate_dataset(dataset_id: str) -> GenericSuccessResponse:
    """Activate a dataset in Ingestion."""
    try:
        success = orch_svc.activate_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return GenericSuccessResponse(success=True, active_dataset_id=dataset_id)


@router.post("/api/orchestrator/datasets/{dataset_id}/enrich")
async def enrich_dataset(dataset_id: str, payload: EnrichDatasetRequest) -> Union[EnrichmentResponse, EnrichmentQueuedResponse]:
    """
    Trigger ML enrichment for a dataset. Runs in background thread.
    Maps to senior's flow: Ingestion.get_data() → ML.enrich()
    """
    if payload.dry_run:
        try:
            return orch_svc.ml_enrichment(
                dataset_id,
                payload.mode,
                payload.dry_run,
                payload.use_model_predictions,
                payload.stop_after_cv,
            )
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    else:
        orch_svc.trigger_ml_enrichment_background(
            dataset_id,
            payload.mode,
            payload.dry_run,
            payload.use_model_predictions,
            payload.stop_after_cv,
        )
        return EnrichmentQueuedResponse(
            status="queued",
            dataset_id=dataset_id,
            message="Enrichment started in background.",
        )


@router.get("/api/orchestrator/datasets/{dataset_id}/progress")
async def enrichment_progress(dataset_id: str):
    """
    Get the current ML enrichment progress.
    """
    persisted = ml_svc.get_enrichment_progress(dataset_id)
    if persisted is not None:
        return JSONResponse(persisted)
    return JSONResponse({"status": "idle"})



@router.post("/api/orchestrator/compute/start")
async def start_foundation_compute(request: StartFoundationComputeRequest) -> StartFoundationComputeResponse:
    """Start Foundation Compute embedding job."""
    try:
        return orch_svc.start_foundation_compute(sync_first=request.sync_first)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/orchestrator/compute/stop")
async def stop_foundation_compute() -> StopFoundationComputeResponse:
    """Stop the background embedding worker."""
    return orch_svc.stop_foundation_compute()


@router.post("/api/orchestrator/compute/resume")
async def resume_foundation_compute() -> ResumeFoundationComputeResponse:
    """Resume pending embedding work."""
    return orch_svc.resume_foundation_compute()


@router.get("/api/orchestrator/compute/status")
async def get_foundation_status() -> FoundationStatusResponse:
    """Get current job status and progress."""
    return orch_svc.get_foundation_status()


@router.post("/api/orchestrator/compute/rebuild_aggregates")
async def rebuild_aggregates():
    """Recompute stock-level aggregates for all stocks with DONE images."""
    return JSONResponse(orch_svc.rebuild_foundation_aggregates())


@router.post("/api/orchestrator/compute/retry_failed")
async def retry_failed_images() -> RetryFailedFoundationResponse:
    """Reset all FAILED images back to PENDING and start the worker."""
    return orch_svc.retry_failed_foundation_images()


@router.get("/api/orchestrator/compute/failure_stats")
async def get_failure_stats() -> FoundationFailureStatsResponse:
    """Get breakdown of failures by reason."""
    return orch_svc.get_foundation_failure_stats()


@router.get("/api/orchestrator/compute/stock/{company_stone_id}")
async def get_stock_aggregate(company_stone_id: str) -> FoundationStockAggregateResponse:
    """Get aggregate data for a specific stock."""
    result = orch_svc.get_foundation_stock_aggregate(company_stone_id)
    if not result:
        raise HTTPException(
            status_code=404, detail="Stock not found or no aggregates computed"
        )
    return result


templates = Jinja2Templates(directory="templates")


@router.get("/admin/foundation")
async def admin_foundation_page(request: Request) -> HTMLResponse:
    """Render the admin foundation control panel."""
    return templates.TemplateResponse(
        request, "admin_foundation.html", {"request": request}
    )
