import json
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from src.modules.ml_inference.dataset_run_history import (
    get_runs_for_dataset,
    find_run_report_path,
    initiate_rollback,
    RollbackError,
)

router = APIRouter()

@router.get("/admin/clean-import/datasets/{dataset_id}/runs")
async def list_clean_dataset_runs(dataset_id: str):
    """Per-dataset color & pattern run history."""
    return JSONResponse(get_runs_for_dataset(dataset_id))

@router.post("/admin/clean-import/datasets/{dataset_id}/runs/{run_id}/rollback")
async def rollback_clean_dataset_run(
    dataset_id: str,
    run_id: str,
    track: str = Form(...),
    actor: str = Form("admin"),
    reason: str = Form(""),
):
    """Track-scoped rollback initiation."""
    try:
        event = initiate_rollback(
            dataset_id=dataset_id,
            track=track,
            run_id=run_id,
            actor=actor,
            reason=reason,
        )
    except RollbackError as e:
        return JSONResponse(
            {"success": False, "error": e.message},
            status_code=e.code,
        )
    return JSONResponse({"success": True, "event": event})


@router.get("/admin/clean-import/datasets/{dataset_id}/runs/{run_id}/report")
async def get_clean_dataset_run_report(dataset_id: str, run_id: str):
    """Read-only passthrough that serves the report file for a single run."""
    path = find_run_report_path(dataset_id, run_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Run report not found")
    if path.suffix == ".json":
        try:
            return JSONResponse(json.loads(path.read_text()))
        except Exception:
            return PlainTextResponse(path.read_text())
    return PlainTextResponse(path.read_text(), media_type="text/markdown")
