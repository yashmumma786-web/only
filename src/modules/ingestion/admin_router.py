"""
Dataset Import Router (Admin UI)
"""
import io
import json
import csv
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from src.modules.ingestion import services

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/admin/clean-import", response_class=HTMLResponse)
async def clean_import_page(request: Request):
    db_info = services.get_db_info()
    existing_validation = None
    if db_info.get("exists"):
        existing_validation = services.get_validation_report()

    return templates.TemplateResponse(request, "admin_clean_import.html", {
        "request": request,
        "clean_db_path": db_info.get("db_path"),
        "db_info": db_info,
        "existing_validation": existing_validation,
    })


@router.post("/admin/clean-import/preview")
async def clean_import_preview(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8-sig")

        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []

        rows = []
        for i, row in enumerate(reader):
            if i >= 20:
                break
            rows.append(row)

        validation = {"missing_columns": []}
        required = ["batch_id", "company_stone_id", "vendor", "stone_name_raw", "image_asset_id", "image_url"]

        for req in required:
            if req not in headers:
                validation["missing_columns"].append(req)

        return JSONResponse({
            "success": True,
            "headers": headers,
            "preview_rows": rows,
            "validation": validation,
            "total_rows": len(text.splitlines()) - 1,
            "filename": file.filename
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@router.get("/admin/clean-import/datasets")
async def list_clean_datasets():
    datasets = services.get_all_clean_datasets()
    return JSONResponse(datasets)


@router.delete("/admin/clean-import/datasets/delete/{dataset_id}")
async def delete_clean_dataset(dataset_id: str):
    success = services.delete_dataset(dataset_id)
    if success:
        return JSONResponse({"success": True, "deleted": dataset_id})
    else:
        raise HTTPException(status_code=400, detail="Cannot delete active dataset")


@router.get("/admin/clean-import/datasets/{dataset_id}/log")
async def download_clean_dataset_log(dataset_id: str):
    stats = services.get_dataset_stats(dataset_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Dataset not found")

    output = io.StringIO()
    json.dump(stats, output, indent=2)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=import_log_{dataset_id}.json"
        },
    )
