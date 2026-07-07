import csv
import io
from typing import Dict, List, Optional, Any
from src.modules.ingestion import repository as _repo
from src.modules.ingestion.repository import DatasetImage
from src.modules.ingestion.analyser_router import (
    _parse_csv_with_vendor_quirk,
    _build_analysis,
)

def get_active_dataset_id() -> str:
    """Read the active dataset ID from datasets.db."""
    val = _repo.get_active_dataset_id()
    return val if val else ""

def get_sample_images_for_stock(company_stone_id: str, dataset_id: str, limit: int = 3) -> List[DatasetImage]:
    """Get sample image objects for a specific stone."""
    return _repo.get_sample_images_for_stock(company_stone_id, dataset_id, limit)



def get_batch_hydrate(company_stone_ids: List[str]) -> List[str]:
    """Get existing company stone IDs from batch."""
    return _repo.get_batch_hydrate(company_stone_ids)


def get_all_dataset_images(dataset_id: str) -> List[Dict[str, Any]]:
    """Get all dataset_images rows for a specific dataset."""
    return _repo.get_all_dataset_images(dataset_id)


def get_all_stone_data(dataset_id: str) -> Dict[str, Any]:
    """Get all stone data for a dataset."""
    return _repo.get_all_stone_data(dataset_id)

# ---------------------------------------------------------
# Admin UI Service Layer
# ---------------------------------------------------------

def set_active_dataset(dataset_id: str) -> bool:
    return _repo.set_active_dataset(dataset_id)

def delete_dataset(dataset_id: str) -> bool:
    return _repo.delete_dataset(dataset_id)

def get_dataset_stats(dataset_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return _repo.get_dataset_stats(dataset_id)

def get_validation_report() -> Dict[str, Any]:
    return _repo.get_validation_report(dataset_id=None)

def get_active_dataset_csids():
    return _repo.get_active_dataset_csids()


def build_analysis_from_bytes(
    csv_bytes: bytes,
    foundation_done: set,
    foundation_all: set,
) -> tuple:
    """Parse CSV and build pre-import analysis.

    foundation_done and foundation_all are injected by orchestrator —
    ingestion never opens foundation.db directly.

    Returns:
        (analysis, raw_headers)
        analysis     — full analysis dict (column_check, counts, overlap, etc.)
        raw_headers  — original CSV header list (caller attaches as headers_raw)
    """

    text = csv_bytes.decode("utf-8-sig")
    raw_headers, rows, _, _ = _parse_csv_with_vendor_quirk(text)
    analysis = _build_analysis(raw_headers, rows, foundation_done, foundation_all)
    return analysis, raw_headers


def get_all_clean_datasets() -> List[Dict]:
    return _repo.get_all_clean_datasets()

def get_db_info() -> Dict[str, Any]:
    return _repo.get_db_info()


def run_clean_import(rows: List[Dict], fieldnames: List[str], filename: str) -> Dict:
    try:
        result = _repo.import_csv_clean_v2(
            rows=rows,
            source_filename=filename,
            fieldnames=fieldnames,
        )
        response_dict = result.to_dict() if hasattr(result, "to_dict") else result
        return response_dict
    except Exception as e:
        return {"success": False, "error": str(e)}


def stream_dataset_rows(dataset_id: str):
    """Proxy to repository generator. Ingestion's public streaming interface."""
    return _repo.stream_dataset_rows(dataset_id)

def get_manifest_rows_for_enrichment(dataset_id: str) -> List[Dict]:
    """Public facade for ml_inference to fetch image rows for enrichment.
    Callers: ml_inference.repository only.
    """
    return _repo.get_manifest_rows_for_enrichment(dataset_id)

def import_csv_from_bytes(csv_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Parses raw CSV bytes and triggers the ingestion import pipeline."""

    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)
    return run_clean_import(rows, fieldnames, filename)




