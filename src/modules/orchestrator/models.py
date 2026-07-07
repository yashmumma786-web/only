from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

# --- Shared Models ---

class GenericSuccessResponse(BaseModel):
    success: bool
    active_dataset_id: Optional[str] = None

# --- Ingestion & Import Models ---

class FoundationEnqueueStatus(BaseModel):
    dataset_id: str
    pairs_enqueued: Optional[int] = None
    rows_touched: Optional[int] = None
    status: Optional[str] = None
    error: Optional[str] = None

class IngestionImportResult(BaseModel):
    success: bool
    dataset_id: Optional[str] = None
    error: Optional[str] = None
    rows_read: Optional[int] = None
    rows_imported: Optional[int] = None
    duplicates_skipped: Optional[int] = None
    errors: List[str] = []
    validation: Dict[str, Any] = {}
    validation_report: Dict[str, Any] = {}
    validation_status: str = "draft"
    activation_blocked: bool = True
    blocked_reasons: List[str] = []
    continuity_report: Dict[str, Any] = {}

class MLImportResult(BaseModel):
    foundation_enqueue: Optional[FoundationEnqueueStatus] = None
    enrichment_plan: Optional[Dict[str, Any]] = None

class ImportResponse(IngestionImportResult, MLImportResult):
    """
    Combines Ingestion and ML Inference results into a single flat API response.
    """
    pass

# --- ML Enrichment Models ---

class EnrichDatasetRequest(BaseModel):
    mode: str = "incremental"
    dry_run: bool = False
    use_model_predictions: bool = False
    stop_after_cv: bool = False

class TaxonomyWrites(BaseModel):
    written: int
    failed_count: int
    failed: Dict[str, Any]

class ClassificationCounts(BaseModel):
    new: int
    changed_image: int
    changed_metadata: int
    unchanged: int
    invalid: int

class ClassificationSummary(BaseModel):
    classified_at: str
    baseline_dataset_id: Optional[str] = None
    total_input_rows: int
    total_stones: int
    counts: ClassificationCounts
    ids: Dict[str, List[str]] = {}
    invalid_reasons: Dict[str, str] = {}
    stages_required: Dict[str, int] = {}

class EnrichmentStagesRun(BaseModel):
    classification: ClassificationCounts
    embedding_cache_invalidated: int
    cv_and_aggregation_built: int
    cv_and_aggregation_skipped: List[str] = []
    metadata_aggregation_built: int
    metadata_aggregation_skipped: List[str] = []
    embeddings_built: int
    embeddings_skipped: int
    embeddings_failed: int
    cache_invalidated: bool

class EnrichmentResponse(BaseModel):
    status: str
    mode: str
    dataset_id: str
    summary: ClassificationSummary
    stages_run: Optional[EnrichmentStagesRun] = None
    failed_stone_ids: List[str] = []
    failed_details: Dict[str, Any] = {}
    stopped_after_cv: Optional[bool] = None
    taxonomy_writes: Optional[TaxonomyWrites] = None
    error: Optional[str] = None

class EnrichmentQueuedResponse(BaseModel):
    status: str
    dataset_id: str
    message: str

# --- Foundation Compute Models ---

class StartFoundationComputeRequest(BaseModel):
    sync_first: bool = True

class StartFoundationComputeResponse(BaseModel):
    job_id: Union[str, int]
    message: str
    total_images: int
    images_synced: int

class StopFoundationComputeResponse(BaseModel):
    message: str
    job_canceled: Optional[Union[str, int]] = None

class ResumeFoundationComputeResponse(BaseModel):
    started: bool
    pending: int

class FoundationStatusResponse(BaseModel):
    job_id: Optional[Union[str, int]] = None
    status: Optional[str] = None
    total: int
    done: int
    failed: int
    pending: int
    percent: float
    worker_running: bool
    recent_errors: List[Dict[str, Any]] = []

class RetryFailedFoundationResponse(BaseModel):
    reset_count: int
    previous_failure_stats: Dict[str, Any]
    worker_started: bool

class FoundationFailureStatsResponse(BaseModel):
    total_failed: int
    by_reason: Dict[str, Any]

class FoundationStockAggregateResponse(BaseModel):
    company_stone_id: str
    rep_image_url: str
    consistency_score: float
    outlier_count: int
    image_count: int
    updated_at: Optional[str] = None

# --- Similar & Taxonomy Models ---

