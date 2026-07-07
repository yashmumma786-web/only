import hashlib
import traceback
import os
import time
import requests as _requests
from PIL import Image as _PILImage
from io import BytesIO
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from src.modules.ml_inference import services as ml_service
from src.modules.ml_inference import repository as _repo
from src.modules.ml_inference.incremental import (
    classify_csv_rows,
    CHANGE_CLASS_CHANGED_IMAGE,
)
from src.modules.ml_inference.incremental import (
    get_stones_requiring_stage,
    invalidate_embedding_cache_for_stones,
)
from src.modules.ml_inference.foundation_storage import foundation_storage as _fs
from src.modules.ml_inference import embeddings as v3_embeddings
from src.utils.cache_manager import invalidate_stone_cache
from src.modules.ml_inference.v3_classifiers import predict_stock
from src.utils.image_fetcher import normalize_url, SESSION_HEADERS
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed as _as_completed,
)

_ENRICHMENT_PROGRESS: dict = {}
_PROGRESS_PERSIST_EVERY = 3


class EnrichmentProgressTracker:
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def initialize(self, total_stones: int):
        _repo.init_progress_table()
        _ENRICHMENT_PROGRESS[self.dataset_id] = {
            "status": "running",
            "stage": "classifying",
            "stage_done": 0,
            "stage_total": total_stones,
            "overall_done": 0,
            "overall_total": total_stones,
            "current_stone": None,
            "stages_completed": [],
            "started_at": time.time(),
            "finished_at": None,
            "error": None,
        }
        _repo.persist_progress(self.dataset_id, _ENRICHMENT_PROGRESS[self.dataset_id])

    def update_stage(
        self,
        stage_name: str,
        stage_done: int,
        stage_total: int,
        overall_done: int = None,
        overall_total: int = None,
        current_stone: str = None,
        classification: dict = None,
    ):
        p = _ENRICHMENT_PROGRESS.get(self.dataset_id)
        if p:
            p.update(
                {
                    "stage": stage_name,
                    "stage_done": stage_done,
                    "stage_total": stage_total,
                }
            )
            if overall_done is not None:
                p["overall_done"] = overall_done
            if overall_total is not None:
                p["overall_total"] = overall_total
            if current_stone is not None:
                p["current_stone"] = current_stone
            if classification is not None:
                p["classification"] = classification
            _repo.persist_progress(self.dataset_id, p)

    def add_completed_stage(self, stage_name: str):

        p = _ENRICHMENT_PROGRESS.get(self.dataset_id)
        if p:
            if stage_name not in p["stages_completed"]:
                p["stages_completed"].append(stage_name)
            _repo.persist_progress(self.dataset_id, p)

    def complete(self, overall_total: int, stages_run: dict, failed_count: int):
        p = _ENRICHMENT_PROGRESS.get(self.dataset_id)
        if p:
            p.update(
                {
                    "status": "done",
                    "stage": "done",
                    "overall_done": overall_total,
                    "finished_at": time.time(),
                    "final_stages_run": {
                        k: v
                        for k, v in stages_run.items()
                        if k
                        not in (
                            "cv_and_aggregation_failed",
                            "metadata_aggregation_failed",
                        )
                    },
                    "failed_count": failed_count,
                }
            )
            _repo.persist_progress(self.dataset_id, p)

    def complete_dry_run(self):
        _ENRICHMENT_PROGRESS.pop(self.dataset_id, None)
        _repo.persist_progress(self.dataset_id, {"status": "done", "stage": "dry_run"})

    def fail(self, exc: Exception):
        p = _ENRICHMENT_PROGRESS.get(self.dataset_id)
        if p:
            p.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "finished_at": time.time(),
                }
            )
            _repo.persist_progress(self.dataset_id, p)


def _run_classification_stage(rows: list, mode: str):

    summary = classify_csv_rows(rows)

    if mode == "full":
        # Force everything to be treated as changed images so it runs through the pipeline
        unchanged_ids = [
            k for k, v in summary.classifications.items() if v.is_unchanged
        ]
        summary.changed_image_ids.extend(unchanged_ids)
        summary.changed_image_ids.extend(summary.changed_metadata_ids)

        for k in unchanged_ids + summary.changed_metadata_ids:
            summary.classifications[k].change_class = CHANGE_CLASS_CHANGED_IMAGE

        summary.changed_metadata_ids.clear()
        summary.changed_image_count = len(summary.changed_image_ids)
        summary.changed_metadata_count = 0
        summary.unchanged_count = 0

    return summary


def _record_stage_failures(dataset_id: str, stage: str, result_dict: dict):
    try:
        _repo.ensure_report_schema()
        run_id_env = os.environ.get("ENRICHMENT_RUN_ID")

        failed_ids = []
        for sid, info in (result_dict.get("failed") or {}).items():
            if isinstance(info, dict):
                ec = info.get("error_class") or info.get("class") or "Exception"
                em = (
                    info.get("error")
                    or info.get("error_message")
                    or info.get("message")
                    or ""
                )
                imgn = int(info.get("image_count_attempted") or 0)
            else:
                em = "" if info is None else str(info)
                ec = "PrefetchError" if em.startswith("prefetch:") else "CVError"
                imgn = 0

            _repo.record_failure(
                run_id=run_id_env,
                dataset_id=dataset_id,
                stage=stage,
                company_stone_id=sid,
                error_class=ec,
                error_message=str(em),
                image_count_attempted=imgn,
            )
            failed_ids.append(sid)

        if failed_ids:
            _repo.mark_stones_resume_status(dataset_id, stage, failed_ids, "failed")

        skipped_ids = result_dict.get("skipped") or []
        if skipped_ids:
            _repo.mark_stones_resume_status(dataset_id, stage, skipped_ids, "skipped")
    except Exception as exc:
        print(f"[Enrichment] failure instrumentation for {stage} skipped: {exc}")


def _run_cv_stage(
    dataset_id: str,
    summary,
    tracker: EnrichmentProgressTracker,
    manifest_df,
    use_model_predictions: bool,
):
    cv_ids = get_stones_requiring_stage(summary, "cv")
    if not cv_ids:
        return {
            "built": {},
            "skipped": [],
            "failed": {},
            "predictions": {},
            "cv_ids_already_done": 0,
            "invalidated_count": 0,
        }

    _repo.init_resume_table()
    cv_done_set = _repo.load_resume_set(dataset_id, "cv")

    cv_ids_remaining = [sid for sid in cv_ids if sid not in cv_done_set]
    cv_ids_already_done = len(cv_ids) - len(cv_ids_remaining)
    print(
        f"[V3-RESUME] CV: {cv_ids_already_done} already completed this run, {len(cv_ids_remaining)} remaining out of {len(cv_ids)}"
    )

    built_all = {}
    predictions_all = {}
    skipped_all = []
    failed_all = {}
    deleted = {}

    if cv_ids_remaining:
        deleted = invalidate_embedding_cache_for_stones(cv_ids_remaining)
        tracker.update_stage(
            stage_name="prefetching",
            stage_done=cv_ids_already_done,
            stage_total=len(cv_ids),
        )

        def _prefetch_progress(done, total, stone_id):
            tracker.update_stage(
                stage_name="prefetching",
                stage_done=cv_ids_already_done + done,
                stage_total=cv_ids_already_done + total,
                current_stone=stone_id,
            )

        def _cv_progress(done, total, stone_id):
            if done % _PROGRESS_PERSIST_EVERY == 0 or done == total:
                tracker.update_stage(
                    stage_name="cv_fingerprints",
                    stage_done=cv_ids_already_done + done,
                    stage_total=cv_ids_already_done + total,
                    overall_done=cv_ids_already_done + done,
                    current_stone=stone_id,
                )
            _repo.mark_stage_done(dataset_id, "cv", stone_id)

        cv_result = rebuild_fingerprints_for_stones(
            stone_ids=cv_ids_remaining,
            manifest_df=manifest_df,
            max_images_per_stock=3,
            use_model_predictions=use_model_predictions,
            progress_callback=_cv_progress,
            prefetch_callback=_prefetch_progress,
        )
        built_all.update(cv_result.get("built", {}))
        predictions_all.update(cv_result.get("predictions", {}))
        skipped_all.extend(cv_result.get("skipped", []))
        failed_all.update(cv_result.get("failed", {}))

        _record_stage_failures(dataset_id, "cv", cv_result)

    tracker.add_completed_stage("cv_fingerprints")

    return {
        "built": built_all,
        "skipped": skipped_all,
        "failed": failed_all,
        "predictions": predictions_all,
        "cv_ids_already_done": cv_ids_already_done,
        "invalidated_count": (
            sum(1 for v in deleted.values() if v) if cv_ids_remaining else 0
        ),
    }


def _run_embeddings_stage(
    dataset_id: str,
    cv_ids: List[str],
    tracker: EnrichmentProgressTracker,
    manifest_df,
):
    emb_done_set = _repo.load_resume_set(dataset_id, "embeddings")
    if not emb_done_set:
        emb_cached = {
            sid for sid in cv_ids if Path(f"cache/embeddings_v3/{sid}.npy").exists()
        }
        if emb_cached:
            print(
                f"[V3-RESUME] Embeddings ledger empty — seeded from .npy cache: {len(emb_cached)} stones"
            )
            emb_done_set = emb_cached
            for sid in emb_cached:
                _repo.mark_stage_done(dataset_id, "embeddings", sid)

    emb_ids_remaining = [sid for sid in cv_ids if sid not in emb_done_set]
    emb_ids_already_done = len(cv_ids) - len(emb_ids_remaining)
    print(
        f"[V3-RESUME] Embeddings: {emb_ids_already_done} already completed this run, {len(emb_ids_remaining)} remaining out of {len(cv_ids)}"
    )

    if emb_ids_remaining:
        emb_manifest = manifest_df[
            manifest_df["company_stone_id"].isin(emb_ids_remaining)
        ]
    else:
        emb_manifest = manifest_df[manifest_df["company_stone_id"].isin([])]

    tracker.update_stage(
        stage_name="embeddings",
        stage_done=emb_ids_already_done,
        stage_total=len(cv_ids),
        overall_done=len(cv_ids) + emb_ids_already_done,
    )

    def _emb_progress(done, total, stone_id):
        if done % _PROGRESS_PERSIST_EVERY == 0 or done == total:
            tracker.update_stage(
                stage_name="embeddings",
                stage_done=emb_ids_already_done + done,
                stage_total=len(cv_ids),
                overall_done=len(cv_ids) + emb_ids_already_done + done,
                current_stone=stone_id,
            )
        _repo.mark_stage_done(dataset_id, "embeddings", stone_id)

    emb_result = v3_embeddings.build_all_embeddings(
        emb_manifest,
        max_images_per_stock=3,
        force_rebuild=False,
        progress_callback=_emb_progress,
    )

    tracker.add_completed_stage("embeddings")

    return {
        "built": emb_result.get("built", 0) + emb_ids_already_done,
        "skipped": emb_result.get("skipped", 0),
        "failed": emb_result.get("failed", 0),
    }


def _run_metadata_stage(
    dataset_id: str,
    meta_only_ids: List[str],
    cv_ids: List[str],
    tracker: EnrichmentProgressTracker,
    manifest_df,
    use_model_predictions: bool,
):
    meta_done_set = _repo.load_resume_set(dataset_id, "metadata")
    meta_ids_remaining = [sid for sid in meta_only_ids if sid not in meta_done_set]
    meta_ids_already_done = len(meta_only_ids) - len(meta_ids_remaining)
    if meta_ids_already_done:
        print(
            f"[V3-RESUME] Metadata: {meta_ids_already_done} already completed, {len(meta_ids_remaining)} remaining out of {len(meta_only_ids)}"
        )

    tracker.update_stage(
        stage_name="metadata_aggregation",
        stage_done=meta_ids_already_done,
        stage_total=len(meta_only_ids),
    )
    meta_overall_offset = len(cv_ids) * 2

    built_all = {}
    predictions_all = {}
    skipped_all = []
    failed_all = {}

    if meta_ids_remaining:

        def _meta_progress(done, total, stone_id):
            if done % _PROGRESS_PERSIST_EVERY == 0 or done == total:
                tracker.update_stage(
                    stage_name="metadata_aggregation",
                    stage_done=meta_ids_already_done + done,
                    stage_total=len(meta_only_ids),
                    overall_done=meta_overall_offset + meta_ids_already_done + done,
                    current_stone=stone_id,
                )
            _repo.mark_stage_done(dataset_id, "metadata", stone_id)

        meta_result = rebuild_fingerprints_for_stones(
            stone_ids=meta_ids_remaining,
            manifest_df=manifest_df,
            max_images_per_stock=3,
            use_model_predictions=use_model_predictions,
            progress_callback=_meta_progress,
        )
        built_all.update(meta_result.get("built", {}))
        predictions_all.update(meta_result.get("predictions", {}))
        skipped_all.extend(meta_result.get("skipped", []))
        failed_all.update(meta_result.get("failed", {}))

        _record_stage_failures(dataset_id, "metadata", meta_result)
    else:
        print(f"[V3-RESUME] Metadata stage fully complete, skipping")

    tracker.add_completed_stage("metadata_aggregation")

    return {
        "built": built_all,
        "skipped": skipped_all,
        "failed": failed_all,
        "predictions": predictions_all,
        "meta_ids_already_done": meta_ids_already_done,
    }


@dataclass
class EnrichmentPipelineResult:
    payload: Dict[str, Any]
    status_code: int


def run_enrichment_pipeline(
    dataset_id: str,
    rows: list,
    mode: str,
    dry_run: bool,
    use_model_predictions: bool,
    stop_after_cv: bool,
) -> EnrichmentPipelineResult:
    """Refactored pipeline entry point that decomposes the ML enrichment process."""
    tracker = EnrichmentProgressTracker(dataset_id)
    try:
        unique_stones = (
            len({r.get("company_stone_id") or r.get("object_key", "") for r in rows})
            if rows
            else 0
        )
        tracker.initialize(unique_stones)

        # 1. Classify CSV rows
        summary = _run_classification_stage(rows, mode)
        if dry_run:
            tracker.complete_dry_run()
            return EnrichmentPipelineResult(
                payload={
                    "status": "dry_run",
                    "mode": mode,
                    "dataset_id": dataset_id,
                    "summary": summary.to_dict(include_full_classifications=False),
                },
                status_code=200,
            )

        cv_ids = get_stones_requiring_stage(summary, "cv")
        meta_only_ids = summary.changed_metadata_ids
        agg_ids = get_stones_requiring_stage(summary, "aggregation")

        overall_total = len(cv_ids) + len(cv_ids) + len(meta_only_ids)

        # Initial overall tracking setup
        tracker.update_stage(
            stage_name="checking_resume",
            stage_done=0,
            stage_total=len(cv_ids),
            overall_done=0,
            overall_total=overall_total,
            classification={
                "new": summary.new_count,
                "changed_image": summary.changed_image_count,
                "changed_metadata": summary.changed_metadata_count,
                "unchanged": summary.unchanged_count,
                "invalid": summary.invalid_count,
            },
        )
        tracker.add_completed_stage("classifying")

        stages_run = {
            "classification": {
                "new": summary.new_count,
                "changed_image": summary.changed_image_count,
                "changed_metadata": summary.changed_metadata_count,
                "unchanged": summary.unchanged_count,
                "invalid": summary.invalid_count,
            },
            "embedding_cache_invalidated": 0,
            "cv_and_aggregation_built": 0,
            "cv_and_aggregation_failed": {},
            "cv_and_aggregation_skipped": [],
            "metadata_aggregation_built": 0,
            "metadata_aggregation_failed": {},
            "metadata_aggregation_skipped": [],
            "embeddings_built": 0,
            "embeddings_skipped": 0,
            "embeddings_failed": 0,
            "cache_invalidated": False,
        }

        manifest_df = _repo.load_manifest_df(dataset_id)

        all_predictions = {}
        all_cv_built = {}

        # 2. CV Fingerprints
        if cv_ids:
            cv_results = _run_cv_stage(
                dataset_id=dataset_id,
                summary=summary,
                tracker=tracker,
                manifest_df=manifest_df,
                use_model_predictions=use_model_predictions,
            )
            all_cv_built.update(cv_results["built"])
            all_predictions.update(cv_results["predictions"])

            stages_run["embedding_cache_invalidated"] = cv_results["invalidated_count"]
            stages_run["cv_and_aggregation_built"] = (
                len(cv_results["built"]) + cv_results["cv_ids_already_done"]
            )
            stages_run["cv_and_aggregation_failed"] = cv_results["failed"]
            stages_run["cv_and_aggregation_skipped"] = cv_results["skipped"]

            if stop_after_cv:
                return EnrichmentPipelineResult(
                    payload={
                        "status": "success",
                        "mode": mode,
                        "dataset_id": dataset_id,
                        "summary": summary.to_dict(include_full_classifications=False),
                        "stages_run": {
                            k: v
                            for k, v in stages_run.items()
                            if k
                            not in (
                                "cv_and_aggregation_failed",
                                "metadata_aggregation_failed",
                            )
                        },
                        "failed_stone_ids": list(
                            stages_run["cv_and_aggregation_failed"].keys()
                        ),
                        "failed_details": stages_run["cv_and_aggregation_failed"],
                        "stopped_after_cv": True,
                        "predictions": all_predictions,
                        "cv_built": all_cv_built,
                    },
                    status_code=200,
                )

            # 3. Embeddings
            emb_results = _run_embeddings_stage(
                dataset_id=dataset_id,
                cv_ids=cv_ids,
                tracker=tracker,
                manifest_df=manifest_df,
            )
            stages_run["embeddings_built"] = emb_results["built"]
            stages_run["embeddings_skipped"] = emb_results["skipped"]
            stages_run["embeddings_failed"] = emb_results["failed"]

        # 4. Metadata
        if meta_only_ids:
            meta_results = _run_metadata_stage(
                dataset_id=dataset_id,
                meta_only_ids=meta_only_ids,
                cv_ids=cv_ids,
                tracker=tracker,
                manifest_df=manifest_df,
                use_model_predictions=use_model_predictions,
            )
            all_cv_built.update(meta_results["built"])
            all_predictions.update(meta_results["predictions"])
            stages_run["metadata_aggregation_built"] = (
                len(meta_results["built"]) + meta_results["meta_ids_already_done"]
            )
            stages_run["metadata_aggregation_failed"] = meta_results["failed"]
            stages_run["metadata_aggregation_skipped"] = meta_results["skipped"]
        else:
            stages_run["metadata_aggregation_built"] = 0

        # Invalidate stone cache
        if agg_ids:
            try:
                invalidate_stone_cache()
                stages_run["cache_invalidated"] = True
            except Exception as cache_exc:
                print(f"[Enrichment] Cache invalidation failed: {cache_exc}")

        all_failed = {
            **stages_run.get("cv_and_aggregation_failed", {}),
            **stages_run.get("metadata_aggregation_failed", {}),
        }

        # Clear resume ledger
        _repo.clear_resume(dataset_id)
        print(f"[V3-RESUME] Run complete — cleared resume ledger for {dataset_id}")

        tracker.complete(
            overall_total=overall_total,
            stages_run=stages_run,
            failed_count=len(all_failed),
        )

        return EnrichmentPipelineResult(
            payload={
                "status": "success",
                "mode": mode,
                "dataset_id": dataset_id,
                "summary": summary.to_dict(include_full_classifications=False),
                "stages_run": {
                    k: v
                    for k, v in stages_run.items()
                    if k
                    not in ("cv_and_aggregation_failed", "metadata_aggregation_failed")
                },
                "failed_stone_ids": list(all_failed.keys()),
                "failed_details": all_failed,
                "predictions": all_predictions,
                "cv_built": all_cv_built,
            },
            status_code=200,
        )

    except Exception as exc:
        traceback.print_exc()
        tracker.fail(exc)
        return EnrichmentPipelineResult(payload={"error": str(exc)}, status_code=500)


def enrichment_plan_and_enqueue_after_import(rows: list, dataset_id: str) -> dict:
    """
    Internal ML Inference function to calculate the enrichment plan and enqueue images.
    Returns a dictionary intended to be merged into the Orchestrator's response.
    """
    response_dict = {}
    try:
        summary = classify_csv_rows(rows)
        response_dict["enrichment_plan"] = summary.to_dict(
            include_full_classifications=False
        )
        response_dict["enrichment_plan"]["dataset_id"] = dataset_id
    except Exception as clf_err:
        response_dict["enrichment_plan"] = {
            "error": str(clf_err),
            "dataset_id": dataset_id,
        }

    try:

        _seen_urls = set()
        _pairs = []
        for row in rows:
            _url = row.get("url") or row.get("image_url")
            _csid = row.get("company_stone_id")
            if _csid and _url and _url not in _seen_urls:
                _pairs.append((_csid, _url))
                _seen_urls.add(_url)

        _touched = _fs.bulk_upsert_images(_pairs)
        response_dict["foundation_enqueue"] = {
            "dataset_id": dataset_id,
            "pairs_enqueued": len(_pairs),
            "rows_touched": _touched,
        }
    except Exception as _enq_err:
        response_dict["foundation_enqueue"] = {
            "dataset_id": dataset_id,
            "error": str(_enq_err),
            "status": "hard_fail",
        }

    return response_dict


def predict_stock_v3(company_stone_id: str) -> Dict[str, Any]:
    """Predicts stock pattern family class using V3 classifiers locally."""
    return predict_stock(company_stone_id)


# --- CV Pipeline moved from taxonomy/aggregator.py ---


_DEFAULT_MAX_IMAGES = 3
_PREFETCH_WORKERS = 6
_DEFAULT_FAST_MAX_PIXELS = 2_000_000


def _prefetch_images_for_stone(
    stone_id: str,
    image_rows: List[dict],
    cache_dir_str: str,
    max_images: int = _DEFAULT_MAX_IMAGES,
) -> Tuple[str, List[str], List[str]]:
    cache_dir = Path(cache_dir_str)
    image_paths: List[str] = []
    image_ids: List[str] = []

    session = _requests.Session()
    session.headers.update(SESSION_HEADERS)
    adapter = _requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        for row in image_rows:
            if len(image_paths) >= max_images:
                break
            img_id = row.get("image_asset_id", "")
            url = row.get("url", "")
            cache_key = hashlib.md5(img_id.encode()).hexdigest()[:16]
            cached_path = cache_dir / f"{cache_key}.jpg"

            if cached_path.exists():
                image_paths.append(str(cached_path))
                image_ids.append(img_id)
            elif normalize_url and url:
                resp = None
                try:
                    normalized = normalize_url(url)
                    resp = session.get(normalized, timeout=10, allow_redirects=True)
                    if resp.status_code == 200 and resp.headers.get(
                        "Content-Type", ""
                    ).startswith("image/"):

                        pil_img = _PILImage.open(BytesIO(resp.content)).convert("RGB")
                        if pil_img.width * pil_img.height <= 50_000_000:
                            pil_img.save(str(cached_path), "JPEG", quality=85)
                            image_paths.append(str(cached_path))
                            image_ids.append(img_id)
                        pil_img.close()
                except Exception:
                    pass
                finally:
                    if resp is not None:
                        resp.close()
    finally:
        session.close()

    return stone_id, image_paths, image_ids


def _cv_process_stone(
    stone_id: str,
    image_paths: List[str],
    image_ids: List[str],
    max_pixels: int = _DEFAULT_FAST_MAX_PIXELS,
    mode: str = "fast",
) -> Tuple[str, Optional[Dict]]:
    if not image_paths:
        return stone_id, None

    analyses = []
    cache_hits = 0
    cache_misses = 0
    for path in image_paths:
        result, was_hit = ml_service._analyze_image_tracked(
            path, max_pixels=max_pixels, mode=mode
        )
        if result:
            if was_hit:
                cache_hits += 1
            else:
                cache_misses += 1
            analyses.append(result)

    if not analyses:
        return stone_id, None

    fingerprint_dict = {
        "analyses": analyses,
        "image_paths": image_paths,
        "image_ids": image_ids,
        "_analysis_cache_hits": cache_hits,
        "_analysis_cache_misses": cache_misses,
    }
    return stone_id, fingerprint_dict


def rebuild_fingerprints_for_stones(
    stone_ids: List[str],
    manifest_df,
    progress_callback=None,
    prefetch_callback=None,
    max_images_per_stock: int = _DEFAULT_MAX_IMAGES,
    use_model_predictions: bool = False,
    cancel_event=None,
    max_pixels: int = _DEFAULT_FAST_MAX_PIXELS,
) -> Dict[str, Any]:
    """Rebuild V3 fingerprints for a specific subset of stones.

    Two-pass pipeline:
      Pass 1 — high-concurrency I/O prefetch (ThreadPoolExecutor)
      Pass 2 — CPU-bound CV/aggregation (ProcessPoolExecutor)

    All DB writes happen in the parent process after Pass 2.
    """
    target_ids = set(stone_ids)

    stocks_in_manifest = {
        csid: group
        for csid, group in manifest_df.groupby("company_stone_id")
        if csid in target_ids
    }

    missing_from_manifest = target_ids - set(stocks_in_manifest.keys())

    cache_dir = Path(os.environ.get("IMAGE_CACHE_DIR"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir)

    total = len(stocks_in_manifest)
    print(
        f"[V3-INCREMENTAL] Rebuilding {total} stones "
        f"({len(missing_from_manifest)} not found in manifest, skipped) "
        f"[max_images={max_images_per_stock}, max_pixels={max_pixels}]"
    )

    built = {}
    _predictions: Dict[str, dict] = {}
    skipped: List[str] = list(missing_from_manifest)
    failed: Dict[str, str] = {}
    t_start = time.time()

    stone_image_rows = {}
    for sid, grp in stocks_in_manifest.items():
        stone_image_rows[sid] = [row.to_dict() for _, row in grp.iterrows()]

    prefetched: Dict[str, Tuple[List[str], List[str]]] = {}
    dl_cache_hits = 0
    dl_cache_misses = 0
    t_prefetch_start = time.time()
    prefetch_done_count = 0
    prefetch_total = len(stone_image_rows)

    with ThreadPoolExecutor(max_workers=_PREFETCH_WORKERS) as executor:
        futures = {
            executor.submit(
                _prefetch_images_for_stone,
                sid,
                rows,
                cache_dir_str,
                max_images_per_stock,
            ): sid
            for sid, rows in stone_image_rows.items()
        }
        for future in _as_completed(futures):
            sid = futures[future]
            prefetch_done_count += 1
            try:
                _, paths, ids = future.result()
                if paths:
                    prefetched[sid] = (paths, ids)
                    for p in paths:
                        cached_path = Path(p)
                        if cached_path.stat().st_mtime < t_start:
                            dl_cache_hits += 1
                        else:
                            dl_cache_misses += 1
                else:
                    skipped.append(sid)
            except Exception as exc:
                failed[sid] = f"prefetch: {exc}"
            if prefetch_callback:
                prefetch_callback(prefetch_done_count, prefetch_total, sid)
            time.sleep(0.005)

    t_prefetch = time.time() - t_prefetch_start
    dl_total = dl_cache_hits + dl_cache_misses
    print(
        f"[V3-INCREMENTAL] Prefetch done in {t_prefetch:.1f}s — "
        f"{len(prefetched)} stones with images, "
        f"download cache: {dl_cache_hits}/{dl_total} hits "
        f"({(dl_cache_hits/dl_total*100) if dl_total else 0:.0f}%)"
    )

    completed = 0
    cv_total_stones = len(prefetched)
    total_images_processed = 0
    analysis_cache_hits = 0
    analysis_cache_misses = 0
    t_cv_start = time.time()

    _cpu_workers = max(1, min(os.cpu_count() or 2, 2))

    with ProcessPoolExecutor(max_workers=_cpu_workers) as executor:
        futures = {
            executor.submit(_cv_process_stone, sid, paths, ids, max_pixels): sid
            for sid, (paths, ids) in prefetched.items()
        }

        for future in _as_completed(futures):
            if cancel_event and cancel_event.is_set():
                print(f"[V3-INCREMENTAL] Cancelled after {completed}/{cv_total_stones}")
                executor.shutdown(wait=False, cancel_futures=True)
                break

            sid = futures[future]
            try:
                _, cv_data = future.result()
            except Exception as exc:
                failed[sid] = str(exc)
                completed += 1
                if progress_callback:
                    progress_callback(completed, cv_total_stones, sid)
                time.sleep(0.01)
                continue

            completed += 1

            if cv_data is None:
                skipped.append(sid)
                if progress_callback:
                    progress_callback(completed, cv_total_stones, sid)
                time.sleep(0.01)
                continue

            analyses = cv_data["analyses"]
            image_paths = cv_data["image_paths"]
            image_ids = cv_data["image_ids"]
            total_images_processed += len(image_paths)
            analysis_cache_hits += cv_data.get("_analysis_cache_hits", 0)
            analysis_cache_misses += cv_data.get("_analysis_cache_misses", 0)

            built[sid] = {
                "analyses": analyses,
                "image_paths": image_paths,
                "image_ids": image_ids,
            }

            if completed % 10 == 0:
                elapsed = time.time() - t_cv_start
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = cv_total_stones - completed
                eta_min = (remaining / rate / 60) if rate > 0 else 0
                print(
                    f"[V3-INCREMENTAL] {completed}/{cv_total_stones} done "
                    f"({rate:.2f} stones/sec, ETA ~{eta_min:.0f} min)"
                )

            if progress_callback:
                progress_callback(completed, cv_total_stones, sid)
            time.sleep(0.01)

    t_cv = time.time() - t_cv_start
    t_total = time.time() - t_start
    analysis_total = analysis_cache_hits + analysis_cache_misses

    print(
        f"[V3-INCREMENTAL] Done: {len(built)} built, "
        f"{len(skipped)} skipped, {len(failed)} failed"
    )
    print(
        f"[V3-TIMING] total={t_total:.1f}s prefetch={t_prefetch:.1f}s "
        f"cv+agg={t_cv:.1f}s "
        f"avg/stone={t_total/max(total,1):.2f}s "
        f"avg/image={t_cv/max(total_images_processed,1):.2f}s "
        f"dl_cache={dl_cache_hits}/{dl_total} "
        f"analysis_cache={analysis_cache_hits}/{analysis_total} "
        f"images={total_images_processed}"
    )

    _predictions: Dict[str, dict] = {}
    if use_model_predictions:
        for sid in built:
            try:
                _predictions[sid] = predict_stock_v3(sid)
            except Exception as e:
                print(f"[V3] Model prediction failed for {sid}: {e}")

    return {
        "built": built,
        "skipped": skipped,
        "failed": failed,
        "predictions": _predictions,
    }
