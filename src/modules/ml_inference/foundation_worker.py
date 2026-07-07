"""
Foundation Compute Worker - Background processing of image embeddings.

Features:
- Runs as asyncio background task in FastAPI process
- Concurrent processing (2-4 images at a time)
- CLIP model cached in memory
- Automatic resume on server restart
- Stale lock recovery
- Robust HTTP fetching with redirects, retries, browser headers
- URL normalization to strip Cloudflare CDN wrappers
"""

import asyncio
import logging
import io
import re
import httpx
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from src.modules.ml_inference import services as ml_services
from src.modules.ml_inference.foundation_storage import (
    foundation_storage, 
    FoundationStorage,
    StockImage, 
    JobStatus,
    OUTLIER_THRESHOLD
)
from src.modules.ml_inference.embeddings import (
    compute_image_embedding,
    compute_embedding_aggregates,
)

logger = logging.getLogger(__name__)

CONCURRENCY = 3
BATCH_SIZE = 3
UPDATE_INTERVAL = 10
IMAGE_TIMEOUT_CONNECT = 10.0
IMAGE_TIMEOUT_READ = 30.0
WORKER_SLEEP_EMPTY = 2.0
WORKER_SLEEP_ERROR = 5.0

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://images.stonestocks.com/",
    "Connection": "keep-alive",
}

BROWSER_HEADERS_NO_REFERER = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_FETCH_RETRIES = 3
RETRY_BACKOFF = [1.0, 3.0, 7.0]


def normalize_image_url(url: str) -> str:
    """
    Normalize Cloudflare CDN-cgi image URLs to extract origin URL.
    
    Examples:
    - /cdn-cgi/image/.../https://images.stonestocks.com/images/foo.jpg 
      → https://images.stonestocks.com/images/foo.jpg
    - https://foo.com/cdn-cgi/image/format=auto/https://images.stonestocks.com/images/bar.jpg
      → https://images.stonestocks.com/images/bar.jpg
    """
    if not url:
        return url
    
    pattern = r'(?:/cdn-cgi/image/[^/]*/)(https://images\.stonestocks\.com/images/[^\s]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    
    pattern2 = r'https://images\.stonestocks\.com/images/[^\s]+'
    match2 = re.search(pattern2, url)
    if match2 and '/cdn-cgi/' in url:
        return match2.group(0)
    
    return url


def classify_error(status_code: Optional[int] = None, error: Optional[Exception] = None, content_type: str = "") -> str:
    """Classify errors into structured failure reasons."""
    if status_code:
        if status_code == 403:
            return "forbidden_403"
        elif status_code == 415:
            return "unsupported_415"
        elif status_code == 404:
            return "not_found_404"
        elif status_code >= 500:
            return f"server_error_{status_code}"
        elif status_code >= 400:
            return f"client_error_{status_code}"
    
    if error:
        error_str = str(error).lower()
        if "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        if "redirect" in error_str:
            return "redirect_loop"
        if "connect" in error_str:
            return "connection_error"
    
    if content_type and not content_type.startswith("image/"):
        return "not_image"
    
    return "unknown"


class FoundationWorker:
    """Background worker for computing image embeddings."""
    
    def __init__(self, storage: FoundationStorage = None):
        self.storage = storage or foundation_storage
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_count = 0
        self._last_update = 0
    
    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()
    
    def start(self):
        """Start the worker as a background task."""
        if self.is_running:
            logger.info("Worker already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Foundation worker started")
    
    def stop(self):
        """Stop the worker gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Foundation worker stopped")
    
    async def _run_loop(self):
        """Main worker loop."""
        logger.info("Foundation worker loop started")
        
        recovered = self.storage.recover_stale_locks()
        if recovered > 0:
            logger.info(f"Recovered {recovered} stale locks")
        
        while self._running:
            try:
                images = self.storage.claim_batch_pending(BATCH_SIZE)
                
                if not images:
                    job = self.storage.get_active_job()
                    if job:
                        pending = self.storage.get_pending_count()
                        if pending == 0:
                            self.storage.update_job_progress(job.id)
                            self.storage.finish_job(job.id, JobStatus.DONE)
                            logger.info(f"Job {job.id} completed")
                    
                    await asyncio.sleep(WORKER_SLEEP_EMPTY)
                    continue
                
                tasks = [self._process_image(img) for img in images]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                self._processed_count += len(images)
                
                if self._processed_count - self._last_update >= UPDATE_INTERVAL:
                    job = self.storage.get_active_job()
                    if job:
                        self.storage.update_job_progress(job.id)
                    self._last_update = self._processed_count
                    logger.debug(f"Processed {self._processed_count} images")
                
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(WORKER_SLEEP_ERROR)
        
        logger.info("Foundation worker loop ended")
    
    async def _fetch_image_with_retry(self, url: str) -> Tuple[Optional[bytes], Optional[str], str]:
        """
        Fetch image with retries, redirects, and browser-like headers.
        
        Returns: (image_bytes, error_reason, final_url)
        """
        normalized_url = normalize_image_url(url)
        urls_to_try = [normalized_url]
        if normalized_url != url:
            urls_to_try.append(url)
        
        last_error = None
        last_reason = "unknown"
        final_url = normalized_url
        
        timeout = httpx.Timeout(IMAGE_TIMEOUT_READ, connect=IMAGE_TIMEOUT_CONNECT)
        
        for try_url in urls_to_try:
            for attempt in range(MAX_FETCH_RETRIES):
                headers = BROWSER_HEADERS if attempt < 2 else BROWSER_HEADERS_NO_REFERER
                
                try:
                    async with httpx.AsyncClient(
                        timeout=timeout,
                        follow_redirects=True,
                        max_redirects=10
                    ) as client:
                        resp = await client.get(try_url, headers=headers)
                        final_url = str(resp.url)
                        
                        if resp.status_code in RETRY_STATUS_CODES and attempt < MAX_FETCH_RETRIES - 1:
                            backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                            await asyncio.sleep(backoff)
                            continue
                        
                        if resp.status_code == 403:
                            if attempt == 0:
                                await asyncio.sleep(1.0)
                                continue
                            last_reason = "forbidden_403"
                            continue
                        
                        if resp.status_code == 415:
                            last_reason = "unsupported_415"
                            break
                        
                        if resp.status_code >= 400:
                            last_reason = classify_error(status_code=resp.status_code)
                            continue
                        
                        content_type = resp.headers.get("content-type", "")
                        if not content_type.startswith("image/"):
                            if try_url == normalized_url and url != normalized_url:
                                break
                            last_reason = "not_image"
                            continue
                        
                        return resp.content, None, final_url
                        
                except httpx.TimeoutException as e:
                    last_error = e
                    last_reason = "timeout"
                    if attempt < MAX_FETCH_RETRIES - 1:
                        backoff = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                        await asyncio.sleep(backoff)
                except httpx.TooManyRedirects:
                    last_reason = "redirect_loop"
                    break
                except Exception as e:
                    last_error = e
                    last_reason = classify_error(error=e)
                    if attempt < MAX_FETCH_RETRIES - 1:
                        await asyncio.sleep(1.0)
        
        return None, last_reason, final_url
    
    async def _process_image(self, image: StockImage):
        """Process a single image: fetch, compute embedding, save."""
        try:
            image_bytes, error_reason, final_url = await self._fetch_image_with_retry(image.image_url)
            
            if image_bytes is None:
                error_msg = f"{error_reason}|url={image.image_url}|final={final_url}"
                self.storage.mark_failed(image.id, error_msg)
                logger.debug(f"Failed {image.id}: {error_reason}")
                return
            
            try:
                pil_image = Image.open(io.BytesIO(image_bytes))
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
            except Exception as e:
                error_msg = f"invalid_image|{str(e)[:100]}"
                self.storage.mark_failed(image.id, error_msg)
                return
            
            embedding_nd = await asyncio.to_thread(
                compute_image_embedding, pil_image
            )
            embedding = embedding_nd.tolist() if embedding_nd is not None else None
            
            if embedding is None:
                raise ValueError("Embedding computation returned None")
            
            self.storage.mark_done(image.id, embedding)
            logger.debug(f"Computed embedding for {image.id}")
            
        except Exception as e:
            error_msg = f"processing_error|{str(e)[:200]}"
            self.storage.mark_failed(image.id, error_msg)
            logger.warning(f"Failed to process {image.id}: {str(e)[:100]}")


_worker_instance: Optional[FoundationWorker] = None


def get_worker() -> FoundationWorker:
    """Get or create the global worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = FoundationWorker()
    return _worker_instance


def start_worker_if_needed():
    """Start worker if there's pending work or an active job."""
    worker = get_worker()
    if worker.is_running:
        return False
    
    job = foundation_storage.get_active_job()
    pending = foundation_storage.get_pending_count()
    
    if job or pending > 0:
        worker.start()
        return True
    return False


def compute_stock_aggregates(company_stone_id: str) -> Optional[dict]:
    """
    Compute aggregates for a single stock:
    - centroid = normalize(mean(embeddings))
    - rep_image = image with max cosine to centroid
    - consistency_score = median cosine to centroid
    - outlier_count = count(cosine < 0.78)
    """
    embeddings_data = foundation_storage.get_done_embeddings_for_stock(company_stone_id)
    
    if not embeddings_data:
        return None
    
    embeddings = np.array([e[2] for e in embeddings_data])
    image_ids = [e[0] for e in embeddings_data]
    image_urls = [e[1] for e in embeddings_data]
    
    math_result = compute_embedding_aggregates(embeddings, OUTLIER_THRESHOLD)
    
    rep_image_id = image_ids[math_result["best_idx"]]
    rep_image_url = image_urls[math_result["best_idx"]]
    
    foundation_storage.save_stock_aggregate(
        company_stone_id=company_stone_id,
        rep_image_id=rep_image_id,
        centroid_embedding=math_result["centroid_embedding"],
        consistency_score=math_result["consistency_score"],
        outlier_count=math_result["outlier_count"],
        image_count=len(embeddings_data)
    )
    
    return {
        "company_stone_id": company_stone_id,
        "rep_image_id": rep_image_id,
        "rep_image_url": rep_image_url,
        "consistency_score": math_result["consistency_score"],
        "outlier_count": math_result["outlier_count"],
        "image_count": len(embeddings_data)
    }


async def rebuild_all_aggregates() -> dict:
    """Rebuild aggregates for all stocks with DONE images."""
    stock_ids = foundation_storage.get_all_stock_ids_with_done_images()
    
    computed = 0
    errors = 0
    
    for stock_id in stock_ids:
        try:
            result = await asyncio.to_thread(compute_stock_aggregates, stock_id)
            if result:
                computed += 1
        except Exception as e:
            logger.error(f"Failed to compute aggregates for {stock_id}: {e}")
            errors += 1
    
    return {
        "total_stocks": len(stock_ids),
        "computed": computed,
        "errors": errors
    }
