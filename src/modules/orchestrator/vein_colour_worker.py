"""Task #358 — Vein Colour Extractor (in-process backfill worker).

Same pattern as foundation_worker (Task #351): a daemon thread launched
from FastAPI startup that drains the queue of active-dataset stones
missing `stock_tags_ai.vein_colour`, classifies the vein colour from
per-pixel LAB stats over `color_agent._compute_simple_vein_mask`, and
writes the tag.  Idempotent + auto-resumable: stones already carrying
`vein_colour` are skipped, so a restart picks up where the prior run
left off.

Vein vocab (8 values): gold, grey, white, pink, green, black, none, mixed.

Decision tree (validated against Galapagos Dark = gold, Fusion Black =
grey, Valle Bianca = grey, White Onyx = pink, plus Alex Black /
Silver Mink controls = grey — see
`scripts/validate_vein_colour_anchors.py`):

  1. vein_share < 0.005             → "none"
  2. chromatic_share < 0.15         → classify by mean L* of neutral
                                       vein pixels (white >=70, black <30,
                                       else grey)
  3. top chromatic bucket share of
     chromatic mass < 0.55          → fall through to neutral path
                                       (noise, not a real chromatic vein)
  4. 2nd bucket share of chromatic
     >= 0.30                        → "mixed"
  5. else                            → top bucket name
                                       (gold | green | grey | pink)
"""

from __future__ import annotations
from src.modules.ingestion import services as ingestion_services
from src.modules.taxonomy import services as taxonomy_svc

from src.utils import image_fetch
from src.modules.ml_inference import services as ml_services
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
logger = logging.getLogger("src.modules.ml_inference.vein_colour_worker")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def _get_active_dataset() -> str:
    return ingestion_services.get_active_dataset_id()

# ---- Worker singleton state ----------------------------------------------
_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_stop_event = threading.Event()

def _log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(f"[VeinWorker] {line}", flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass




def classify_vein_colour_for_stone(sid: str) -> tuple[Optional[str], dict]:
    """Fetch images + classify. Returns (label_or_None, debug_dict).
    Returns (None, {...}) when no images are available for the stone."""


    active_dataset = ingestion_services.get_active_dataset_id()
    if not active_dataset:
        return None, {"reason": "no_active_dataset"}

    meta = ingestion_services.get_sample_images_for_stock(sid, active_dataset, limit=3)
    urls = [m.image_url for m in meta if m.image_url]
    if not urls:
        return None, {"reason": "no_image_urls"}
    imgs = image_fetch.fetch_images_from_urls(urls, max_images=3)
    if not imgs:
        return None, {"reason": "fetch_failed", "url_count": len(urls)}
        
    label = ml_services.classify_vein_colour_from_images(imgs)
    return label, {}


# --------------------------------------------------------------------------
# Pending queue + write path
# --------------------------------------------------------------------------
def _get_pending_stones() -> list[str]:
    """Active-dataset stones with NO vein_colour tag yet."""

    dataset_data = ingestion_services.get_all_stone_data(_get_active_dataset())
    cids = [
        s.get("company_stone_id")
        for s in dataset_data.get("stones", [])
        if s.get("company_stone_id")
    ]
    return taxonomy_svc.get_stones_missing_ai_tag(cids, "vein_colour")


def _write_tag(
    sid: str, value: str, source_label: str = "backfill_t358_worker"
) -> None:
    taxonomy_svc.write_ml_predictions_batch(sid, {"vein_colour": (value, 0.85)})


# --------------------------------------------------------------------------
# Worker loop
# --------------------------------------------------------------------------
def _worker_loop() -> None:
    pending = _get_pending_stones()
    total = len(pending)
    _log(f"START pending={total}")
    processed = 0
    written = 0
    no_img = 0
    failed = 0
    t0 = time.time()
    for sid in pending:
        if _stop_event.is_set():
            _log(f"STOP_REQUESTED at processed={processed}")
            break
        processed += 1
        try:
            label, debug = classify_vein_colour_for_stone(sid)
            if label is None:
                no_img += 1
            else:
                _write_tag(sid, label)
                written += 1
        except Exception as exc:
            failed += 1
            _log(f"error sid={sid}: {exc!r}")

        if processed % 100 == 0:
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1)
            eta_min = (total - processed) / max(rate, 0.01) / 60
            _log(
                f"PROGRESS {processed}/{total} "
                f"written={written} no_img={no_img} failed={failed} "
                f"rate={rate:.2f}/s eta={eta_min:.1f}min"
            )
    elapsed = time.time() - t0
    _log(
        f"DONE processed={processed}/{total} "
        f"written={written} no_img={no_img} failed={failed} "
        f"elapsed_min={elapsed/60:.1f}"
    )


def start_worker_if_needed() -> bool:
    """Idempotent launcher. Returns True if a worker thread was started."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return False
        try:
            pending_count = len(_get_pending_stones())
        except Exception as exc:
            _log(f"pending-check failed: {exc!r}")
            return False
        if pending_count == 0:
            _log("no pending stones, worker not started")
            return False
        _log(f"launching daemon thread (pending={pending_count})")
        _stop_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="VeinColourBackfillWorker",
            daemon=True,
        )
        _worker_thread.start()
        return True
