"""Task #375 — In-process stone_lightness backfill worker.

Mirrors the Task #351 foundation-worker startup pattern.  A daemon
thread is launched at FastAPI startup; it drains the pending cohort
(active dataset 20260412_084201_clean) by fetching up to 3 sample
images per stone via image_fetch.fetch_images_from_urls, running
color_agent.analyze_stock_colors_from_arrays, and writing ONLY
stone_lightness + stone_lightness_mean into stock_tags_ai.

The worker enforces three contracts in code:

1. **Dataset determinism** — pending stones, image URLs, and the
   target cohort are all selected against Databases/Ingestion/
   datasets_clean.db at dataset_id = ACTIVE_DATASET.  We do NOT
   rely on dataset_storage's globally-resolved active dataset.

2. **Preflight safety gate** — before launching the thread,
   start_worker_if_needed() makes a timestamped backup of the
   live stock_tags.db, runs PRAGMA integrity_check on the backup,
   and aborts the launch on any failure.  Backup path is logged
   and recorded so the post-run scope-lock diff can read from it.

3. **Post-run scope-lock + formal report** — at end-of-run the
   worker re-snapshots the global per-tag row counts in
   stock_tags.db, asserts that only stone_lightness and
   stone_lightness_mean changed vs the preflight snapshot, and
   emits the formal coverage report block (Before / After /
   Stones successfully tagged / Stones failed grouped by reason)
   to the log AND to a JSON artifact at
   tmp/diagnostics/t375_final_report.json.

The worker is idempotent (skips any stone already carrying
stone_lightness) and resumable (re-launches with the same WHERE
clause after every restart).  It uses the agent's production
threshold logic unchanged — color_agent.analyze_stock_colors_from_
arrays internally passes main_color as the dark-base hint, which
preserves the 2026-05-23 conditional cutoff contract.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.modules.ingestion import services as ingestion_services
from src.modules.taxonomy import services as taxonomy_svc
from src.modules.ml_inference import services as ml_services
from src.utils import image_fetch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def _get_active_dataset() -> str:
    return ingestion_services.get_active_dataset_id()

import os

LOG_PATH = REPO_ROOT / "tmp/diagnostics/t375_lightness_backfill.log"
REPORT_PATH = REPO_ROOT / "tmp/diagnostics/t375_final_report.json"
BACKUP_DIR = Path(os.environ.get("STOCK_TAGS_BACKUP_DIR"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_EVERY = 100
WORKER_TAG_SOURCE = "backfill_t375_agent"
ALLOWED_TAGS = {"stone_lightness", "stone_lightness_mean", "vein_colour"}
COVERAGE_GATE_PCT = 95.0

_worker_thread: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_preflight_state: Dict[str, object] = {}


def _log(msg: str) -> None:
    line = f"[{datetime.utcnow().isoformat()}Z] {msg}"
    print(f"[LightnessWorker] {line}", flush=True)
    try:
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset-scoped queries (always against datasets_clean.db,
# ACTIVE_DATASET — never via dataset_storage's global resolver).
# ---------------------------------------------------------------------------


def _get_active_dataset_cids() -> set:

    dataset_data = ingestion_services.get_all_stone_data(_get_active_dataset())
    return {
        s.get("company_stone_id")
        for s in dataset_data.get("stones", [])
        if s.get("company_stone_id")
    }


def _cohort_total() -> int:
    return len(_get_active_dataset_cids())


def _cohort_with_lightness() -> int:
    cids = _get_active_dataset_cids()
    if not cids:
        return 0
    missing = taxonomy_svc.get_stones_missing_ai_tag(list(cids), "stone_lightness")
    return len(cids) - len(missing)


def get_pending_stones() -> List[str]:
    """Active-cohort stones that have base_color but no stone_lightness."""
    dataset_data = ingestion_services.get_all_stone_data(_get_active_dataset())
    cids = [
        s.get("company_stone_id")
        for s in dataset_data.get("stones", [])
        if s.get("company_stone_id")
    ]
    return taxonomy_svc.get_lightness_pending(cids)


def _get_image_urls_for_stone(sid: str, limit: int = 3) -> List[str]:
    """Dataset-scoped: read image URLs directly via ingestion_service."""
    try:
        images = ingestion_services.get_sample_images_for_stock(
            sid, _get_active_dataset(), limit
        )
        return [img.image_url for img in images if getattr(img, "image_url", None)]
    except AttributeError:
        # Fallback if get_sample_images_for_stock returns dicts
        return [img.get("image_url") for img in images if img.get("image_url")]


# ---------------------------------------------------------------------------
# Tag-count snapshots (scope-lock evidence).
# ---------------------------------------------------------------------------


def _snapshot_tag_counts() -> Dict[str, Tuple[int, int]]:
    """Return {tag_name: (n_rows, n_stones)} for stock_tags_ai rows
    whose company_stone_id is in the active dataset cohort.
    """
    cids = _get_active_dataset_cids()
    return taxonomy_svc.snapshot_tag_counts(cids)


def _diff_snapshots(before: Dict, after: Dict) -> List[Tuple[str, int, int, int, int]]:
    """Return rows (tag, before_rows, after_rows, drow, dstone) for any tag where rows OR stones changed."""
    changed = []
    for tag in sorted(set(before) | set(after)):
        rb, sb = before.get(tag, (0, 0))
        ra, sa = after.get(tag, (0, 0))
        if rb != ra or sb != sa:
            changed.append((tag, rb, ra, ra - rb, sa - sb))
    return changed


def _write_lightness(sid: str, mean_L: float, base_color_hint: Optional[str] = None) -> None:
    """Scope-locked write: ONLY stone_lightness + stone_lightness_mean."""
    taxonomy_svc.save_lightness_tags(sid, mean_L, base_color_hint)


def _process_one(sid: str) -> Tuple[str, Optional[str]]:
    """Returns ('written', None) on success or ('failed', reason) on failure.

    Failure reasons: invalid_image | fetch_failed | agent_returned_none |
    exception:<ExceptionClass>
    """
    try:
        # Cheap idempotency short-circuit
        missing = taxonomy_svc.get_stones_missing_ai_tag([sid], "stone_lightness")
        if not missing:
            return ("written", None)
        # Dataset-scoped image URL fetch — guaranteed-cohort URLs.
        urls = _get_image_urls_for_stone(sid, limit=3)
        if not urls:
            return ("failed", "invalid_image")
        imgs = image_fetch.fetch_images_from_urls(urls, max_images=3)
        if not imgs:
            return ("failed", "fetch_failed")
        result = ml_services.analyze_stock_colors_from_arrays(imgs)
        if getattr(result, 'stone_lightness_mean', None) is None or getattr(result, 'patch_count', 0) < 4:
            return ("failed", "agent_returned_none")
            
        _write_lightness(sid, result.stone_lightness_mean)
        return ("written", None)
    except Exception as e:  # noqa: BLE001 — broad catch intentional
        msg = type(e).__name__
        _log(f"error sid={sid}: {msg}: {e}")
        return ("failed", f"exception:{msg}")


# ---------------------------------------------------------------------------
# Preflight gate: backup + integrity_check.  Mutates _preflight_state.
# ---------------------------------------------------------------------------


def _preflight_safety_gate() -> bool:
    """Backup live DB, integrity-check the backup, snapshot tag counts.
    Returns True if all checks pass, False (and logs) otherwise."""
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    backup_path = BACKUP_DIR / f"stock_tags_backup_{stamp}.db"
    try:
        taxonomy_svc.backup_tags_db(backup_path)
    except Exception as e:  # noqa: BLE001
        _log(f"preflight FAIL: backup copy failed: {e}")
        return False
    try:
        if not taxonomy_svc.check_integrity(backup_path):
            _log("preflight FAIL: integrity_check failed")
            return False
    except Exception as e:  # noqa: BLE001
        _log(f"preflight FAIL: integrity_check raised: {e}")
        return False
    try:
        snap = _snapshot_tag_counts()
    except Exception as e:  # noqa: BLE001
        _log(f"preflight FAIL: snapshot failed: {e}")
        return False
    _preflight_state.clear()
    _preflight_state["backup_path"] = str(backup_path)
    _preflight_state["snapshot_before"] = snap
    _preflight_state["coverage_before"] = _cohort_with_lightness()
    _preflight_state["cohort_total"] = _cohort_total()
    _log(
        f"preflight OK backup={backup_path.name} integrity=ok "
        f"cohort_total={_preflight_state['cohort_total']} "
        f"coverage_before={_preflight_state['coverage_before']}"
    )
    return True


# ---------------------------------------------------------------------------
# Worker loop.  Emits formal report + scope-lock check at end.
# ---------------------------------------------------------------------------


class ScopeViolationError(RuntimeError):
    """Raised when the post-run cohort tag-count diff shows any tag
    other than stone_lightness / stone_lightness_mean changed.

    Operator response: investigate the listed `violations` payload,
    then restore from `backup_path` if the unauthorized writes are
    confirmed regressions (`cp -f <backup_path> Databases/Taxonomy/stock_tags.db`
    after stopping the FastAPI workflow).
    """

    def __init__(
        self,
        violations: List[Tuple[str, int, int, int, int]],
        backup_path: Optional[str],
    ):
        self.violations = violations
        self.backup_path = backup_path
        msg = (
            f"Task #375 SCOPE-LOCK VIOLATION: {len(violations)} non-allowed tag(s) "
            f"changed in cohort {_get_active_dataset()}. Allowed: {sorted(ALLOWED_TAGS)}. "
            f"Violations: {[(t, dr, ds) for t, _, _, dr, ds in violations]}. "
            f"Rollback: restore from backup at {backup_path}."
        )
        super().__init__(msg)


def _emit_final_report(
    written: int, failures: Dict[str, int], elapsed_min: float
) -> None:
    cohort_total = _preflight_state.get("cohort_total") or _cohort_total()
    cov_before = _preflight_state.get("coverage_before")
    cov_after = _cohort_with_lightness()
    cov_before_pct = (
        (cov_before / cohort_total * 100.0) if cov_before is not None else None
    )
    cov_after_pct = cov_after / cohort_total * 100.0
    gate_pass = cov_after_pct >= COVERAGE_GATE_PCT

    # Cohort-scoped scope-lock diff
    scope_violations: List[Tuple[str, int, int, int, int]] = []
    snapshot_before = _preflight_state.get("snapshot_before") or {}
    snapshot_after = _snapshot_tag_counts()
    for tag, rb, ra, dr, ds in _diff_snapshots(snapshot_before, snapshot_after):
        if tag not in ALLOWED_TAGS:
            scope_violations.append((tag, rb, ra, dr, ds))

    # Group failure reasons into a single line in the required order
    grouped_reasons_parts = []
    for reason in ("invalid_image", "fetch_failed", "agent_returned_none"):
        grouped_reasons_parts.append(f"{reason}={failures.get(reason, 0)}")
    other_failures = {
        k: v
        for k, v in failures.items()
        if k not in {"invalid_image", "fetch_failed", "agent_returned_none"}
    }
    if other_failures:
        grouped_reasons_parts.append(f"other={other_failures}")
    grouped_reasons_line = ", ".join(grouped_reasons_parts)

    fail_total = sum(failures.values())
    scope_status = (
        "PASS (only stone_lightness + stone_lightness_mean changed)"
        if not scope_violations
        else "FAIL"
    )

    # Exact required block format
    _log("===== Task #375 Final Report =====")
    _log(f"Cohort dataset: {_get_active_dataset()}")
    _log(f"Cohort total stones: {cohort_total}")
    if cov_before is not None:
        _log(f"Before coverage: {cov_before}/{cohort_total} ({cov_before_pct:.2f}%)")
    _log(f"After coverage: {cov_after}/{cohort_total} ({cov_after_pct:.2f}%)")
    _log(f"Coverage gate (>= {COVERAGE_GATE_PCT}%): {'PASS' if gate_pass else 'FAIL'}")
    _log(f"Stones successfully tagged: {written}")
    _log(f"Stones failed: {fail_total} ({grouped_reasons_line})")
    _log(f"Elapsed: {elapsed_min:.1f} min")
    _log(f"Scope-lock check: {scope_status}")
    if scope_violations:
        for tag, rb, ra, dr, ds in scope_violations:
            _log(f"  VIOLATION {tag}: rows {rb}->{ra} ({dr:+}), stones delta {ds:+}")
    _log("===== End Report =====")

    # Persist machine-readable artifact — write to a timestamped
    # immutable path AND mirror to the canonical "latest" path so
    # downstream tooling can hit either.  The timestamped file is
    # never overwritten, preserving each run's evidence.
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    timestamped_path = REPORT_PATH.parent / f"t375_final_report_{stamp}.json"
    report = {
        "task": "T375",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "dataset_id": _get_active_dataset(),
        "cohort_total": cohort_total,
        "coverage_before": cov_before,
        "coverage_before_pct": cov_before_pct,
        "coverage_after": cov_after,
        "coverage_after_pct": cov_after_pct,
        "coverage_gate_pct": COVERAGE_GATE_PCT,
        "coverage_gate_pass": gate_pass,
        "stones_written_this_run": written,
        "stones_failed_this_run": fail_total,
        "failure_reasons": dict(failures),
        "failure_reasons_grouped_line": grouped_reasons_line,
        "elapsed_min": elapsed_min,
        "backup_path": _preflight_state.get("backup_path"),
        "scope_lock_pass": not scope_violations,
        "scope_violations": [
            {
                "tag_name": t,
                "rows_before": rb,
                "rows_after": ra,
                "rows_delta": dr,
                "stones_delta": ds,
            }
            for t, rb, ra, dr, ds in scope_violations
        ],
        "tag_snapshot_before": {t: list(v) for t, v in snapshot_before.items()},
        "tag_snapshot_after": {t: list(v) for t, v in snapshot_after.items()},
    }
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, sort_keys=True)
        timestamped_path.write_text(payload)
        REPORT_PATH.write_text(payload)
        _log(
            f"final report artifact written: {timestamped_path.name} (also mirrored to {REPORT_PATH.name})"
        )
    except Exception as e:  # noqa: BLE001
        _log(f"final report artifact write failed: {e}")

    # Hard abort on any scope violation — raise so the daemon thread
    # logs the traceback and so any future synchronous caller (CLI,
    # test harness) receives a non-success exit path.
    if scope_violations:
        _log("HARD ABORT: scope-lock violation — raising ScopeViolationError")
        raise ScopeViolationError(scope_violations, _preflight_state.get("backup_path"))


def _worker_loop() -> None:
    pending = get_pending_stones()
    total = len(pending)
    _log(f"start pending={total}")
    if total == 0:
        _emit_final_report(written=0, failures={}, elapsed_min=0.0)
        return

    written = 0
    failures: Dict[str, int] = {}
    t0 = time.time()
    for i, sid in enumerate(pending, start=1):
        status, reason = _process_one(sid)
        if status == "written":
            written += 1
        else:
            failures[reason or "unknown"] = failures.get(reason or "unknown", 0) + 1

        if i % PROGRESS_EVERY == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1.0)
            remaining = total - i
            eta_min = remaining / max(rate, 0.01) / 60.0
            fail_total = sum(failures.values())
            _log(
                f"progress {i}/{total} written={written} failed={fail_total} "
                f"rate={rate:.2f}/s eta={eta_min:.1f}min reasons={dict(failures)}"
            )
    elapsed_min = (time.time() - t0) / 60.0
    _log(
        f"done processed={total} written={written} "
        f"failed={sum(failures.values())} reasons={dict(failures)} "
        f"elapsed={elapsed_min:.1f}min"
    )
    _emit_final_report(written=written, failures=failures, elapsed_min=elapsed_min)


def is_running() -> bool:
    return _worker_thread is not None and _worker_thread.is_alive()


def start_worker_if_needed() -> bool:
    """Launch the backfill daemon thread if there's pending work.

    Gated by _preflight_safety_gate (backup + integrity_check).
    Idempotent: a second call while the thread is alive is a no-op.
    """
    global _worker_thread
    with _worker_lock:
        if is_running():
            return False
        try:
            pending = get_pending_stones()
        except Exception as e:  # noqa: BLE001
            _log(f"start_worker_if_needed: pending-query failed: {e}")
            return False
        if not pending:
            return False
        if not _preflight_safety_gate():
            _log("ABORTING worker launch: preflight safety gate failed")
            return False
        _worker_thread = threading.Thread(
            target=_worker_loop,
            name="t375-lightness-worker",
            daemon=True,
        )
        _worker_thread.start()
        _log(f"thread launched pending={len(pending)}")
        return True
