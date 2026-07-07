"""Per-dataset run-history aggregator and rollback initiator.

Surfaces per-dataset color and pattern enrichment runs in the canonical
schema defined by
``docs/admin_product_direction/color_pattern_admin_tracks_v1.md`` and
owns track-scoped rollback initiation for Surface C (Task #270).

Two responsibilities:

1. Read-only aggregation (Tasks #268, #269) — exposes
   ``get_runs_for_dataset`` (combined color+pattern payload) and
   ``get_rollback_events_for_dataset`` (read-only ledger view consumed
   by Surfaces A and B). The aggregation payload itself never carries a
   rollback action affordance; the UI synthesizes it from
   ``reversible`` + ``status`` only.

2. Rollback initiation (Task #270, extended in Task #274) — exposes
   ``initiate_rollback`` and the ``ROLLBACK_NOT_REVERSIBLE_TOOLTIP``
   constant. Strictly track-scoped: a color rollback never mutates
   pattern state and vice versa. Successful rollbacks append a
   ``rolled_back`` event to
   ``artifacts/rollback_events/<dataset>/<track>.jsonl``; subsequent
   reads through ``get_runs_for_dataset`` flip the run's status to
   ``rolled_back`` so all three surfaces (A/B/C) converge on the same
   state.

   Per-run reversibility signal (Task #274): a single pattern run is
   reversible iff its own ``run_meta.json`` carries a non-empty
   ``rollback_command`` string. This signal lives ONLY inside that one
   run's artifact — never in a global flag, env var, code constant, or
   version gate — so introducing the pattern rollback wrapper does not
   automatically flip historical pattern runs to reversible. Pattern
   runs without that field stay not-reversible and the endpoint refuses
   them with the spec-locked tooltip copy. Color runs already followed
   the same per-run rule via ``summary.json``'s ``rollback_command``.

Data sources (existing artifacts only — no schema changes):
- ``artifacts/enrichment_runs/<dataset_id>/<run_id>/run_meta.json`` — generic
  enrichment runs (color when ``coverage_field == "main_color"``, pattern when
  ``coverage_field == "pattern_family"``). For pattern runs the optional
  ``rollback_command`` field is the per-run reversibility signal.
- ``artifacts/color_coverage_fix/<run_id>/summary.json`` — Task #223-style
  targeted color enrichment runs, attributed to the ``dataset_id`` carried in
  ``summary.json``. The summary's ``rollback_command`` is what makes the run
  reversible.
- ``artifacts/rollback_events/<dataset_id>/<track>.jsonl`` — append-only
  rollback ledger written by ``initiate_rollback`` and read by all three
  admin surfaces.
"""

from __future__ import annotations
import uuid
import json
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# parent × 3 = src/modules/ml_inference → src/modules → src → project root
ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENRICHMENT_RUNS_DIR = ROOT / "artifacts" / "enrichment_runs"
COLOR_COVERAGE_FIX_DIR = ROOT / "artifacts" / "color_coverage_fix"
ROLLBACK_EVENTS_DIR = ROOT / "artifacts" / "rollback_events"
EXPERIMENTS_DIR = ROOT / "artifacts" / "experiments"

STATUS_PREVIEWED = "previewed"
STATUS_APPLIED = "applied"
STATUS_FAILED = "failed"
STATUS_ROLLED_BACK = "rolled_back"

ALLOWED_STATUSES = frozenset(
    {STATUS_PREVIEWED, STATUS_APPLIED, STATUS_FAILED, STATUS_ROLLED_BACK}
)

COLOR_TARGET_FIELDS = frozenset({"main_color"})
PATTERN_TARGET_FIELDS = frozenset({"pattern_family"})

TRACK_COLOR = "color"
TRACK_PATTERN = "pattern"
ALLOWED_TRACKS = frozenset({TRACK_COLOR, TRACK_PATTERN})

EMPTY_STATE_COPY = "No runs yet for this track."
ROLLBACKS_EMPTY_COPY = "No rollbacks recorded for this track yet."


@dataclass
class RunRow:
    """One row in a Color Runs / Pattern Runs sub-table.

    Shape mirrors the canonical schema in the product direction brief:
    ``run_id``, target field, formula/version, status, cohort size,
    ``applied_at``, ``previewed_at``, reversible indicator, report link,
    and (Task #280) the optional ``not_reversible_reason`` consumed by
    the column's hover tooltip.

    There is intentionally no ``rollback_*`` field on this dataclass — this
    module never returns a rollback affordance. Rollback wiring lives in
    a separate downstream task.
    """

    run_id: str
    target_field: str
    formula_version: str
    status: str
    cohort_size: Optional[int]
    applied_at: Optional[str]
    previewed_at: Optional[str]
    reversible: Optional[bool]
    report_url: Optional[str]
    not_reversible_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["status"] not in ALLOWED_STATUSES:
            d["status"] = STATUS_FAILED
        return d


# Task #280: status-aware hover-tooltip copy for the "not reversible"
# pill on the per-dataset run history sub-tables. Operators flagged that
# the honest-but-vague "not reversible" pill from Task #277 doesn't say
# *why*. These four literal strings are spec-locked by the task brief —
# do NOT paraphrase, truncate, or concatenate. Adding a new reason means
# updating the brief first.
NOT_REVERSIBLE_REASON_FAILED = (
    "Run did not complete successfully. Nothing was applied, "
    "so there is nothing to roll back."
)
NOT_REVERSIBLE_REASON_PREVIEWED = (
    "Preview only — not applied. Apply this run first before " "rollback is available."
)
NOT_REVERSIBLE_REASON_ROLLED_BACK = (
    "Already rolled back. The prior applied state has been restored."
)
NOT_REVERSIBLE_REASON_APPLIED_NO_SCRIPT = (
    "No rollback script available for this run. "
    "Contact engineering to reverse manually."
)


def resolve_not_reversible_reason(
    status: str, reversible: Optional[bool], has_rollback_command: bool
) -> Optional[str]:
    """Map a row's (status, reversible, has_rollback_command) triple to
    the spec-locked tooltip string for the "not reversible" pill.

    Returns ``None`` for any row that should NOT carry a tooltip:
    - ``reversible is True`` (the green "reversible" pill — out of scope).
    - ``reversible is None`` (the ``—`` cell — no claim being made).
    - The preview row, which has its own dedicated "n/a — preview" pill
      and tooltip rendered by the template (handled there, not here).

    For ``reversible is False`` rows the four spec branches are:
    - ``status == failed`` → run did not complete.
    - ``status == previewed`` → preview only, never applied.
    - ``status == rolled_back`` → already rolled back.
    - ``status == applied`` AND no ``rollback_command`` → applied but no
      rollback script available.

    An applied row with a rollback command is, by construction, reversible
    and therefore not handled here (the helper returns ``None``).
    """
    if reversible is not False:
        return None
    if status == STATUS_FAILED:
        return NOT_REVERSIBLE_REASON_FAILED
    if status == STATUS_PREVIEWED:
        return NOT_REVERSIBLE_REASON_PREVIEWED
    if status == STATUS_ROLLED_BACK:
        return NOT_REVERSIBLE_REASON_ROLLED_BACK
    if status == STATUS_APPLIED and not has_rollback_command:
        return NOT_REVERSIBLE_REASON_APPLIED_NO_SCRIPT
    return None


def _classify_status(run_id: str, run_meta: Dict[str, Any]) -> str:
    """Map an enrichment_runs run_meta payload to a canonical status.

    Vocabulary (from the brief): ``previewed`` / ``applied`` / ``failed`` /
    ``rolled_back``. ``rolled_back`` is intentionally unreachable from
    on-disk artifacts in this read-only task — it requires a rollback-event
    ledger that is established in a downstream task. ``failed`` is derived
    from explicit failure markers in the run metadata.
    """
    rid = (run_id or "").lower()
    if isinstance(run_meta, dict):
        # explicit failure markers take precedence over preview/applied
        if run_meta.get("aborted") is True:
            return STATUS_FAILED
        gate_passed = run_meta.get("gate_passed")
        if gate_passed is False:
            return STATUS_FAILED
        status = str(run_meta.get("status") or "").lower()
        if status in ALLOWED_STATUSES:
            return status
    if rid.startswith("audit_") or rid.endswith("_dryrun") or "_dryrun_" in rid:
        return STATUS_PREVIEWED
    extra = run_meta.get("extra") if isinstance(run_meta, dict) else None
    mode = ""
    if isinstance(extra, dict):
        mode = str(extra.get("mode") or "").lower()
    if mode in {"audit", "preview", "dry_run", "dryrun"}:
        return STATUS_PREVIEWED
    return STATUS_APPLIED


def _enrichment_runs_for_dataset(dataset_id: str) -> List[RunRow]:
    base = ENRICHMENT_RUNS_DIR / dataset_id
    if not base.exists() or not base.is_dir():
        return []
    rows: List[RunRow] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        run_id = str(meta.get("run_id") or run_dir.name)
        target = str(meta.get("coverage_field") or "")
        if not target:
            continue
        status = _classify_status(run_id, meta)
        formula = str(meta.get("algorithm_version") or "unknown")
        ts = meta.get("started_at")
        applied_at = ts if status == STATUS_APPLIED else None
        previewed_at = ts if status == STATUS_PREVIEWED else None
        report_url = None
        if (run_dir / "report.md").exists() or (run_dir / "report.json").exists():
            report_url = (
                f"/admin/clean-import/datasets/{dataset_id}/runs/{run_id}/report"
            )
        # Per-run reversibility signal (Task #274). Pattern runs derive
        # reversibility from THIS run's own ``rollback_command`` field
        # in run_meta.json — never from a global flag. A run is only
        # reversible if it is currently in ``applied`` status AND the
        # signal is present; otherwise the column reads "not reversible"
        # and the rollback button stays disabled with the spec-locked
        # tooltip. Generic color enrichment runs (target == main_color)
        # in this directory keep ``reversible=None`` because their
        # rollback path lives in the targeted-color-fix artifacts and
        # is read by ``_color_coverage_fix_runs_for_dataset``.
        if target in PATTERN_TARGET_FIELDS:
            has_signal = bool(meta.get("rollback_command"))
            reversible: Optional[bool] = status == STATUS_APPLIED and has_signal
        else:
            reversible = None
            has_signal = False
        # Task #280: status-aware tooltip for the "not reversible" pill.
        # Resolver returns ``None`` for green-pill / ``—`` rows so only
        # genuine "not reversible" pattern rows get a hover reason.
        not_reversible_reason = resolve_not_reversible_reason(
            status, reversible, has_signal
        )
        rows.append(
            RunRow(
                run_id=run_id,
                target_field=target,
                formula_version=formula,
                status=status,
                cohort_size=None,
                applied_at=applied_at,
                previewed_at=previewed_at,
                reversible=reversible,
                report_url=report_url,
                not_reversible_reason=not_reversible_reason,
            )
        )
    return rows


def _color_coverage_fix_runs_for_dataset(dataset_id: str) -> List[RunRow]:
    if not COLOR_COVERAGE_FIX_DIR.exists():
        return []
    rows: List[RunRow] = []
    for run_dir in sorted(COLOR_COVERAGE_FIX_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            continue
        if summary.get("dataset_id") != dataset_id:
            continue
        run_id = str(summary.get("run_id") or run_dir.name)
        is_dry = run_id.lower().endswith("_dryrun")
        # Failure markers in summary.json take precedence over the
        # dryrun/applied heuristic so a failed run never gets surfaced as
        # `applied` in the history sub-table.
        if summary.get("aborted") is True or summary.get("gate_passed") is False:
            status = STATUS_FAILED
        elif is_dry:
            status = STATUS_PREVIEWED
        else:
            status = STATUS_APPLIED
        ts = summary.get("started_at")
        applied_at = ts if status == STATUS_APPLIED else None
        previewed_at = ts if status == STATUS_PREVIEWED else None
        cohort = summary.get("cohort_size_full")
        if cohort is None:
            cohort = summary.get("stones_attempted")
        # Task #277: a color run is only reportable as reversible when
        # rolling back is even meaningful — i.e. it is currently in
        # ``applied`` status AND its summary.json carries a non-empty
        # ``rollback_command``. Failed (aborted / gate_passed=False),
        # previewed, and rolled-back rows resolve to ``False`` so the
        # column reads "not reversible" instead of contradicting the
        # (correctly hidden) rollback button. Without the status gate
        # a failed color run with a stale rollback_command in its
        # summary would render the green pill while the button is
        # invisible — operators flagged this as suspicious.
        has_rollback_command = bool(summary.get("rollback_command"))
        reversible = status == STATUS_APPLIED and has_rollback_command
        # Task #280: status-aware tooltip for the "not reversible" pill.
        not_reversible_reason = resolve_not_reversible_reason(
            status, reversible, has_rollback_command
        )
        report_url = f"/admin/clean-import/datasets/{dataset_id}/runs/{run_id}/report"
        rows.append(
            RunRow(
                run_id=run_id,
                target_field="main_color",
                formula_version="targeted_color_enrichment.v1",
                status=status,
                cohort_size=cohort,
                applied_at=applied_at,
                previewed_at=previewed_at,
                reversible=reversible,
                report_url=report_url,
                not_reversible_reason=not_reversible_reason,
            )
        )
    return rows


def _experiment_color_runs_for_dataset(dataset_id: str) -> List[RunRow]:
    """Surface display-only color experiment runs (Task #292) on the
    Color Runs sub-table.

    Each experiment run lives at::

        artifacts/experiments/<experiment_id>/<run_id>/run_meta.json

    and carries an ``applies_to_dataset_ids`` list (the dataset_ids that
    contain at least one cohort-vendor stone at baseline time). A run is
    surfaced for ``dataset_id`` iff it appears in that list.

    These runs intentionally carry ``reversible = None`` — the column
    renders the ``—`` cell, never a green pill or a rollback button.
    Per Task #268's spec the read-only aggregator never returns a
    rollback action affordance; for a display-only experiment whose
    rollback path is "flip the flag back to off in config/experiments.json",
    the convention is that there is no per-run rollback script and the
    lifecycle is owned by the experiment artifact directory, not by the
    aggregator.
    """
    if not EXPERIMENTS_DIR.exists():
        return []
    rows: List[RunRow] = []
    for experiment_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not experiment_dir.is_dir():
            continue
        for run_dir in sorted(experiment_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            if str(meta.get("track") or "").lower() != TRACK_COLOR:
                continue
            applies = meta.get("applies_to_dataset_ids") or []
            if not isinstance(applies, list) or dataset_id not in applies:
                continue
            run_id = str(meta.get("run_id") or run_dir.name)
            target = str(meta.get("coverage_field") or "main_color")
            status_raw = str(meta.get("status") or STATUS_PREVIEWED).lower()
            status = status_raw if status_raw in ALLOWED_STATUSES else STATUS_PREVIEWED
            formula = str(
                meta.get("algorithm_version")
                or meta.get("experiment_label")
                or experiment_dir.name
            )
            ts = meta.get("started_at") or meta.get("created_at")
            applied_at = ts if status == STATUS_APPLIED else None
            previewed_at = ts if status == STATUS_PREVIEWED else None
            cohort = meta.get("cohort_size")
            report_url: Optional[str] = None
            for name in (
                "pre_flip_verification.md",
                "decision.md",
                "report.md",
            ):
                if (run_dir / name).exists():
                    report_url = (
                        f"/admin/clean-import/datasets/{dataset_id}"
                        f"/runs/{run_id}/report"
                    )
                    break
            rows.append(
                RunRow(
                    run_id=run_id,
                    target_field=target,
                    formula_version=formula,
                    status=status,
                    cohort_size=cohort if isinstance(cohort, int) else None,
                    applied_at=applied_at,
                    previewed_at=previewed_at,
                    reversible=None,
                    report_url=report_url,
                    not_reversible_reason=None,
                )
            )
    return rows


def _split_track(rows: List[RunRow]) -> Dict[str, Any]:
    """Split a flat list of runs into the inline-expandable shape.

    Returns a dict with:
    - ``latest_preview``: at most one ``previewed`` row (the most recent), or
      ``None`` if no previews exist.
    - ``latest_preview_stale``: ``True`` when the latest preview is older than
      the latest applied run on this track. Drives the muted/stale hint.
    - ``runs``: history rows (applied / failed / rolled_back), most recent
      first. Latest Preview is **not** included here.
    """
    previewed = [r for r in rows if r.status == STATUS_PREVIEWED]
    history = [r for r in rows if r.status != STATUS_PREVIEWED]
    previewed.sort(key=lambda r: r.previewed_at or "", reverse=True)
    history.sort(key=lambda r: r.applied_at or "", reverse=True)
    latest_preview = previewed[0] if previewed else None
    latest_applied = next((r for r in history if r.status == STATUS_APPLIED), None)
    latest_applied_at = latest_applied.applied_at if latest_applied else None
    stale = False
    if latest_preview and latest_applied_at and latest_preview.previewed_at:
        stale = latest_preview.previewed_at < latest_applied_at
    return {
        "latest_preview": latest_preview.to_dict() if latest_preview else None,
        "latest_preview_stale": stale,
        "runs": [r.to_dict() for r in history],
        "empty_state_copy": EMPTY_STATE_COPY,
    }


def get_runs_for_dataset(dataset_id: str) -> Dict[str, Any]:
    """Return the inline-expandable payload for a single dataset row.

    Shape::

        {
          "color":   {"latest_preview": {...}|None,
                      "latest_preview_stale": bool,
                      "runs": [...],
                      "empty_state_copy": "No runs yet for this track."},
          "pattern": {... same shape ...},
        }

    Both ``color`` and ``pattern`` keys are always present, even when there
    are zero runs for that track. Callers must render both sub-tables and
    show ``empty_state_copy`` when ``runs`` is empty and
    ``latest_preview`` is ``None``.
    """
    if not dataset_id or not isinstance(dataset_id, str):
        return {
            "color": _split_track([]),
            "pattern": _split_track([]),
        }
    enrichment = _enrichment_runs_for_dataset(dataset_id)
    color_extra = _color_coverage_fix_runs_for_dataset(dataset_id)
    experiment_color = _experiment_color_runs_for_dataset(dataset_id)
    color_rows = (
        [r for r in enrichment if r.target_field in COLOR_TARGET_FIELDS]
        + color_extra
        + experiment_color
    )
    pattern_rows = [r for r in enrichment if r.target_field in PATTERN_TARGET_FIELDS]
    _apply_rollback_status(dataset_id, TRACK_COLOR, color_rows)
    _apply_rollback_status(dataset_id, TRACK_PATTERN, pattern_rows)
    return {
        "color": _split_track(color_rows),
        "pattern": _split_track(pattern_rows),
    }


def _apply_rollback_status(dataset_id: str, track: str, rows: List[RunRow]) -> None:
    """Mutate `rows` in place: any run whose run_id is the latest
    `reverted_run_id` in the ledger flips to ``rolled_back``.

    Task #280: when a row flips from ``applied`` to ``rolled_back``,
    its ``reversible`` indicator must follow (only applied runs can be
    reversible) and its ``not_reversible_reason`` is recomputed so the
    column's hover tooltip says "Already rolled back…" instead of
    silently keeping the stale "applied" reason or none at all."""
    if not rows:
        return
    events = get_rollback_events_for_dataset(dataset_id, track)
    if not events:
        return
    reverted_ids = {e["reverted_run_id"] for e in events}
    for r in rows:
        if r.run_id in reverted_ids and r.status == STATUS_APPLIED:
            r.status = STATUS_ROLLED_BACK
            # Once rolled back, the row is no longer reversible; recompute
            # the tooltip reason from the new (status, reversible) pair.
            # ``has_rollback_command`` is preserved as ``True`` because
            # only rows that previously had it could reach this branch
            # (an applied-without-script row would never have been the
            # target of a successful rollback in the first place).
            r.reversible = False
            r.not_reversible_reason = resolve_not_reversible_reason(
                r.status, r.reversible, has_rollback_command=True
            )


def _is_safe_segment(s: str) -> bool:
    return bool(s) and "/" not in s and "\\" not in s and ".." not in s


def get_rollback_events_for_dataset(
    dataset_id: str, track: str
) -> List[Dict[str, Any]]:
    """Read the rollback-events ledger at
    ``artifacts/rollback_events/<dataset_id>/<track>.jsonl``.

    Returns events sorted by ``occurred_at`` descending. Each event has
    ``reverted_run_id`` and ``restored_to_run_id`` (None means no prior
    applied run; ``restored_to_status`` is then ``"none"`` else
    ``"applied"``). Missing file / malformed lines / wrong-track lines
    are silently skipped.
    """
    if not _is_safe_segment(dataset_id):
        return []
    if track not in ALLOWED_TRACKS:
        return []
    path = ROLLBACK_EVENTS_DIR / dataset_id / f"{track}.jsonl"
    if not path.exists() or not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        ev_track = str(ev.get("track") or "").lower()
        if ev_track and ev_track != track:
            continue
        reverted = ev.get("reverted_run_id")
        if not reverted:
            continue
        restored = ev.get("restored_to_run_id")
        if restored in ("", None):
            restored_norm: Optional[str] = None
            restored_status = "none"
        else:
            restored_norm = str(restored)
            restored_status = "applied"
        events.append(
            {
                "event_id": str(ev.get("event_id") or ""),
                "dataset_id": dataset_id,
                "track": track,
                "target_field": str(ev.get("target_field") or ""),
                "reverted_run_id": str(reverted),
                "restored_to_run_id": restored_norm,
                "restored_to_status": restored_status,
                "actor": str(ev.get("actor") or ""),
                "occurred_at": str(ev.get("occurred_at") or ""),
                "reason": str(ev.get("reason") or ""),
            }
        )
    events.sort(key=lambda e: e.get("occurred_at") or "", reverse=True)
    return events


def find_run_report_path(dataset_id: str, run_id: str) -> Optional[Path]:
    """Resolve the on-disk report file for a (dataset_id, run_id) pair.

    Returns the path to the report markdown / JSON if the run exists in either
    artifacts source, else ``None``. Constrained to known artifact roots —
    this is a read-only passthrough; no path traversal outside those roots is
    permitted.
    """
    if not dataset_id or not run_id:
        return None
    if "/" in dataset_id or ".." in dataset_id:
        return None
    if "/" in run_id or ".." in run_id:
        return None
    enrichment_run = ENRICHMENT_RUNS_DIR / dataset_id / run_id
    if enrichment_run.is_dir():
        for name in ("report.md", "summary.md", "report.json"):
            p = enrichment_run / name
            if p.exists():
                return p
    color_run = COLOR_COVERAGE_FIX_DIR / run_id
    if color_run.is_dir():
        summary_path = color_run / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                if summary.get("dataset_id") != dataset_id:
                    return None
            except Exception:
                return None
            for name in ("run_report.md", "summary.json"):
                p = color_run / name
                if p.exists():
                    return p
    # Task #292 — experiment artifact passthrough. Constrained to the
    # `artifacts/experiments/<experiment_id>/<run_id>/` root and gated on
    # the run's own ``applies_to_dataset_ids`` so a report can only be
    # served for a dataset the experiment actually targets.
    if EXPERIMENTS_DIR.exists():
        for experiment_dir in EXPERIMENTS_DIR.iterdir():
            if not experiment_dir.is_dir():
                continue
            run_dir = experiment_dir / run_id
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            applies = meta.get("applies_to_dataset_ids") or []
            if not isinstance(applies, list) or dataset_id not in applies:
                continue
            for name in (
                "pre_flip_verification.md",
                "decision.md",
                "report.md",
                "run_meta.json",
            ):
                p = run_dir / name
                if p.exists():
                    return p
    return None


# ── Rollback initiation (Task #270, Surface C only) ──────────────────────

ROLLBACK_NOT_REVERSIBLE_TOOLTIP = (
    "No rollback script available for this run. "
    "Contact engineering to reverse manually."
)


class RollbackError(Exception):
    """Raised when a rollback request cannot be honored."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _find_run(dataset_id: str, track: str, run_id: str):
    """Return the RunRow for (dataset, track, run_id) or None.

    Applies ledger-derived `rolled_back` status so callers see the
    current state, not the on-disk-artifact state.
    """
    if track == TRACK_COLOR:
        rows = [
            r
            for r in (
                _enrichment_runs_for_dataset(dataset_id)
                + _color_coverage_fix_runs_for_dataset(dataset_id)
            )
            if r.target_field in COLOR_TARGET_FIELDS
        ]
    elif track == TRACK_PATTERN:
        rows = [
            r
            for r in _enrichment_runs_for_dataset(dataset_id)
            if r.target_field in PATTERN_TARGET_FIELDS
        ]
    else:
        return None
    _apply_rollback_status(dataset_id, track, rows)
    for r in rows:
        if r.run_id == run_id:
            return r
    return None


def _find_color_summary(dataset_id: str, run_id: str):
    """Return (summary_dict, run_dir) for a color_coverage_fix run, or None."""
    p = COLOR_COVERAGE_FIX_DIR / run_id / "summary.json"
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text())
    except Exception:
        return None
    if s.get("dataset_id") != dataset_id:
        return None
    return s, p.parent


def _find_pattern_run_meta(dataset_id: str, run_id: str):
    """Return (meta_dict, run_dir) for a pattern enrichment run, or None.

    Mirrors ``_find_color_summary`` for the pattern track. Used by
    ``initiate_rollback`` to look up the per-run ``rollback_command``
    that lives inside this single run's own ``run_meta.json``.
    """
    p = ENRICHMENT_RUNS_DIR / dataset_id / run_id / "run_meta.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text())
    except Exception:
        return None
    target = str(meta.get("coverage_field") or "")
    if target not in PATTERN_TARGET_FIELDS:
        return None
    return meta, p.parent


def _previous_applied_run_id(
    dataset_id: str, track: str, exclude_run_id: str
) -> Optional[str]:
    """Return the most recent prior applied run_id on `track`, or None."""
    payload = get_runs_for_dataset(dataset_id)
    candidates = [
        r
        for r in payload[track]["runs"]
        if r.get("status") == STATUS_APPLIED and r.get("run_id") != exclude_run_id
    ]
    candidates.sort(key=lambda r: r.get("applied_at") or "", reverse=True)
    return candidates[0]["run_id"] if candidates else None


def _default_executor(command: str, cwd: Path) -> int:
    """Default rollback-command runner. Returns process exit code."""
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode


# Module-level executor so tests can monkeypatch it.
ROLLBACK_EXECUTOR = _default_executor


def initiate_rollback(
    dataset_id: str,
    track: str,
    run_id: str,
    *,
    actor: str = "admin",
    reason: str = "",
) -> Dict[str, Any]:
    """Perform a track-scoped rollback and append a ledger event.

    Raises RollbackError on validation failure or executor failure.
    On success returns the recorded event dict.
    """
    if not _is_safe_segment(dataset_id):
        raise RollbackError(400, "invalid dataset_id")
    if track not in ALLOWED_TRACKS:
        raise RollbackError(400, "invalid track")
    if not _is_safe_segment(run_id):
        raise RollbackError(400, "invalid run_id")

    row = _find_run(dataset_id, track, run_id)
    if row is None:
        raise RollbackError(404, "run not found on this track for this dataset")
    if row.status != STATUS_APPLIED:
        raise RollbackError(
            409, f"run is not in 'applied' status (current: {row.status})"
        )
    if row.reversible is not True:
        raise RollbackError(409, ROLLBACK_NOT_REVERSIBLE_TOOLTIP)

    target_field = row.target_field
    cohort_size = row.cohort_size

    # Track-scoped execution. Both branches dispatch through the same
    # ROLLBACK_EXECUTOR injection point (so tests can stub either track),
    # and both branches require a per-run rollback command that lives
    # inside the single run's own artifact:
    #   - color  → summary.json::rollback_command
    #   - pattern → run_meta.json::rollback_command  (Task #274)
    # Pattern runs without that field are filtered out by the
    # ``reversible`` check above and never reach this branch.
    if track == TRACK_COLOR:
        s = _find_color_summary(dataset_id, run_id)
        if s is None:
            raise RollbackError(404, "rollback artifacts missing")
        summary, run_dir = s
        cmd = summary.get("rollback_command")
        if not cmd:
            raise RollbackError(409, ROLLBACK_NOT_REVERSIBLE_TOOLTIP)
        rc = ROLLBACK_EXECUTOR(cmd, run_dir)
        if rc != 0:
            raise RollbackError(500, f"rollback command failed with exit code {rc}")
    elif track == TRACK_PATTERN:
        m = _find_pattern_run_meta(dataset_id, run_id)
        if m is None:
            raise RollbackError(404, "rollback artifacts missing")
        meta, run_dir = m
        cmd = meta.get("rollback_command")
        if not cmd:
            # Defensive: reversible=True implies cmd is present, but if
            # the artifact is mutated between the read above and here
            # we still refuse with the spec-locked tooltip rather than
            # mis-executing.
            raise RollbackError(409, ROLLBACK_NOT_REVERSIBLE_TOOLTIP)
        rc = ROLLBACK_EXECUTOR(cmd, run_dir)
        if rc != 0:
            raise RollbackError(500, f"rollback command failed with exit code {rc}")
    else:
        # Defense in depth — already gated by the ALLOWED_TRACKS check.
        raise RollbackError(409, ROLLBACK_NOT_REVERSIBLE_TOOLTIP)

    restored_to = _previous_applied_run_id(dataset_id, track, run_id)
    event = _append_rollback_event(
        dataset_id=dataset_id,
        track=track,
        target_field=target_field,
        reverted_run_id=run_id,
        restored_to_run_id=restored_to,
        actor=actor or "admin",
        reason=reason or "",
        cohort_size=cohort_size,
    )
    return event


def _append_rollback_event(
    *,
    dataset_id,
    track,
    target_field,
    reverted_run_id,
    restored_to_run_id,
    actor,
    reason,
    cohort_size,
) -> Dict[str, Any]:
    occurred_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    event = {
        "event_id": f"rb_{uuid.uuid4().hex[:12]}",
        "dataset_id": dataset_id,
        "track": track,
        "target_field": target_field,
        "reverted_run_id": reverted_run_id,
        "restored_to_run_id": restored_to_run_id,
        "actor": actor,
        "reason": reason,
        "occurred_at": occurred_at,
        "cohort_size": cohort_size,
    }
    out_dir = ROLLBACK_EVENTS_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{track}.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event
