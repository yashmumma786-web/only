"""LOW-band display suppression experiment (Task #292).

Display-only filter applied during the color resolver's stock_tags_ai
merge step. When the experiment flag is "on", AI rows that satisfy ALL
of the following are skipped at display time only:

  1. The stone's vendor exactly matches one of the cohort allow-list
     vendor strings (byte-for-byte against `dataset_images.vendor` /
     `color_correction_context.vendor_at_correction`).
  2. The AI row's confidence falls in the LOW band (None or
     < `low_band_confidence_threshold`, default 0.40 — the same
     threshold compute_band() uses for LOW vs MID).
  3. The tag_name is one of the color-relevant tag names listed in
     `scoped_tag_names` (default: main_color, bg_color, base_color).
     Out-of-scope tag names are never suppressed by this experiment.

The experiment writes nothing. It only filters at the display projection.
Stored DB values (stock_tags_ai, stock_aggregated, stock_overrides) are
unchanged. correction_log / color_correction_context grow as usual via
status-quo write paths.

Config lives in `config/experiments.json` under the
`low_band_display_suppression` key. Defaults are safe (flag = "off"):
loading config without the file present, or with a malformed entry, or
with the flag missing, must resolve to "do nothing" (no rows skipped).

The module is hot-reloadable via mtime check on the config file so
operators can flip the flag without a server restart; the resolver's
own request-level cache (~5 min TTL per the decision pack) bounds the
effective rollover window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional, Tuple
import json
import threading

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "experiments.json"
CONFIG_KEY = "low_band_display_suppression"

DEFAULT_LOW_BAND_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_SCOPED_TAG_NAMES: Tuple[str, ...] = (
    "main_color",
    "bg_color",
    "base_color",
)


@dataclass(frozen=True)
class LowBandSuppressionConfig:
    """Resolved, immutable view of the experiment config."""

    flag_on: bool
    cohort: FrozenSet[str]
    scoped_tag_names: FrozenSet[str]
    low_band_confidence_threshold: float
    raw_cohort_list: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_active(self) -> bool:
        """True iff the flag is on AND there is at least one cohort vendor.

        An on-flag with an empty cohort allow-list is treated as inert
        (defensive: avoids accidental global suppression).
        """
        return bool(self.flag_on and self.cohort)


_CACHE_LOCK = threading.Lock()
_CACHE: Optional[LowBandSuppressionConfig] = None
_CACHE_MTIME: Optional[float] = None


def _safe_read_config() -> dict:
    try:
        if not CONFIG_PATH.exists():
            return {}
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_config(raw: dict) -> LowBandSuppressionConfig:
    section = {}
    if isinstance(raw, dict):
        candidate = raw.get(CONFIG_KEY)
        if isinstance(candidate, dict):
            section = candidate

    flag_raw = section.get("flag", "off")
    flag_on = isinstance(flag_raw, str) and flag_raw.strip().lower() == "on"

    cohort_raw = section.get("cohort", [])
    if isinstance(cohort_raw, list):
        cohort_list = tuple(str(v) for v in cohort_raw if isinstance(v, str) and v)
    else:
        cohort_list = tuple()
    cohort = frozenset(cohort_list)

    scoped_raw = section.get("scoped_tag_names")
    if isinstance(scoped_raw, list) and scoped_raw:
        scoped = frozenset(
            str(v) for v in scoped_raw if isinstance(v, str) and v
        )
    else:
        scoped = frozenset(DEFAULT_SCOPED_TAG_NAMES)

    threshold_raw = section.get(
        "low_band_confidence_threshold", DEFAULT_LOW_BAND_CONFIDENCE_THRESHOLD
    )
    try:
        threshold = float(threshold_raw)
    except Exception:
        threshold = DEFAULT_LOW_BAND_CONFIDENCE_THRESHOLD

    return LowBandSuppressionConfig(
        flag_on=flag_on,
        cohort=cohort,
        scoped_tag_names=scoped,
        low_band_confidence_threshold=threshold,
        raw_cohort_list=cohort_list,
    )


def get_config(force_reload: bool = False) -> LowBandSuppressionConfig:
    """Return the current resolved config.

    Cached, with mtime-based invalidation so a config edit takes effect
    on the next call without restarting the server. Pass ``force_reload``
    to bypass the mtime check (used by tests and the verification script).
    """
    global _CACHE, _CACHE_MTIME
    try:
        mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else None
    except Exception:
        mtime = None

    with _CACHE_LOCK:
        if (
            not force_reload
            and _CACHE is not None
            and _CACHE_MTIME == mtime
        ):
            return _CACHE
        cfg = _build_config(_safe_read_config())
        _CACHE = cfg
        _CACHE_MTIME = mtime
        return cfg


def is_low_band_confidence(
    confidence: Optional[float], threshold: Optional[float] = None
) -> bool:
    """Faithful proxy for compute_band() == 'LOW' at display time.

    The resolver's display projection does not have ``image_count`` or
    ``has_truth`` available alongside an AI row. compute_band() returns
    LOW when ``ai_main_conf is None`` OR when ``ai_main_conf < 0.40``,
    so checking the confidence value reproduces the LOW classification
    for the dominant case the experiment targets ("AI low evidence",
    73 of 85 LOW-band corrections per Task #291's report).

    image_count == 0 stones also resolve to LOW under compute_band but
    those typically lack an AI row in stock_tags_ai at all (no image to
    classify), so they are not the target of this filter.
    """
    if threshold is None:
        threshold = DEFAULT_LOW_BAND_CONFIDENCE_THRESHOLD
    if confidence is None:
        return True
    try:
        return float(confidence) < threshold
    except Exception:
        return True


def should_suppress_ai_row(
    *,
    vendor_name: Optional[str],
    tag_name: Optional[str],
    confidence: Optional[float],
    config: Optional[LowBandSuppressionConfig] = None,
) -> bool:
    """Conjunctive filter — all three conditions must hold.

    Returns True iff this AI row should be skipped at display projection
    time. False otherwise (the row passes through unchanged). Safe to
    call on every AI row in the hot path: the fast-path early-exits when
    the experiment is inactive.
    """
    cfg = config if config is not None else get_config()
    if not cfg.is_active:
        return False
    if not vendor_name or vendor_name not in cfg.cohort:
        return False
    if not tag_name or tag_name not in cfg.scoped_tag_names:
        return False
    return is_low_band_confidence(confidence, cfg.low_band_confidence_threshold)
