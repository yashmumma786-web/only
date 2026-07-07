"""
Business rules and thresholds for stone color and lightness classification.
Moved from ml_inference/color_agent.py as part of the Phase 4 DDD refactor.
"""

from typing import Optional

# ── Thresholds (Stonestock business decisions, not ML math) ──
STONE_LIGHTNESS_LIGHT_MIN = 65.0
STONE_LIGHTNESS_DARK_MAX = 35.0
STONE_LIGHTNESS_DARK_MAX_DARK_BASE = 50.0
DARK_BASE_COLORS = frozenset({"black", "charcoal", "off-black", "dark grey", "dark brown", "navy"})

def classify_lightness_band(mean_L: float, base_color_hint: Optional[str] = None) -> str:
    """Stonestock rule: map mean L* → 'light' / 'mid' / 'dark'."""
    dark_max = STONE_LIGHTNESS_DARK_MAX_DARK_BASE if (
        base_color_hint and base_color_hint.lower() in DARK_BASE_COLORS
    ) else STONE_LIGHTNESS_DARK_MAX
    if mean_L >= STONE_LIGHTNESS_LIGHT_MIN: return "light"
    if mean_L < dark_max: return "dark"
    return "mid"
