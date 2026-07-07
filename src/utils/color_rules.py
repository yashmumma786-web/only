from typing import Optional

# ---------------------------------------------------------------------------
# Retrieval color-family classification (moved from search/similar_explorer.py)
#
# These tables and functions are pure color math — they read only a stone's
# display_tags dict (in-memory) and perform zero I/O.  They live here in
# taxonomy because they express color-classification business rules, not
# search or similarity logic.  Both search (name_search) and similar
# (explorer) consume them via taxonomy.services.get_retrieval_color_family().
# ---------------------------------------------------------------------------

_COLOR_VETO_BUCKET: dict = {
    "black": "dark", "ebony": "dark", "charcoal": "dark", "anthracite": "dark",
    "white": "light", "cream": "light", "beige": "light", "ivory": "light",
    "off-white": "light", "offwhite": "light", "light grey": "light", "light gray": "light",
    "pale": "light",
    "red": "red", "rose": "red", "pink": "red", "crimson": "red",
    "green": "green", "olive": "green", "sage": "green", "forest": "green", "emerald": "green",
    "blue": "blue", "navy": "blue", "cobalt": "blue", "royal blue": "blue", "sapphire": "blue",
}

_VETO_PAIRS: frozenset = frozenset([
    frozenset({"dark", "light"}),
    frozenset({"red", "light"}),
    frozenset({"red", "green"}),
    frozenset({"red", "blue"}),
])

_DARK_MAIN_COLORS: frozenset = frozenset({
    "black", "charcoal", "ebony", "anthracite", "dark grey", "dark gray"
})

_DARK_WARM_HUES: frozenset = frozenset({"red", "orange"})
_DARK_COOL_HUES: frozenset = frozenset({"green"})

_COLOR_NORMALIZE: dict = {"rose": "pink"}

_COLOR_FAMILY: dict = {
    # WARM_NEUTRAL — brown, tan, earthy warm tones
    "brown": "WARM_NEUTRAL", "tan": "WARM_NEUTRAL", "taupe": "WARM_NEUTRAL",
    "sand": "WARM_NEUTRAL", "terracotta": "WARM_NEUTRAL",
    "caramel": "WARM_NEUTRAL", "walnut": "WARM_NEUTRAL", "chocolate": "WARM_NEUTRAL",
    "coffee": "WARM_NEUTRAL", "toffee": "WARM_NEUTRAL",
    # COOL_NEUTRAL — grey tones
    "grey": "COOL_NEUTRAL", "gray": "COOL_NEUTRAL", "silver": "COOL_NEUTRAL",
    "slate": "COOL_NEUTRAL", "ash": "COOL_NEUTRAL",
    # WHITE — pure white / near-white / cool off-whites
    "white": "WHITE", "ivory": "WHITE", "cream": "WHITE",
    "off-white": "WHITE", "offwhite": "WHITE", "pale": "WHITE",
    "light grey": "WHITE", "light gray": "WHITE",
    # BEIGE — warm off-whites
    "beige": "BEIGE", "champagne": "BEIGE",
    "light tan": "BEIGE", "warm white": "BEIGE",
    # DARK — black and very dark tones
    "black": "DARK", "charcoal": "DARK", "ebony": "DARK", "anthracite": "DARK",
    "dark grey": "DARK", "dark gray": "DARK",
    # RED
    "red": "RED", "crimson": "RED", "burgundy": "RED", "scarlet": "RED",
    "ruby": "RED", "maroon": "RED", "coral": "RED", "salmon": "RED",
    # PINK
    "pink": "PINK", "rose": "PINK", "blush": "PINK", "fuchsia": "PINK",
    # GOLD
    "orange": "GOLD", "gold": "GOLD", "yellow": "GOLD", "amber": "GOLD",
    "copper": "GOLD", "bronze": "GOLD", "terra cotta": "GOLD", "ochre": "GOLD",
    # COOL_CHROMATIC — blue, green, teal family
    "blue": "COOL_CHROMATIC", "navy": "COOL_CHROMATIC", "cobalt": "COOL_CHROMATIC",
    "royal blue": "COOL_CHROMATIC", "sapphire": "COOL_CHROMATIC",
    "green": "COOL_CHROMATIC", "olive": "COOL_CHROMATIC", "sage": "COOL_CHROMATIC",
    "forest": "COOL_CHROMATIC", "emerald": "COOL_CHROMATIC", "teal": "COOL_CHROMATIC",
}

# Tier-2 adjacency pairs — buyer-accepted neighbouring families.
_ADJACENT_FAMILIES: frozenset = frozenset([
    frozenset({"WARM_NEUTRAL",        "BEIGE"}),
    frozenset({"COOL_NEUTRAL",        "TRUE_DARK"}),
    frozenset({"DARK_COOL_CHROMATIC", "COOL_CHROMATIC"}),
    frozenset({"DARK_COOL_CHROMATIC", "COOL_NEUTRAL"}),
    frozenset({"RED",  "DARK_WARM_CHROMATIC"}),
    frozenset({"RED",  "TRUE_DARK"}),
    frozenset({"RED",  "WARM_NEUTRAL"}),
    frozenset({"PINK", "WHITE"}),
    frozenset({"GOLD", "WARM_NEUTRAL"}),
    frozenset({"GOLD", "TRUE_DARK"}),
    frozenset({"LIGHT_COOL",          "DARK_COOL"}),
    frozenset({"DARK_COOL_CHROMATIC", "LIGHT_COOL"}),
    frozenset({"DARK_COOL_CHROMATIC", "DARK_COOL"}),
])

# Hard veto pairs — families that must never appear as similar candidates.
_RETRIEVAL_FAMILY_VETO_PAIRS: frozenset = frozenset([
    frozenset({"TRUE_DARK",           "WHITE"}),
    frozenset({"TRUE_DARK",           "BEIGE"}),
    frozenset({"TRUE_DARK",           "COOL_CHROMATIC"}),
    frozenset({"DARK_WARM_CHROMATIC", "WHITE"}),
    frozenset({"DARK_WARM_CHROMATIC", "BEIGE"}),
    frozenset({"DARK_COOL_CHROMATIC", "WHITE"}),
    frozenset({"DARK_COOL_CHROMATIC", "BEIGE"}),
    frozenset({"DARK_WARM_CHROMATIC", "COOL_CHROMATIC"}),
    frozenset({"TRUE_DARK",           "RED"}),
    frozenset({"TRUE_DARK",           "PINK"}),
    frozenset({"TRUE_DARK",           "GOLD"}),
    frozenset({"DARK_COOL_CHROMATIC", "RED"}),
    frozenset({"DARK_COOL_CHROMATIC", "PINK"}),
    frozenset({"DARK_COOL_CHROMATIC", "GOLD"}),
    frozenset({"RED",   "COOL_CHROMATIC"}),
    frozenset({"PINK",  "COOL_CHROMATIC"}),
    frozenset({"GOLD",  "COOL_CHROMATIC"}),
    frozenset({"RED",   "WHITE"}),
    frozenset({"PINK",  "WHITE"}),
    frozenset({"GOLD",  "WHITE"}),
    frozenset({"RED",   "BEIGE"}),
    frozenset({"PINK",  "BEIGE"}),
    frozenset({"GOLD",  "BEIGE"}),
    frozenset({"RED",   "DARK_WARM_CHROMATIC"}),
    frozenset({"PINK",  "DARK_WARM_CHROMATIC"}),
    frozenset({"GOLD",  "DARK_WARM_CHROMATIC"}),
    frozenset({"RED",   "COOL_NEUTRAL"}),
    frozenset({"PINK",  "COOL_NEUTRAL"}),
    frozenset({"GOLD",  "COOL_NEUTRAL"}),
    frozenset({"RED",   "PINK"}),
    frozenset({"RED",   "GOLD"}),
    frozenset({"PINK",  "GOLD"}),
    frozenset({"WHITE",               "COOL_CHROMATIC"}),
    frozenset({"BEIGE",               "COOL_CHROMATIC"}),
    frozenset({"WHITE",               "COOL_NEUTRAL"}),
    frozenset({"BEIGE",               "COOL_NEUTRAL"}),
    frozenset({"WHITE",               "BEIGE"}),
    frozenset({"WARM_NEUTRAL",        "COOL_NEUTRAL"}),
    frozenset({"DARK_WARM_CHROMATIC", "WARM_NEUTRAL"}),
    frozenset({"TRUE_DARK",           "WARM_NEUTRAL"}),
    frozenset({"TRUE_DARK",           "LIGHT_COOL"}),
    frozenset({"TRUE_DARK",           "DARK_COOL"}),
    frozenset({"DARK_WARM_CHROMATIC", "LIGHT_COOL"}),
    frozenset({"DARK_WARM_CHROMATIC", "DARK_COOL"}),
    frozenset({"RED",   "LIGHT_COOL"}),
    frozenset({"RED",   "DARK_COOL"}),
    frozenset({"PINK",  "LIGHT_COOL"}),
    frozenset({"PINK",  "DARK_COOL"}),
    frozenset({"GOLD",  "LIGHT_COOL"}),
    frozenset({"GOLD",  "DARK_COOL"}),
    frozenset({"WHITE",               "LIGHT_COOL"}),
    frozenset({"WHITE",               "DARK_COOL"}),
    frozenset({"BEIGE",               "LIGHT_COOL"}),
    frozenset({"BEIGE",               "DARK_COOL"}),
])

LIGHT_COOL_DISTANCE_PAIRS: frozenset = frozenset([
    frozenset({"LIGHT_COOL", "DARK_COOL"}),
    frozenset({"LIGHT_COOL", "DARK_COOL_CHROMATIC"}),
])
LIGHT_COOL_DISTANCE_THRESHOLD: float = 20.0


def _get_retrieval_tag_value(stone: dict, key: str, default=None):
    """Read display_tags[key][value] from a stone dict (pure, no I/O)."""
    tags = stone.get("display_tags", {})
    entry = tags.get(key)
    if entry is None:
        return default
    if isinstance(entry, dict):
        return entry.get("value", default)
    return entry


def _get_stone_veto_color_from_tags(stone: dict) -> Optional[str]:
    """Read base_color / main_color from display_tags only (no DB calls).
    Used by get_retrieval_color_family when no DB-backed override is needed.
    The full DB-backed chain (override → aggregated → ai) lives in
    similar/explorer.py and is used by the resolver at runtime.
    """
    color = (
        _get_retrieval_tag_value(stone, "base_color")
        or _get_retrieval_tag_value(stone, "main_color")
    )
    if not color or not isinstance(color, str):
        return None
    color = color.strip().lower()
    color = _COLOR_NORMALIZE.get(color, color)
    if color in ("multi", "multicolor"):
        return None
    return color if color else None


def get_retrieval_color_family(stone: dict) -> Optional[str]:
    """Composite color-family classifier for Similar tier assignment.

    Reads only ``stone["display_tags"]`` — zero I/O, pure function.
    Used by:
      - ``search/name_search.py`` (color confirmation gate in name search)
      - ``similar/explorer.py``   (veto / tier logic in the resolver)

    Returns one of the known family strings (e.g. ``"TRUE_DARK"``,
    ``"WARM_NEUTRAL"``, ``"WHITE"``, ``"LIGHT_COOL"``, etc.) or ``None``
    when the stone has no usable color signal.

    COOL_CHROMATIC lightness split (Option A): the pre-split COOL_CHROMATIC
    family is partitioned at runtime into LIGHT_COOL (stone_lightness="light")
    and DARK_COOL (mid / dark / missing — safe default).
    """
    main_color = _get_stone_veto_color_from_tags(stone)
    dominant_hue = _get_retrieval_tag_value(stone, "dominant_hue") or ""
    dominant_hue = dominant_hue.strip().lower() if dominant_hue else ""

    if main_color and main_color in _DARK_MAIN_COLORS:
        if dominant_hue in _DARK_COOL_HUES:
            return "DARK_COOL_CHROMATIC"
        if dominant_hue in _DARK_WARM_HUES:
            return "DARK_WARM_CHROMATIC"
        return "TRUE_DARK"

    family = _COLOR_FAMILY.get(main_color) if main_color else None
    if family == "COOL_CHROMATIC":
        lightness = _get_retrieval_tag_value(stone, "stone_lightness") or ""
        lightness = lightness.strip().lower() if lightness else ""
        if lightness == "light":
            return "LIGHT_COOL"
        return "DARK_COOL"
    return family
