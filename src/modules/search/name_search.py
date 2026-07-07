"""
Name Search — fuzzy name matching with color confirmation.

Multiplicative scoring model:
  final_name_score = base_identity_score × color_multiplier × material_multiplier

- Identity drives user intent (fuzzy full-string + token rescue)
- Color is a strong visual gate (0.15 penalty for wrong family)
- Material is a soft gate (0.6 penalty only on clear conflict)
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from rapidfuzz import fuzz

from src.utils.color_rules import get_retrieval_color_family
from src.utils.stone_utils import normalize_name

logger = logging.getLogger(__name__)

IDENTITY_BYPASS_THRESHOLD = 92
IDENTITY_BYPASS_COLOR_FLOOR = 0.5

COLOR_MULT_NO_SIGNAL = 0.65
COLOR_MULT_MISMATCH = 0.15
MATERIAL_MULT_MATCH = 1.1
MATERIAL_MULT_CONFLICT = 0.6

COMMERCIAL_STOPWORDS: frozenset = frozenset({
    "premium", "extra", "select", "classic", "super",
    "first", "choice", "natural", "stone", "slab", "tile",
    "polished", "honed", "leathered", "brushed",
    "cm", "mm", "the", "a", "an", "x",
})

COLOR_LEXICON: Dict[str, str] = {
    "red": "red", "rosso": "red", "rojo": "red", "rouge": "red",
    "rot": "red", "ruby": "red", "rubino": "red",
    "green": "green", "verde": "green", "vert": "green",
    "gruen": "green", "emerald": "green", "esmeralda": "green",
    "black": "black", "nero": "black", "noir": "black", "negro": "black",
    "white": "white", "bianco": "white", "blanco": "white", "blanc": "white",
    "grey": "grey", "gray": "grey", "grigio": "grey", "gris": "grey",
    "gold": "gold", "oro": "gold", "golden": "gold", "giallo": "gold",
    "dorado": "gold",
    "beige": "beige", "crema": "beige", "cream": "beige",
    "ivory": "beige", "avorio": "beige",
    "brown": "brown", "marrone": "brown", "cafe": "brown",
    "coffee": "brown", "chocolate": "brown",
    "blue": "blue", "blu": "blue", "azul": "blue",
    "pink": "pink", "rosa": "pink", "rose": "pink",
}

MATERIAL_SET: frozenset = frozenset({
    "marble", "travertine", "onyx", "granite", "limestone",
    "quartzite", "dolomite", "basalt", "slate", "soapstone",
    "terrazzo", "sandstone", "porphyry", "porcelain",
})

COLOR_TO_FAMILIES: Dict[str, Set[str]] = {
    # Task #363 — WARM_CHROMATIC split into RED, PINK, GOLD.  Each colour
    # token now reaches only the sub-family it actually belongs to so the
    # name-based colour confirmation gate cannot bridge red anchors to
    # yellow/pink stones (and vice versa) via the legacy lump bucket.
    "red":   {"RED", "DARK_WARM_CHROMATIC"},
    "pink":  {"PINK"},
    # COOL_CHROMATIC lightness split (Option A) — green tokens reach
    # both LIGHT_COOL (light Green Onyx) and DARK_COOL (mid/dark/missing-
    # lightness green stones) plus DARK_COOL_CHROMATIC for the dark-base
    # chromatic path, so name-based search keeps finding every green
    # stone regardless of which side of the split it lands on.
    "green": {"LIGHT_COOL", "DARK_COOL", "DARK_COOL_CHROMATIC"},
    "black": {"TRUE_DARK"},
    # Task #346 — LIGHT family split into WHITE and BEIGE.  "grey" no
    # longer reaches into LIGHT (the COOL_NEUTRAL ↔ LIGHT adjacency was
    # already removed in #324) and "beige" no longer reaches into WHITE
    # (the new WHITE ↔ BEIGE hard veto in similar_explorer makes that
    # mapping internally inconsistent).
    "white": {"WHITE"},
    "grey":  {"COOL_NEUTRAL"},
    "brown": {"WARM_NEUTRAL", "DARK_WARM_CHROMATIC"},
    # "gold" keeps the WARM_NEUTRAL link (golden-brown stones in the
    # active corpus) and now also reaches the new GOLD sub-family.
    "gold":  {"WARM_NEUTRAL", "GOLD"},
    "beige": {"BEIGE", "WARM_NEUTRAL"},
    "blue":  {"LIGHT_COOL", "DARK_COOL"},
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_text = ascii_text.lower()
    ascii_text = _PUNCT_RE.sub(" ", ascii_text)
    ascii_text = _MULTI_SPACE_RE.sub(" ", ascii_text).strip()
    words = ascii_text.split()
    words = [w for w in words if w not in COMMERCIAL_STOPWORDS and not w.isdigit()]
    return " ".join(words)


@dataclass
class ParsedQuery:
    raw: str
    normalized: str
    color_tokens: List[str] = field(default_factory=list)
    material_tokens: List[str] = field(default_factory=list)
    identity_tokens: List[str] = field(default_factory=list)
    canonical_color: Optional[str] = None
    canonical_material: Optional[str] = None


def parse_query(text: str) -> ParsedQuery:
    normalized = normalize_query(text)
    tokens = normalized.split()

    color_tokens: List[str] = []
    material_tokens: List[str] = []
    identity_tokens: List[str] = []
    canonical_color: Optional[str] = None
    canonical_material: Optional[str] = None

    for tok in tokens:
        if tok in COLOR_LEXICON:
            color_tokens.append(tok)
            if canonical_color is None:
                canonical_color = COLOR_LEXICON[tok]
        elif tok in MATERIAL_SET:
            material_tokens.append(tok)
            if canonical_material is None:
                canonical_material = tok
        else:
            identity_tokens.append(tok)

    return ParsedQuery(
        raw=text,
        normalized=normalized,
        color_tokens=color_tokens,
        material_tokens=material_tokens,
        identity_tokens=identity_tokens,
        canonical_color=canonical_color,
        canonical_material=canonical_material,
    )


def _compute_base_identity_score(pq: ParsedQuery, norm_stone_name: str) -> float:
    if not pq.normalized or not norm_stone_name:
        return 0.0

    wratio = fuzz.WRatio(pq.normalized, norm_stone_name) / 100.0

    if not pq.identity_tokens:
        return wratio

    best_partial = 0.0
    for tok in pq.identity_tokens:
        if len(tok) < 4:
            if tok in norm_stone_name.split():
                best_partial = max(best_partial, 100.0)
        else:
            score = fuzz.partial_ratio(tok, norm_stone_name)
            best_partial = max(best_partial, score)
    best_partial /= 100.0

    return 0.70 * wratio + 0.30 * best_partial


def _compute_color_multiplier(
    pq: ParsedQuery,
    stone: Dict,
    wratio_raw: float,
) -> float:
    if not pq.canonical_color:
        return 1.0

    expected = COLOR_TO_FAMILIES.get(pq.canonical_color)
    if not expected:
        return 1.0

    stone_family = get_retrieval_color_family(stone)

    if stone_family is None:
        mult = COLOR_MULT_NO_SIGNAL
    elif stone_family in expected:
        mult = 1.0
    else:
        mult = COLOR_MULT_MISMATCH

    if wratio_raw >= IDENTITY_BYPASS_THRESHOLD and mult < IDENTITY_BYPASS_COLOR_FLOOR:
        mult = IDENTITY_BYPASS_COLOR_FLOOR

    return mult


def _compute_material_multiplier_cached(pq: ParsedQuery, stone_material: Optional[str]) -> float:
    """Same logic as _compute_material_multiplier but accepts pre-extracted stone_material."""
    if not pq.canonical_material:
        return 1.0
    if stone_material is None:
        return 1.0
    if stone_material == pq.canonical_material:
        return MATERIAL_MULT_MATCH
    return MATERIAL_MULT_CONFLICT


def _precompute_stone_tokens(stone: Dict) -> Dict:
    """Compute and cache stone-derived preprocessing on the stone dict.

    Stored under stone["_ns_tokens"] so it is computed at most once per
    hydration cycle.  The stone dict is rebuilt fresh on each
    _load_all_stone_data_batched() call, so stale tokens are never an issue.
    """
    stone_name = stone.get("stone_name") or ""
    norm_stone = normalize_name(stone_name)
    norm_stone_tokens = set(norm_stone.split())

    # material token: check canonical_family first, then stone name tokens
    canonical_family = (stone.get("canonical_family") or "").strip().lower()
    stone_material: Optional[str] = None
    if canonical_family in MATERIAL_SET:
        stone_material = canonical_family
    else:
        for tok in norm_stone_tokens:
            if tok in MATERIAL_SET:
                stone_material = tok
                break

    return {
        "norm_stone": norm_stone,
        "stone_id_lower": (stone.get("company_stone_id") or "").lower(),
        "stone_material": stone_material,
    }


def score_name_match(query: str, stone: Dict, debug: bool = False) -> float:
    if not query:
        return 0.5

    pq = parse_query(query)
    if not pq.normalized:
        return 0.5

    # Cache per-stone preprocessing — computed once per hydration cycle.
    if "_ns_tokens" not in stone:
        stone["_ns_tokens"] = _precompute_stone_tokens(stone)
    cached = stone["_ns_tokens"]

    norm_stone = cached["norm_stone"]
    stone_id = cached["stone_id_lower"]

    if pq.normalized and pq.normalized in stone_id:
        base_score = 1.0
    else:
        base_score = _compute_base_identity_score(pq, norm_stone)

    wratio_raw = fuzz.WRatio(pq.normalized, norm_stone)
    color_mult = _compute_color_multiplier(pq, stone, wratio_raw)
    material_mult = _compute_material_multiplier_cached(pq, cached["stone_material"])

    final = base_score * color_mult * material_mult

    if debug:
        stone["_name_search_debug"] = {
            "query_parsed": {
                "normalized": pq.normalized,
                "color_tokens": pq.color_tokens,
                "material_tokens": pq.material_tokens,
                "identity_tokens": pq.identity_tokens,
                "canonical_color": pq.canonical_color,
                "canonical_material": pq.canonical_material,
            },
            "base_identity_score": round(base_score, 4),
            "color_multiplier": round(color_mult, 4),
            "material_multiplier": round(material_mult, 4),
            "final_name_score": round(final, 4),
        }

    return final


def log_parsed_query(query: str) -> ParsedQuery:
    pq = parse_query(query)
    logger.info(
        "name_search query=%r normalized=%r color=%s material=%s identity=%s",
        pq.raw, pq.normalized,
        pq.canonical_color, pq.canonical_material,
        pq.identity_tokens,
    )
    return pq
