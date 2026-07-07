"""
Computer Vision Analyzer for Fingerprints V3

Per-image analysis including:
- Stone mask segmentation with inner_bbox
- Glare and reflection mask detection
- Quality metrics (blur, exposure, glare, reflection)
- Pattern/vein/spot mask detection using delta-E
- Design classification excluding glare/reflection
- Neutral-first base color detection with chroma gating
- Exotic detection using chroma-based delta_ab
"""

import cv2
import hashlib
import json
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans

_CV_ANALYZER_VERSION = "v3.8"

import os
_ANALYSIS_CACHE_DIR = Path(os.environ.get("ANALYSIS_CACHE_DIR"))
_ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(x) for x in obj]
    return obj


# ============== THRESHOLD CONSTANTS ==============
C_NEUTRAL = 8.0  # Chroma threshold for neutral colors
L_WHITE = 75.0  # L* threshold for white (lowered to catch underexposed white slabs)
L_CHARCOAL = 42.0  # L* threshold for charcoal/black
B_BEIGE = 8.0  # b* warmth threshold for beige
INNER_BBOX_SHRINK = 0.10  # Shrink bbox by 10% each side
GLARE_LUM_THRESH = 0.92  # Luminance threshold for glare (normalized 0..1)
GLARE_SAT_THRESH = 0.08  # Saturation threshold for glare
REFLECTION_L_OFFSET = 12  # L* offset above median for reflection
REFLECTION_SAT_THRESH = 0.12
VEIN_DELTA_E_THRESH = 10.0  # Delta-E threshold for vein detection
VEIN_L_WEIGHT = 0.35  # Weight for L* in delta-E (lighting robust)
EXOTIC_SECOND_PROP = 0.30  # Min proportion for second cluster
EXOTIC_DELTA_AB = 10.0  # Min chroma separation for exotic
EXOTIC_GRID_CELLS = 6  # Min cells (of 16) for spatial significance
MIN_PIXELS = 100  # Minimum pixels for valid mask
DARK_STONE_L_THRESH = 50  # L* threshold for dark stone sampling
DARK_QUANTILE = 0.30  # Use lowest 30% L* for dark stones


@dataclass
class StoneMaskResult:
    """Result of stone mask segmentation."""

    mask: np.ndarray
    stone_area_ratio: float
    bbox: Tuple[int, int, int, int]
    inner_bbox: Tuple[int, int, int, int]
    stone_area: int


@dataclass
class MaskSet:
    """Collection of exclusion masks."""

    glare_mask: np.ndarray
    reflection_mask: np.ndarray
    glare_ratio: float
    reflection_ratio: float


@dataclass
class QualityMetrics:
    """Image quality metrics for weighting."""

    blur_score: float
    exposure_score: float
    glare_score: float
    glare_ratio: float
    reflection_ratio: float
    stone_area_ratio: float
    image_weight: float


@dataclass
class PatternMasks:
    """Pattern detection results."""

    pattern_mask: np.ndarray
    vein_mask: np.ndarray
    spot_mask: np.ndarray
    pattern_coverage: float
    vein_coverage: float
    spot_coverage: float
    low_freq_energy: float
    spot_blob_count: int


@dataclass
class VeinAnalysis:
    """Vein-specific analysis for Design=Veiny."""

    direction: str
    direction_conf: float
    distribution: str
    distribution_conf: float
    thickness: str
    thickness_conf: float
    colors: List[str]
    colors_conf: float
    dominant_angle: float
    coherence: float
    entropy: float
    rel_width: float
    coverage: float


@dataclass
class SpotAnalysis:
    """Spot-specific analysis for Design=Spotty."""

    distribution: str
    distribution_conf: float
    colors: List[str]
    colors_conf: float
    coverage: float
    cv: float


@dataclass
class ColorAnalysis:
    """Color analysis results."""

    base_color: str
    base_color_conf: float
    is_exotic: bool
    colors_in_stone: List[str]
    colors_in_stone_conf: float
    tonality: str
    tonality_conf: float
    luminance_mean: float
    luminance_var: float
    chroma: float
    cluster_proportions: List[float]
    cluster_distances: List[float]
    debug_info: Dict


# COLOR_CENTROIDS_LAB uses proper L*a*b* scale:
# L* = 0..100, a* = -128..+127, b* = -128..+127
COLOR_CENTROIDS_LAB = {
    "white": (92, 0, 0),
    "grey": (60, 0, 0),
    "black": (15, 0, 0),
    "beige": (80, 5, 15),
    "brown": (45, 10, 20),
    "yellow": (80, -5, 50),
    "red": (45, 45, 20),
    "rose": (70, 25, 10),
    "pink": (75, 20, 5),
    "green": (45, -25, 15),
}

NEUTRAL_COLORS = {"white", "grey", "black", "beige"}

NEUTRAL_CENTROIDS_LAB = {
    k: v
    for k, v in COLOR_CENTROIDS_LAB.items()
    if k in {"white", "grey", "beige", "brown", "black"}
}


def _nearest_neutral_centroid(lab_star: np.ndarray) -> str:
    input_arr = np.array([float(lab_star[0]), float(lab_star[1]), float(lab_star[2])])
    min_dist = float("inf")
    best = "grey"
    for name, centroid in NEUTRAL_CENTROIDS_LAB.items():
        dist = np.sqrt(np.sum((input_arr - np.array(centroid)) ** 2))
        if dist < min_dist:
            min_dist = dist
            best = name
    return best


def _opencv_lab_pixels_to_lab_star(opencv_lab_pixels: np.ndarray) -> np.ndarray:
    """
    Convert array of OpenCV LAB pixels to L*a*b* scale.
    Input: (N, 3) array of uint8/float32 OpenCV LAB
    Output: (N, 3) array of float32 L*a*b*
    """
    result = opencv_lab_pixels.astype(np.float32).copy()
    result[:, 0] = result[:, 0] * (100.0 / 255.0)  # L -> L*
    result[:, 1] = result[:, 1] - 128.0  # a -> a*
    result[:, 2] = result[:, 2] - 128.0  # b -> b*
    return result


def segment_stone_mask(image: np.ndarray) -> StoneMaskResult:
    """
    Segment the stone surface from background/stickers/floor.
    Returns mask with bbox and inner_bbox (shrunk for safe sampling).
    """
    h, w = image.shape[:2]
    total_area = h * w

    # The user specifically requested to use a simple center clip
    # to guarantee we only extract color from the center of the image,
    # safely avoiding all edges, factory walls, or stickers.
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)

    final_mask = np.zeros((h, w), dtype=np.uint8)
    final_mask[margin_y : h - margin_y, margin_x : w - margin_x] = 255

    stone_area = np.sum(final_mask > 0)
    stone_area_ratio = stone_area / total_area
    bbox = (margin_x, margin_y, w - margin_x, h - margin_y)

    bx1, by1, bx2, by2 = bbox
    bw, bh = bx2 - bx1, by2 - by1
    shrink_x = int(bw * INNER_BBOX_SHRINK)
    shrink_y = int(bh * INNER_BBOX_SHRINK)
    inner_bbox = (bx1 + shrink_x, by1 + shrink_y, bx2 - shrink_x, by2 - shrink_y)

    return StoneMaskResult(
        mask=final_mask,
        stone_area_ratio=stone_area_ratio,
        bbox=bbox,
        inner_bbox=inner_bbox,
        stone_area=stone_area,
    )


def compute_glare_and_reflection_masks(
    image: np.ndarray, stone_mask: np.ndarray, inner_bbox: Tuple[int, int, int, int]
) -> MaskSet:
    """
    Compute glare_mask and reflection_mask separately.
    Glare = very bright + low saturation
    Reflection = moderately bright above median + low saturation + high local variance
    """
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].astype(np.float32) * 2.0  # Convert to 0..360 degrees
    s_channel = hsv[:, :, 1].astype(np.float32) / 255.0  # Normalize to 0..1
    v_channel = hsv[:, :, 2].astype(np.float32) / 255.0  # Normalize to 0..1

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)  # L* is 0..255 in OpenCV

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    bx1, by1, bx2, by2 = inner_bbox
    inner_mask[by1:by2, bx1:bx2] = 255
    inner_mask = cv2.bitwise_and(inner_mask, stone_mask)

    glare_mask = (
        (v_channel > GLARE_LUM_THRESH)
        & (s_channel < GLARE_SAT_THRESH)
        & (stone_mask > 0)
    ).astype(np.uint8) * 255

    if np.sum(inner_mask > 0) > MIN_PIXELS:
        median_L = np.median(l_channel[inner_mask > 0])
    else:
        median_L = (
            np.median(l_channel[stone_mask > 0]) if np.sum(stone_mask > 0) > 0 else 128
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    ksize = 9
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f * gray_f, (ksize, ksize))
    local_var = mean_sq - mean * mean
    local_var = np.maximum(local_var, 0)
    var_thresh = (
        np.percentile(local_var[stone_mask > 0], 70)
        if np.sum(stone_mask > 0) > 0
        else 100
    )

    reflection_mask = (
        (l_channel > median_L + REFLECTION_L_OFFSET)
        & (s_channel < REFLECTION_SAT_THRESH)
        & (local_var > var_thresh)
        & (stone_mask > 0)
        & (glare_mask == 0)  # Exclude already-glare pixels
    ).astype(np.uint8) * 255

    stone_area = max(np.sum(stone_mask > 0), 1)
    glare_ratio = np.sum(glare_mask > 0) / stone_area
    reflection_ratio = np.sum(reflection_mask > 0) / stone_area

    return MaskSet(
        glare_mask=glare_mask,
        reflection_mask=reflection_mask,
        glare_ratio=glare_ratio,
        reflection_ratio=reflection_ratio,
    )


def compute_quality_metrics(
    image: np.ndarray, stone_mask: np.ndarray, mask_set: MaskSet
) -> QualityMetrics:
    """Compute image quality metrics for weighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(laplacian_var / 500.0, 1.0)

    if np.sum(stone_mask > 0) > 0:
        stone_pixels = gray[stone_mask > 0]
        mean_lum = np.mean(stone_pixels)
    else:
        mean_lum = np.mean(gray)

    exposure_score = 1.0 - abs(mean_lum - 128) / 128.0
    exposure_score = max(0.0, exposure_score)

    glare_score = 1.0 - min(mask_set.glare_ratio * 2, 1.0)

    stone_area_ratio = np.sum(stone_mask > 0) / (image.shape[0] * image.shape[1])

    image_weight = (
        blur_score * 0.25
        + exposure_score * 0.25
        + glare_score * 0.25
        + min(stone_area_ratio / 0.5, 1.0) * 0.25
    )

    image_weight *= max(1.0 - mask_set.glare_ratio * 1.5, 0.2)
    image_weight *= max(1.0 - mask_set.reflection_ratio * 1.0, 0.3)

    if stone_area_ratio < 0.25:
        image_weight *= 0.3

    return QualityMetrics(
        blur_score=blur_score,
        exposure_score=exposure_score,
        glare_score=glare_score,
        glare_ratio=mask_set.glare_ratio,
        reflection_ratio=mask_set.reflection_ratio,
        stone_area_ratio=stone_area_ratio,
        image_weight=image_weight,
    )
def build_background_mask(
    stone_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
    glare_mask: np.ndarray,
    reflection_mask: np.ndarray,
    vein_mask: np.ndarray,
    spot_mask: np.ndarray,
) -> np.ndarray:
    """
    Build robust background mask for color sampling:
    inner_bbox ∩ stone_mask ∩ NOT(glare) ∩ NOT(reflection) ∩ NOT(vein) ∩ NOT(spot)
    """
    h, w = stone_mask.shape

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    bx1, by1, bx2, by2 = inner_bbox
    inner_mask[by1:by2, bx1:bx2] = 255

    background_mask = cv2.bitwise_and(inner_mask, stone_mask)
    background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(glare_mask))
    background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(reflection_mask))
    background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(vein_mask))
    background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(spot_mask))

    return background_mask


def detect_patterns_delta_e(
    image: np.ndarray,
    stone_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
    glare_mask: np.ndarray,
    reflection_mask: np.ndarray,
) -> PatternMasks:
    """
    Detect pattern, vein, and spot masks using delta-E from background.
    Excludes glare and reflection from detection.
    """
    h, w = image.shape[:2]

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    bx1, by1, bx2, by2 = inner_bbox
    inner_mask[by1:by2, bx1:bx2] = 255

    clean_mask = cv2.bitwise_and(inner_mask, stone_mask)
    clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(glare_mask))
    clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(reflection_mask))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_img, a_img, b_img = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    if np.sum(clean_mask > 0) >= MIN_PIXELS:
        bg_L = np.median(l_img[clean_mask > 0])
        bg_a = np.median(a_img[clean_mask > 0])
        bg_b = np.median(b_img[clean_mask > 0])
    else:
        bg_L, bg_a, bg_b = 128.0, 128.0, 128.0

    delta_e = np.sqrt(
        VEIN_L_WEIGHT * (l_img - bg_L) ** 2 + (a_img - bg_a) ** 2 + (b_img - bg_b) ** 2
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edge_thresh = (
        np.percentile(edges[stone_mask > 0], 85) if np.sum(stone_mask > 0) > 0 else 50
    )
    edge_mask = (edges > edge_thresh).astype(np.uint8) * 255

    vein_candidate = ((delta_e > VEIN_DELTA_E_THRESH) & (clean_mask > 0)).astype(
        np.uint8
    ) * 255

    vein_mask = cv2.bitwise_and(vein_candidate, edge_mask)

    kernel = np.ones((3, 3), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        vein_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    vein_mask_clean = np.zeros_like(vein_mask)
    spot_mask = np.zeros_like(vein_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = max(cw, ch) / (min(cw, ch) + 1e-6)

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)

        if aspect_ratio > 2.5:
            cv2.drawContours(vein_mask_clean, [contour], -1, 255, -1)
        elif circularity > 0.5 and aspect_ratio < 2.0:
            cv2.drawContours(spot_mask, [contour], -1, 255, -1)
        elif aspect_ratio > 1.5:
            cv2.drawContours(vein_mask_clean, [contour], -1, 255, -1)
        else:
            cv2.drawContours(spot_mask, [contour], -1, 255, -1)

    pattern_mask = cv2.bitwise_or(vein_mask_clean, spot_mask)

    stone_area = max(np.sum(stone_mask > 0), 1)
    pattern_coverage = np.sum(pattern_mask > 0) / stone_area
    vein_coverage = np.sum(vein_mask_clean > 0) / stone_area
    spot_coverage = np.sum(spot_mask > 0) / stone_area

    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    if np.sum(clean_mask > 0) > 0:
        low_freq_var = np.var(blurred[clean_mask > 0])
        low_freq_energy = low_freq_var / 1000.0
    else:
        low_freq_energy = 0.0

    spot_contours, _ = cv2.findContours(
        spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    spot_blob_count = len(spot_contours)

    return PatternMasks(
        pattern_mask=pattern_mask,
        vein_mask=vein_mask_clean,
        spot_mask=spot_mask,
        pattern_coverage=pattern_coverage,
        vein_coverage=vein_coverage,
        spot_coverage=spot_coverage,
        low_freq_energy=low_freq_energy,
        spot_blob_count=spot_blob_count,
    )


def classify_design(
    patterns: PatternMasks,
    stone_area: int,
    glare_mask: np.ndarray,
    reflection_mask: np.ndarray,
    stone_mask: np.ndarray,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Classify design as Plain/Veiny/Spotty/Cloudy.
    Excludes glare and reflection from scoring.
    """
    clean_mask = cv2.bitwise_and(stone_mask, cv2.bitwise_not(glare_mask))
    clean_mask = cv2.bitwise_and(clean_mask, cv2.bitwise_not(reflection_mask))
    clean_area = max(np.sum(clean_mask > 0), 1)

    vein_coverage_clean = (
        np.sum(cv2.bitwise_and(patterns.vein_mask, clean_mask) > 0) / clean_area
    )
    spot_coverage_clean = (
        np.sum(cv2.bitwise_and(patterns.spot_mask, clean_mask) > 0) / clean_area
    )

    vein_score = vein_coverage_clean * 10 + (1.0 if vein_coverage_clean > 0.02 else 0.0)

    spot_density = patterns.spot_blob_count / max(clean_area / 10000, 1)
    spot_score = spot_coverage_clean * 10 + min(spot_density * 0.5, 1.0)

    cloudy_score = (
        patterns.low_freq_energy - (vein_coverage_clean + spot_coverage_clean) * 2
    )
    cloudy_score = max(cloudy_score, 0.0)

    plain_score = 1.0 - (patterns.pattern_coverage * 5)
    plain_score = max(plain_score, 0.0)

    scores = {
        "veiny": vein_score,
        "spotty": spot_score,
        "cloudy": cloudy_score,
        "plain": plain_score,
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_design = sorted_scores[0][0]
    top_score = sorted_scores[0][1]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    total = sum(scores.values()) + 1e-6
    confidence = (top_score - second_score) / total
    confidence = min(max(confidence, 0.0), 1.0)

    if max(scores.values()) < 0.1:
        top_design = "plain"
        confidence = 0.8

    return top_design, confidence, scores


def analyze_veins(
    image: np.ndarray,
    vein_mask: np.ndarray,
    stone_mask: np.ndarray,
    stone_bbox: Tuple[int, int, int, int],
    glare_mask: np.ndarray,
    reflection_mask: np.ndarray,
) -> VeinAnalysis:
    """Analyze vein properties for Design=Veiny, excluding glare/reflection."""
    clean_vein_mask = cv2.bitwise_and(vein_mask, cv2.bitwise_not(glare_mask))
    clean_vein_mask = cv2.bitwise_and(clean_vein_mask, cv2.bitwise_not(reflection_mask))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    angles = np.arctan2(sobely, sobelx) * 180 / np.pi
    angles = angles % 180

    vein_angles = angles[clean_vein_mask > 0]

    if len(vein_angles) > 0:
        hist, bins = np.histogram(vein_angles, bins=18, range=(0, 180))
        hist = hist.astype(float)

        dominant_bin = np.argmax(hist)
        dominant_angle = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2

        hist_sum = np.sum(hist) + 1e-6
        coherence = hist[dominant_bin] / hist_sum

        hist_norm = hist / hist_sum
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    else:
        dominant_angle = 0.0
        coherence = 0.0
        entropy = 4.0

    if coherence < 0.15 or entropy > 3.5:
        direction = "spider"
        direction_conf = min(coherence + 0.3, 0.9)
    else:
        if dominant_angle < 25 or dominant_angle > 155:
            direction = "horizontal"
        elif 65 < dominant_angle < 115:
            direction = "vertical"
        else:
            direction = "diagonal"
        direction_conf = coherence

    stone_area = max(np.sum(stone_mask > 0), 1)
    vein_coverage = np.sum(clean_vein_mask > 0) / stone_area

    if vein_coverage < 0.02:
        distribution = "light_vein"
    elif vein_coverage < 0.08:
        distribution = "medium_vein"
    else:
        distribution = "heavy_vein"
    distribution_conf = 0.7

    if np.sum(clean_vein_mask > 0) > 0:
        dist_transform = cv2.distanceTransform(clean_vein_mask, cv2.DIST_L2, 5)
        vein_widths = dist_transform[clean_vein_mask > 0]
        median_width = np.median(vein_widths) * 2

        bbox_width = stone_bbox[2] - stone_bbox[0]
        rel_width = median_width / max(bbox_width, 1)
    else:
        median_width = 0.0
        rel_width = 0.0

    if rel_width < 0.005:
        thickness = "light"
    elif rel_width < 0.015:
        thickness = "normal"
    else:
        thickness = "heavy"
    thickness_conf = 0.7

    colors, colors_conf = _extract_mask_colors_safe(image, clean_vein_mask)

    return VeinAnalysis(
        direction=direction,
        direction_conf=direction_conf,
        distribution=distribution,
        distribution_conf=distribution_conf,
        thickness=thickness,
        thickness_conf=thickness_conf,
        colors=colors,
        colors_conf=colors_conf,
        dominant_angle=dominant_angle,
        coherence=coherence,
        entropy=entropy,
        rel_width=rel_width,
        coverage=vein_coverage,
    )


def analyze_spots(
    image: np.ndarray,
    spot_mask: np.ndarray,
    stone_mask: np.ndarray,
    glare_mask: np.ndarray,
    reflection_mask: np.ndarray,
) -> SpotAnalysis:
    """Analyze spot properties for Design=Spotty, excluding glare/reflection."""
    clean_spot_mask = cv2.bitwise_and(spot_mask, cv2.bitwise_not(glare_mask))
    clean_spot_mask = cv2.bitwise_and(clean_spot_mask, cv2.bitwise_not(reflection_mask))

    stone_area = max(np.sum(stone_mask > 0), 1)
    spot_coverage = np.sum(clean_spot_mask > 0) / stone_area

    h, w = stone_mask.shape
    grid_size = 4
    cell_h, cell_w = h // grid_size, w // grid_size

    densities = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell_stone = stone_mask[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
            ]
            cell_spot = clean_spot_mask[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w
            ]

            cell_stone_area = np.sum(cell_stone > 0)
            if cell_stone_area > 100:
                density = np.sum(cell_spot > 0) / cell_stone_area
                densities.append(density)

    if len(densities) > 1:
        cv_value = np.std(densities) / (np.mean(densities) + 1e-6)
    else:
        cv_value = 0.0

    if cv_value > 0.8:
        homogeneity = "condensed"
    else:
        homogeneity = "homogeneous"

    if spot_coverage < 0.02:
        level = "light"
    elif spot_coverage < 0.08:
        level = "medium"
    else:
        level = "heavy"

    if homogeneity == "condensed":
        distribution = "condensed"
    else:
        distribution = f"{level}_homogeneous"

    distribution_conf = 0.7

    colors, colors_conf = _extract_mask_colors_safe(image, clean_spot_mask)

    return SpotAnalysis(
        distribution=distribution,
        distribution_conf=distribution_conf,
        colors=colors,
        colors_conf=colors_conf,
        coverage=spot_coverage,
        cv=cv_value,
    )


def analyze_colors(
    image: np.ndarray,
    stone_mask: np.ndarray,
    background_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
    glare_mask: np.ndarray,
    reflection_mask: np.ndarray,
    glare_ratio: float,
    exposure_score: float,
) -> ColorAnalysis:
    """
    Analyze base color with neutral-first gating, exotic detection with chroma-based delta_ab.
    Uses dark quantile sampling for dark stones.
    Includes Hard White Override and Orange saturation requirements.
    """
    h, w = stone_mask.shape

    debug_info = {
        "L_star": 0.0,
        "a_star": 0.0,
        "b_star": 0.0,
        "chroma": 0.0,
        "median_L_star": 0.0,
        "mapping_rule": "none",
        "bg_pixel_count": 0,
        "used_fallback": False,
        "is_dark_stone": False,
        "dark_quantile_used": False,
        "L_raw": 0.0,
        "a_raw": 0.0,
        "b_raw": 0.0,
        "H_raw": 0.0,
        "S_raw": 0.0,
        "V_raw": 0.0,
        "hue_deg": 0.0,
        "sat": 0.0,
        "val": 0.0,
        "fallback_pixel_count": 0,
        "hard_white_triggered": False,
        "orange_blocked": False,
    }

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    bx1, by1, bx2, by2 = inner_bbox
    inner_mask[by1:by2, bx1:bx2] = 255
    fallback_mask = cv2.bitwise_and(inner_mask, stone_mask)
    fallback_mask = cv2.bitwise_and(fallback_mask, cv2.bitwise_not(glare_mask))

    original_bg_pixel_count = int(np.sum(background_mask > 0))
    fallback_pixel_count = int(np.sum(fallback_mask > 0))
    debug_info["bg_pixel_count"] = original_bg_pixel_count
    debug_info["fallback_pixel_count"] = fallback_pixel_count

    if original_bg_pixel_count < MIN_PIXELS:
        print(
            f"[COLOR DEBUG] bg_pixel_count={original_bg_pixel_count} < MIN_PIXELS={MIN_PIXELS}, using fallback_mask with {fallback_pixel_count} pixels"
        )
        background_mask = fallback_mask
        bg_pixel_count = fallback_pixel_count
        debug_info["used_fallback"] = True
    else:
        bg_pixel_count = original_bg_pixel_count

    if bg_pixel_count < 10:
        return ColorAnalysis(
            base_color="grey",
            base_color_conf=0.30,
            is_exotic=False,
            colors_in_stone=[],
            colors_in_stone_conf=0.0,
            tonality="medium",
            tonality_conf=0.30,
            luminance_mean=128.0,
            luminance_var=100.0,
            chroma=0.0,
            cluster_proportions=[1.0],
            cluster_distances=[],
            debug_info=debug_info,
        )

    # Get OpenCV LAB (0-255 scale)
    lab_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_pixels_opencv = lab_opencv[background_mask > 0]

    # Get OpenCV HSV for saturation checks
    hsv_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_pixels_hsv = hsv_opencv[background_mask > 0]

    # Compute RAW OpenCV means (before conversion)
    mean_lab_raw = np.mean(bg_pixels_opencv, axis=0)
    mean_hsv_raw = np.mean(bg_pixels_hsv, axis=0)

    L_raw = float(mean_lab_raw[0])
    a_raw = float(mean_lab_raw[1])
    b_raw = float(mean_lab_raw[2])
    H_raw = float(mean_hsv_raw[0])
    S_raw = float(mean_hsv_raw[1])
    V_raw = float(mean_hsv_raw[2])

    # Store raw values
    debug_info["L_raw"] = L_raw
    debug_info["a_raw"] = a_raw
    debug_info["b_raw"] = b_raw
    debug_info["H_raw"] = H_raw
    debug_info["S_raw"] = S_raw
    debug_info["V_raw"] = V_raw

    # Compute normalized HSV
    hue_deg = H_raw * 2.0  # OpenCV H is 0-180, convert to 0-360
    sat = S_raw / 255.0
    val = V_raw / 255.0

    debug_info["hue_deg"] = hue_deg
    debug_info["sat"] = sat
    debug_info["val"] = val

    # Convert to proper L*a*b* scale for all operations
    bg_pixels_star = _opencv_lab_pixels_to_lab_star(bg_pixels_opencv)

    L_star_values = bg_pixels_star[:, 0]  # Now in 0-100 range
    a_star_values = bg_pixels_star[:, 1]
    b_star_values = bg_pixels_star[:, 2]

    median_L_star = np.median(L_star_values)
    median_a_star = np.median(a_star_values)
    median_b_star = np.median(b_star_values)
    median_chroma = np.sqrt(median_a_star**2 + median_b_star**2)

    debug_info["is_dark_stone"] = median_L_star < DARK_STONE_L_THRESH
    debug_info["median_L_star"] = float(median_L_star)
    debug_info["median_chroma"] = float(median_chroma)

    # Dynamic neutral threshold for dark stones
    dynamic_C_NEUTRAL = max(4.0, C_NEUTRAL * (median_L_star / 70.0))

    # ONLY apply the dark quantile if the stone is both DARK and NEUTRAL!
    # If the stone is dark but HIGHLY COLORED (e.g. Red, Green), taking the darkest 30%
    # will isolate the black shadows and strip the chroma, causing it to misclassify as Black!
    if median_L_star < DARK_STONE_L_THRESH and median_chroma < dynamic_C_NEUTRAL:
        L_threshold = np.percentile(L_star_values, DARK_QUANTILE * 100)
        dark_mask = L_star_values <= L_threshold
        if np.sum(dark_mask) >= 10:
            bg_pixels_for_color = bg_pixels_star[dark_mask]
            debug_info["dark_quantile_used"] = True
        else:
            bg_pixels_for_color = bg_pixels_star
    else:
        bg_pixels_for_color = bg_pixels_star

    # Mean is now in proper L*a*b* scale
    mean_lab_star = np.mean(bg_pixels_for_color, axis=0)
    L_star = float(mean_lab_star[0])  # 0-100
    a_star = float(mean_lab_star[1])  # -128 to +127
    b_star = float(mean_lab_star[2])  # -128 to +127
    chroma = np.sqrt(a_star**2 + b_star**2)

    debug_info["L_star"] = L_star
    debug_info["a_star"] = a_star
    debug_info["b_star"] = b_star
    debug_info["chroma"] = float(chroma)

    # Dynamic neutral threshold based on final mean L_star
    dynamic_final_C_NEUTRAL = max(4.0, C_NEUTRAL * (L_star / 70.0))

    # ===== HARD WHITE OVERRIDE (SAFETY NET) =====
    # If L_star >= 75 AND sat <= 0.10, force White with high confidence
    # This catches white slabs even under warm factory lighting
    if L_star >= 75 and sat <= 0.10:
        base_color = "white"
        base_conf = 0.85
        debug_info["mapping_rule"] = "hard_white_override"
        debug_info["hard_white_triggered"] = True
        print(f"[COLOR DEBUG] HARD WHITE OVERRIDE: L_star={L_star:.1f}, sat={sat:.3f}")
    # Neutral-first gating using proper L*a*b* thresholds
    elif chroma < dynamic_final_C_NEUTRAL:
        if L_star > L_WHITE:
            base_color = "white"
            debug_info["mapping_rule"] = "neutral_white"
        elif L_star < L_CHARCOAL:
            base_color = "black"
            debug_info["mapping_rule"] = "neutral_black"
        else:
            if b_star > B_BEIGE:
                base_color = "beige"
                debug_info["mapping_rule"] = "neutral_beige"
            else:
                base_color = "grey"
                debug_info["mapping_rule"] = "neutral_grey"

        base_conf = 0.75
        if L_star > 90 or L_star < 30:
            base_conf = 0.85
    else:
        # Use L*a*b* scale for color mapping (includes orange saturation check)
        base_color, base_conf = _map_lab_to_color_safe(
            mean_lab_star, chroma, sat=sat, hue_deg=hue_deg
        )
        debug_info["mapping_rule"] = f"chroma_{base_color}"

        # Double-check: Orange without saturation is blocked
        if base_color == "orange" and sat < 0.18:
            debug_info["orange_blocked"] = True
            base_color = "beige"  # Fallback to beige for warm low-sat colors
            base_conf = 0.60
            debug_info["mapping_rule"] = "orange_blocked_to_beige"
            print(
                f"[COLOR DEBUG] ORANGE BLOCKED: sat={sat:.3f} < 0.18, falling back to beige"
            )

        # ===== RED/GREEN SATURATION GATING =====
        if base_color == "red" and sat < 0.15:
            debug_info["red_sat_blocked"] = True
            base_color = "brown"
            base_conf = 0.60
            debug_info["mapping_rule"] = "red_sat_blocked_to_brown"
            print(
                f"[COLOR DEBUG] RED SAT BLOCKED: sat={sat:.3f} < 0.15, falling back to brown"
            )

        if base_color == "green" and sat < 0.15:
            debug_info["green_sat_blocked"] = True
            base_color = "grey"
            base_conf = 0.60
            debug_info["mapping_rule"] = "green_sat_blocked_to_grey"
            print(
                f"[COLOR DEBUG] GREEN SAT BLOCKED: sat={sat:.3f} < 0.15, falling back to grey"
            )

        # ===== RED/GREEN CONFIDENCE GATING =====
        if base_color == "red" and base_conf < 0.75:
            debug_info["conf_gated_from"] = "red"
            if base_conf >= 0.50:
                base_color = "brown"
                debug_info["conf_gated_to"] = "brown"
            else:
                base_color = _nearest_neutral_centroid(mean_lab_star)
                debug_info["conf_gated_to"] = base_color
            debug_info["mapping_rule"] = f"red_conf_gated_to_{base_color}"
            print(
                f"[COLOR DEBUG] RED CONF GATED: conf={base_conf:.3f} < 0.75, snapped to {base_color}"
            )

        if base_color == "green" and base_conf < 0.75:
            debug_info["conf_gated_from"] = "green"
            if base_conf >= 0.50:
                base_color = "grey"
                debug_info["conf_gated_to"] = "grey"
            else:
                base_color = _nearest_neutral_centroid(mean_lab_star)
                debug_info["conf_gated_to"] = base_color
            debug_info["mapping_rule"] = f"green_conf_gated_to_{base_color}"
            print(
                f"[COLOR DEBUG] GREEN CONF GATED: conf={base_conf:.3f} < 0.75, snapped to {base_color}"
            )

    # ===== MULTI DETECTION (on base pixels, before exotic override) =====
    if bg_pixel_count >= 200 and base_color not in ("exotic",):
        try:
            sample_size_multi = min(len(bg_pixels_for_color), 5000)
            sample_idx = (
                np.random.choice(
                    len(bg_pixels_for_color), sample_size_multi, replace=False
                )
                if len(bg_pixels_for_color) > sample_size_multi
                else np.arange(len(bg_pixels_for_color))
            )
            sample_px = bg_pixels_for_color[sample_idx].astype(np.float32)

            n_clust = max(2, min(3, len(sample_px) // 50))
            if len(sample_px) < 100:
                raise ValueError("Too few pixels for multi detection")
            kmeans_multi = KMeans(n_clusters=n_clust, random_state=42, n_init=3)
            labels_multi = kmeans_multi.fit_predict(sample_px)
            centers_multi = kmeans_multi.cluster_centers_

            props_multi = []
            for ci in range(kmeans_multi.n_clusters):
                props_multi.append(
                    float(np.sum(labels_multi == ci) / len(labels_multi))
                )
            sorted_order = np.argsort(props_multi)[::-1]
            props_multi = [props_multi[i] for i in sorted_order]
            centers_multi = centers_multi[sorted_order]

            top1_prop = props_multi[0] if len(props_multi) > 0 else 1.0
            top2_prop = props_multi[1] if len(props_multi) > 1 else 0.0

            debug_info["multi_top1_prop"] = top1_prop
            debug_info["multi_top2_prop"] = top2_prop

            if top1_prop < 0.60 and top2_prop > 0.30 and len(centers_multi) >= 2:
                c1_a, c1_b = float(centers_multi[0][1]), float(centers_multi[0][2])
                c2_a, c2_b = float(centers_multi[1][1]), float(centers_multi[1][2])

                c1_chroma = np.sqrt(c1_a**2 + c1_b**2)
                c2_chroma = np.sqrt(c2_a**2 + c2_b**2)
                c1_neutral = c1_chroma < C_NEUTRAL
                c2_neutral = c2_chroma < C_NEUTRAL

                c1_hue = np.degrees(np.arctan2(c1_b, c1_a)) % 360
                c2_hue = np.degrees(np.arctan2(c2_b, c2_a)) % 360
                hue_delta = abs(c1_hue - c2_hue)
                if hue_delta > 180:
                    hue_delta = 360 - hue_delta

                debug_info["multi_hue_delta"] = float(hue_delta)

                distinct = hue_delta > 60 or (c1_neutral != c2_neutral)

                if distinct:
                    multi_conf = min(0.5 + (0.60 - top1_prop) * 2, 0.90)
                    debug_info["multi_detected"] = True
                    debug_info["multi_conf"] = float(multi_conf)

                    if multi_conf >= 0.75:
                        base_color = "multi"
                        base_conf = multi_conf
                        debug_info["mapping_rule"] = "multi_detected"
                        print(
                            f"[COLOR DEBUG] MULTI DETECTED: top1={top1_prop:.2f}, top2={top2_prop:.2f}, hue_delta={hue_delta:.1f}, conf={multi_conf:.3f}"
                        )
                    elif multi_conf >= 0.50:
                        debug_info["conf_gated_from"] = "multi"
                        debug_info["conf_gated_to"] = base_color
                        debug_info["mapping_rule"] = f"multi_conf_gated_to_{base_color}"
                        print(
                            f"[COLOR DEBUG] MULTI CONF GATED: conf={multi_conf:.3f} < 0.75, keeping {base_color}"
                        )
                    else:
                        fallback = _nearest_neutral_centroid(mean_lab_star)
                        debug_info["conf_gated_from"] = "multi"
                        debug_info["conf_gated_to"] = fallback
                        base_color = fallback
                        debug_info["mapping_rule"] = f"multi_conf_low_to_{fallback}"
                        print(
                            f"[COLOR DEBUG] MULTI LOW CONF: conf={multi_conf:.3f} < 0.50, fallback to {fallback}"
                        )
        except Exception:
            pass

    # Apply glare/exposure penalties (but protect hard white override)
    hard_white_triggered = debug_info.get("hard_white_triggered", False)

    if not hard_white_triggered:
        base_conf *= max(1.0 - glare_ratio * 1.5, 0.2)
        base_conf *= max(exposure_score, 0.3)

    base_conf = min(base_conf, 0.95)

    # Ensure hard white override stays at minimum 0.80
    if hard_white_triggered:
        base_conf = max(base_conf, 0.80)

    is_exotic, colors_in_stone, colors_conf, proportions, cluster_distances = (
        _detect_exotic_chroma_based(lab_opencv, background_mask, stone_mask, inner_bbox)
    )

    if is_exotic:
        base_color = "exotic"
        base_conf = min(colors_conf, 0.90)

    # Use L* values (0-100 scale) for luminance metrics
    luminance_mean = float(np.mean(L_star_values))
    luminance_var = float(np.var(L_star_values))

    # Tonality based on L* (0-100 scale)
    if luminance_mean < 40:
        tonality = "dark"
    elif luminance_mean > 70:
        tonality = "light"
    else:
        tonality = "medium"

    tonality_conf = 1.0 - min(luminance_var / 1000.0, 0.5)

    # Convert numpy types before storing in dataclass
    debug_info = _convert_numpy_types(debug_info)

    return ColorAnalysis(
        base_color=base_color,
        base_color_conf=base_conf,
        is_exotic=is_exotic,
        colors_in_stone=colors_in_stone,
        colors_in_stone_conf=colors_conf,
        tonality=tonality,
        tonality_conf=tonality_conf,
        luminance_mean=luminance_mean,
        luminance_var=luminance_var,
        chroma=chroma,
        cluster_proportions=proportions,
        cluster_distances=cluster_distances,
        debug_info=debug_info,
    )


def _detect_exotic_chroma_based(
    lab_opencv: np.ndarray,
    background_mask: np.ndarray,
    stone_mask: np.ndarray,
    inner_bbox: Tuple[int, int, int, int],
) -> Tuple[bool, List[str], float, List[float], List[float]]:
    """
    Detect exotic using chroma-based delta_ab (not brightness).
    Requires spatial significance via grid coverage.
    Input: lab_opencv is OpenCV LAB format (0-255 scale).
    """
    bg_pixels_opencv = lab_opencv[background_mask > 0]

    if len(bg_pixels_opencv) < 200:
        return False, [], 0.0, [1.0], []

    # Convert to L*a*b* for clustering and comparison
    bg_pixels_star = _opencv_lab_pixels_to_lab_star(bg_pixels_opencv)

    n_clusters = 3
    sample_size = min(len(bg_pixels_star), 5000)
    sample_indices = np.random.choice(len(bg_pixels_star), sample_size, replace=False)
    sample_pixels = bg_pixels_star[sample_indices].astype(np.float32)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(sample_pixels)
        centers = kmeans.cluster_centers_  # Centers are now in L*a*b* scale

        proportions = []
        for i in range(n_clusters):
            prop = np.sum(labels == i) / len(labels)
            proportions.append(prop)

        sorted_idx = np.argsort(proportions)[::-1]
        proportions = [proportions[i] for i in sorted_idx]
        centers = centers[sorted_idx]

        total_prop = sum(proportions)
        if total_prop > 0:
            proportions = [p / total_prop for p in proportions]

    except Exception:
        return False, [], 0.0, [1.0], []

    if len(proportions) < 2 or proportions[1] < EXOTIC_SECOND_PROP:
        return False, [], 0.0, proportions, []

    # Centers are already in L*a*b* scale, a* and b* are already centered around 0
    a1_star, b1_star = centers[0][1], centers[0][2]
    a2_star, b2_star = centers[1][1], centers[1][2]
    delta_ab = np.sqrt((a1_star - a2_star) ** 2 + (b1_star - b2_star) ** 2)

    cluster_distances = [float(delta_ab)]

    if delta_ab < EXOTIC_DELTA_AB:
        return False, [], 0.0, proportions, cluster_distances

    h, w = stone_mask.shape
    grid_size = 4
    cell_h, cell_w = h // grid_size, w // grid_size

    second_cluster_pixels = sample_indices[labels == sorted_idx[1]]

    bg_coords = np.argwhere(background_mask > 0)
    if len(second_cluster_pixels) > 0 and len(bg_coords) >= sample_size:
        second_coords = bg_coords[second_cluster_pixels % len(bg_coords)]

        cells_covered = set()
        for y, x in second_coords:
            cell_i = min(y // cell_h, grid_size - 1)
            cell_j = min(x // cell_w, grid_size - 1)
            cells_covered.add((cell_i, cell_j))

        grid_coverage = len(cells_covered)
    else:
        grid_coverage = 0

    if grid_coverage < EXOTIC_GRID_CELLS:
        return False, [], 0.0, proportions, cluster_distances

    colors_in_stone = []
    for i, center in enumerate(centers[:3]):
        if proportions[i] >= 0.15:
            # Center is already in L*a*b* scale
            chroma_i = np.sqrt(center[1] ** 2 + center[2] ** 2)
            color_name, _ = _map_lab_to_color_safe(center, chroma_i)
            if color_name != "exotic" and color_name not in colors_in_stone:
                colors_in_stone.append(color_name)

    colors_conf = min(proportions[1] * 2, 0.90)

    return True, colors_in_stone, colors_conf, proportions, cluster_distances


def _map_lab_to_color_safe(
    lab_star: np.ndarray,
    chroma: Optional[float] = None,
    sat: Optional[float] = None,
    hue_deg: Optional[float] = None,
) -> Tuple[str, float]:
    """
    Map L*a*b* color to closest color enum value.
    EXPECTS: L*a*b* scale input (L*=0-100, a*/b*=-128 to +127).
    Low-chroma colors cannot map to Orange/Red/Yellow.
    ORANGE requires: sat >= 0.18 AND hue_deg in 15..45 AND L_star >= 55
    """
    L_star = float(lab_star[0])  # 0-100
    a_star = float(lab_star[1])  # -128 to +127
    b_star = float(lab_star[2])  # -128 to +127

    if chroma is None:
        chroma = np.sqrt(a_star**2 + b_star**2)

    # Low-chroma guard: cannot map to warm/chromatic colors
    dynamic_C_NEUTRAL = max(4.0, C_NEUTRAL * (L_star / 70.0))
    if chroma < dynamic_C_NEUTRAL:
        allowed_colors = {"white", "grey", "black", "beige"}
    else:
        allowed_colors = set(COLOR_CENTROIDS_LAB.keys())

    min_dist = float("inf")
    best_color = "grey"  # Default fallback is Grey

    # Weighted Euclidean distance to preserve hue/chroma over lightness
    # L_weight < 1.0 penalizes hue mismatches more strongly than lightness
    L_weight = 0.5
    ab_weight = 1.0

    for color_name, centroid in COLOR_CENTROIDS_LAB.items():
        if color_name not in allowed_colors:
            continue

        centroid_arr = np.array(centroid)  # Already in L*a*b* scale
        dL = L_star - centroid_arr[0]
        da = a_star - centroid_arr[1]
        db = b_star - centroid_arr[2]

        dist = np.sqrt(L_weight * (dL**2) + ab_weight * (da**2 + db**2))
        if dist < min_dist:
            min_dist = dist
            best_color = color_name

    confidence = 1.0 - min(min_dist / 50.0, 0.5)

    return best_color, confidence


def _extract_mask_colors_safe(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[List[str], float]:
    """Extract colors from masked region. Low-chroma clusters map to neutral colors only."""
    if np.sum(mask > 0) < 50:
        return [], 0.0

    lab_opencv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels_opencv = lab_opencv[mask > 0]

    # Convert to L*a*b* scale
    pixels_star = _opencv_lab_pixels_to_lab_star(pixels_opencv)

    if len(pixels_star) < 50:
        mean_color = np.mean(pixels_star, axis=0)
        chroma = np.sqrt(mean_color[1] ** 2 + mean_color[2] ** 2)
        color_name, conf = _map_lab_to_color_safe(mean_color, chroma)
        return [color_name], conf

    n_clusters = min(3, len(pixels_star) // 100)
    n_clusters = max(n_clusters, 1)

    try:
        sample_size = min(len(pixels_star), 2000)
        sample_indices = np.random.choice(len(pixels_star), sample_size, replace=False)
        sample_pixels = pixels_star[sample_indices]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(sample_pixels)

        colors = []
        total_conf = 0.0

        for i, center in enumerate(kmeans.cluster_centers_):
            # Centers are in L*a*b* scale
            prop = np.sum(labels == i) / len(labels)
            if prop >= 0.15:
                chroma = np.sqrt(center[1] ** 2 + center[2] ** 2)
                color_name, conf = _map_lab_to_color_safe(center, chroma)
                if color_name not in colors:
                    colors.append(color_name)
                    total_conf += conf * prop

        return colors, min(total_conf, 0.95)

    except Exception:
        mean_color = np.mean(pixels_star, axis=0)
        chroma = np.sqrt(mean_color[1] ** 2 + mean_color[2] ** 2)
        color_name, conf = _map_lab_to_color_safe(mean_color, chroma)
        return [color_name], conf


def _file_content_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _analysis_cache_key(image_path: str, max_pixels: int, mode: str = "default") -> str:
    file_hash = _file_content_hash(image_path)
    return f"{file_hash}_{_CV_ANALYZER_VERSION}_{mode}_{max_pixels}"


def _load_cached_analysis(cache_key: str) -> Optional[Dict]:
    cache_file = _ANALYSIS_CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_cached_analysis(cache_key: str, result: Dict) -> None:
    cache_file = _ANALYSIS_CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, separators=(",", ":"))
    except Exception:
        pass


def analyze_image(
    image_path: str,
    verbose: bool = False,
    max_pixels: int = 4_000_000,
    mode: str = "default",
) -> Optional[Dict]:
    """
    Full analysis pipeline for a single image.
    Returns dict with all metrics and classifications.

    max_pixels controls the resize cap (default 4MP for backward compat,
    use ~1_500_000–2_000_000 for fast mode).
    mode labels the analysis context (e.g. "fast", "full", "default") and
    is embedded in the cache key so different modes don't collide.
    """

    start_time = time.time()

    try:
        cache_key = _analysis_cache_key(image_path, max_pixels, mode)
        cached = _load_cached_analysis(cache_key)
        if cached is not None:
            return cached
    except Exception:
        cache_key = None

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[CV] Failed to load image: {image_path}")
            return None

        if verbose:
            print(f"[CV] Loaded {image.shape[1]}x{image.shape[0]} image")

        if image.shape[0] * image.shape[1] > max_pixels:
            scale = np.sqrt(max_pixels / (image.shape[0] * image.shape[1]))
            image = cv2.resize(image, None, fx=scale, fy=scale)
            if verbose:
                print(f"[CV] Resized to {image.shape[1]}x{image.shape[0]}")

        stone_result = segment_stone_mask(image)

        mask_set = compute_glare_and_reflection_masks(
            image, stone_result.mask, stone_result.inner_bbox
        )

        quality = compute_quality_metrics(image, stone_result.mask, mask_set)

        h, w = stone_result.mask.shape
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        bx1, by1, bx2, by2 = stone_result.inner_bbox
        inner_mask[by1:by2, bx1:bx2] = 255
        inner_mask = cv2.bitwise_and(inner_mask, stone_result.mask)

        # IMPORTANT: Do not apply gray-world white balance, as it destroys natural stone colors (forces Yellow/Red to Grey)
        balanced = image.copy()

        patterns = detect_patterns_delta_e(
            balanced,
            stone_result.mask,
            stone_result.inner_bbox,
            mask_set.glare_mask,
            mask_set.reflection_mask,
        )

        background_mask = build_background_mask(
            stone_result.mask,
            stone_result.inner_bbox,
            mask_set.glare_mask,
            mask_set.reflection_mask,
            patterns.vein_mask,
            patterns.spot_mask,
        )

        design, design_conf, design_scores = classify_design(
            patterns,
            stone_result.stone_area,
            mask_set.glare_mask,
            mask_set.reflection_mask,
            stone_result.mask,
        )

        vein_analysis = None
        if design == "veiny" or patterns.vein_coverage > 0.01:
            vein_analysis = analyze_veins(
                balanced,
                patterns.vein_mask,
                stone_result.mask,
                stone_result.bbox,
                mask_set.glare_mask,
                mask_set.reflection_mask,
            )

        spot_analysis = None
        if design == "spotty" or patterns.spot_coverage > 0.01:
            spot_analysis = analyze_spots(
                balanced,
                patterns.spot_mask,
                stone_result.mask,
                mask_set.glare_mask,
                mask_set.reflection_mask,
            )

        color_analysis = analyze_colors(
            balanced,
            stone_result.mask,
            background_mask,
            stone_result.inner_bbox,
            mask_set.glare_mask,
            mask_set.reflection_mask,
            mask_set.glare_ratio,
            quality.exposure_score,
        )

        result = {
            "quality": {
                "blur_score": quality.blur_score,
                "exposure_score": quality.exposure_score,
                "glare_score": quality.glare_score,
                "glare_ratio": quality.glare_ratio,
                "reflection_ratio": quality.reflection_ratio,
                "stone_area_ratio": quality.stone_area_ratio,
                "image_weight": quality.image_weight,
            },
            "design": design,
            "design_conf": design_conf,
            "design_scores": design_scores,
            "base_color": color_analysis.base_color,
            "base_color_conf": color_analysis.base_color_conf,
            "is_exotic": color_analysis.is_exotic,
            "colors_in_stone": color_analysis.colors_in_stone,
            "colors_in_stone_conf": color_analysis.colors_in_stone_conf,
            "tonality": color_analysis.tonality,
            "tonality_conf": color_analysis.tonality_conf,
            "luminance_mean": color_analysis.luminance_mean,
            "luminance_var": color_analysis.luminance_var,
            "chroma": color_analysis.chroma,
            "cluster_proportions": color_analysis.cluster_proportions,
            "cluster_distances": color_analysis.cluster_distances,
            "color_debug": color_analysis.debug_info,
            "vein_analysis": {
                "direction": vein_analysis.direction if vein_analysis else None,
                "direction_conf": (
                    vein_analysis.direction_conf if vein_analysis else 0.0
                ),
                "distribution": vein_analysis.distribution if vein_analysis else None,
                "distribution_conf": (
                    vein_analysis.distribution_conf if vein_analysis else 0.0
                ),
                "thickness": vein_analysis.thickness if vein_analysis else None,
                "thickness_conf": (
                    vein_analysis.thickness_conf if vein_analysis else 0.0
                ),
                "colors": vein_analysis.colors if vein_analysis else [],
                "colors_conf": vein_analysis.colors_conf if vein_analysis else 0.0,
                "dominant_angle": (
                    vein_analysis.dominant_angle if vein_analysis else 0.0
                ),
                "coherence": vein_analysis.coherence if vein_analysis else 0.0,
                "entropy": vein_analysis.entropy if vein_analysis else 0.0,
                "rel_width": vein_analysis.rel_width if vein_analysis else 0.0,
                "coverage": vein_analysis.coverage if vein_analysis else 0.0,
            },
            "spot_analysis": {
                "distribution": spot_analysis.distribution if spot_analysis else None,
                "distribution_conf": (
                    spot_analysis.distribution_conf if spot_analysis else 0.0
                ),
                "colors": spot_analysis.colors if spot_analysis else [],
                "colors_conf": spot_analysis.colors_conf if spot_analysis else 0.0,
                "coverage": spot_analysis.coverage if spot_analysis else 0.0,
                "cv": spot_analysis.cv if spot_analysis else 0.0,
            },
            "pattern_coverage": patterns.pattern_coverage,
            "vein_coverage": patterns.vein_coverage,
            "spot_coverage": patterns.spot_coverage,
            "bbox": stone_result.bbox,
            "inner_bbox": stone_result.inner_bbox,
            "stone_area": stone_result.stone_area,
        }

        result = _convert_numpy_types(result)
        if cache_key:
            _save_cached_analysis(cache_key, result)
        return result

    except Exception as e:
        print(f"[CV] Error analyzing {image_path}: {e}")

        traceback.print_exc()
        return None


def _analyze_image_tracked(
    image_path: str, max_pixels: int = 4_000_000, mode: str = "default"
) -> Tuple[Optional[Dict], bool]:
    """Internal wrapper: returns (result, was_cache_hit) without mutating result."""
    try:
        cache_key = _analysis_cache_key(image_path, max_pixels, mode)
        cached = _load_cached_analysis(cache_key)
        if cached is not None:
            return cached, True
    except Exception:
        pass
    result = analyze_image(image_path, max_pixels=max_pixels, mode=mode)
    return result, False
