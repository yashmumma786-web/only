"""
Patch-based "Overall Impression" Color Agent

Uses grid-balanced patch sampling (6x6) to determine:
- main_color: dominant overall impression
- second_color: second dominant (only if >= 25% share and spatially spread)
- exotic: true only if second_color is strong, different, and spread

Key features:
- LAB colorspace for accurate color mapping
- Glare/reflection exclusion
- Spatial spread test to prevent stripes/reflections becoming second color
- Weighted clustering based on patch quality
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
import math

from src.utils import image_fetch


def crop_to_slab_region(img: np.ndarray) -> np.ndarray:
    """Wrapper for image_fetch.crop_to_slab_region with default slab crop params."""
    return image_fetch.crop_to_slab_region(
        img,
        left_pct=0.10,
        right_pct=0.10,
        top_pct=0.10,
        bottom_pct=0.20
    )

SECOND_COLOR_MIN_SHARE = 0.25
MIN_CELL_SPREAD = 6
MIN_CELL_SPREAD_RATIO = 0.25

ENABLE_WHITE_FIELD_RESCUE = True
ENABLE_VEIN_MASK_SAMPLING = False
VEIN_MASK_L_WEIGHT = 0.35
VEIN_MASK_DELTA_E_THRESH = 10.0
VEIN_MASK_MIN_FIELD_PATCHES = 4

@dataclass
class PatchInfo:
    center: Tuple[int, int]
    grid_cell: Tuple[int, int]
    mean_lab: Tuple[float, float, float]
    weight: float
    glare_ratio: float
    valid_pixels: int


@dataclass
class ColorAgentResult:
    main_L: float
    main_a: float
    main_b: float
    main_share: float
    second_L: float
    second_a: float
    second_b: float
    second_share: float
    delta_ab: float
    second_cell_spread: int
    total_cells: int
    has_second: bool
    stone_lightness_mean: Optional[float] = None
    vein_coverage_pct: float = 0.0
    patch_count: int = 0
    rescue_lighter_L: float = 0.0
    rescue_lighter_share: float = 0.0
    rescue_lighter_chroma: float = 0.0
    rescue_enabled: bool = False
    debug_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "main_L": self.main_L,
            "main_a": self.main_a,
            "main_b": self.main_b,
            "main_share": self.main_share,
            "second_L": self.second_L,
            "second_a": self.second_a,
            "second_b": self.second_b,
            "second_share": self.second_share,
            "delta_ab": self.delta_ab,
            "second_cell_spread": self.second_cell_spread,
            "total_cells": self.total_cells,
            "has_second": self.has_second,
            "stone_lightness_mean": self.stone_lightness_mean,
            "vein_coverage_pct": self.vein_coverage_pct,
            "patch_count": self.patch_count,
            "rescue_lighter_L": self.rescue_lighter_L,
            "rescue_lighter_share": self.rescue_lighter_share,
            "rescue_lighter_chroma": self.rescue_lighter_chroma,
            "rescue_enabled": self.rescue_enabled,
            "debug_metrics": self.debug_metrics
        }


def compute_stone_lightness_mean(all_patches: List["PatchInfo"]) -> Optional[float]:
    if not all_patches:
        return None
    total_w = 0.0
    weighted_L = 0.0
    for p in all_patches:
        w = float(p.weight) if p.weight is not None else 0.0
        if w <= 0:
            continue
        weighted_L += float(p.mean_lab[0]) * w
        total_w += w
    if total_w <= 0:
        return None
    return round(float(weighted_L / total_w), 1)


def opencv_lab_to_lab_star(L_raw: float, a_raw: float, b_raw: float) -> Tuple[float, float, float]:
    L_star = L_raw * (100.0 / 255.0)
    a_star = a_raw - 128.0
    b_star = b_raw - 128.0
    return L_star, a_star, b_star



def create_glare_mask(img_lab: np.ndarray, img_hsv: np.ndarray) -> np.ndarray:
    L_channel = img_lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    S_channel = img_hsv[:, :, 1].astype(np.float32) / 255.0
    
    glare_mask = (L_channel > 92) & (S_channel < 0.08)
    
    return glare_mask.astype(np.uint8)


def create_reflection_mask(img_lab: np.ndarray, img_hsv: np.ndarray) -> np.ndarray:
    L_channel = img_lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    S_channel = img_hsv[:, :, 1].astype(np.float32) / 255.0
    
    median_L = np.median(L_channel)
    
    gray = cv2.cvtColor(cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture_abs = np.abs(texture)
    texture_thresh = np.percentile(texture_abs, 75)
    
    reflection_mask = (
        (L_channel > median_L + 12) & 
        (S_channel < 0.12) & 
        (texture_abs > texture_thresh)
    )
    
    return reflection_mask.astype(np.uint8)


def get_inner_bbox(img_shape: Tuple[int, int], shrink_ratio: float = 0.10) -> Tuple[int, int, int, int]:
    h, w = img_shape[:2]
    margin_x = int(w * shrink_ratio)
    margin_y = int(h * shrink_ratio)
    
    x1 = margin_x
    y1 = margin_y
    x2 = w - margin_x
    y2 = h - margin_y
    
    return x1, y1, x2, y2


def sample_patches_grid(
    img_lab: np.ndarray,
    img_hsv: np.ndarray,
    stone_mask: Optional[np.ndarray] = None,
    extra_exclude_mask: Optional[np.ndarray] = None,
    grid_size: int = 6,
    patch_size: int = 64
) -> List[PatchInfo]:
    h, w = img_lab.shape[:2]
    x1, y1, x2, y2 = get_inner_bbox((h, w))
    roi_w = x2 - x1
    roi_h = y2 - y1
    
    if roi_w < grid_size * 10 or roi_h < grid_size * 10:
        return []
    
    glare_mask = create_glare_mask(img_lab, img_hsv)
    reflection_mask = create_reflection_mask(img_lab, img_hsv)
    exclude_mask = np.maximum(glare_mask, reflection_mask)

    if extra_exclude_mask is not None:
        exclude_mask = np.maximum(exclude_mask, extra_exclude_mask)
    
    cell_w = roi_w // grid_size
    cell_h = roi_h // grid_size
    
    adaptive_patch = min(patch_size, cell_w // 2, cell_h // 2)
    adaptive_patch = max(adaptive_patch, 16)
    half_patch = adaptive_patch // 2
    
    patches = []
    
    for gi in range(grid_size):
        for gj in range(grid_size):
            cell_x1 = x1 + gj * cell_w + half_patch
            cell_y1 = y1 + gi * cell_h + half_patch
            cell_x2 = x1 + (gj + 1) * cell_w - half_patch
            cell_y2 = y1 + (gi + 1) * cell_h - half_patch
            
            if cell_x2 <= cell_x1 or cell_y2 <= cell_y1:
                continue
            
            cx = np.random.randint(cell_x1, cell_x2)
            cy = np.random.randint(cell_y1, cell_y2)
            
            px1 = max(0, cx - half_patch)
            py1 = max(0, cy - half_patch)
            px2 = min(w, cx + half_patch)
            py2 = min(h, cy + half_patch)
            
            patch_lab = img_lab[py1:py2, px1:px2]
            patch_exclude = exclude_mask[py1:py2, px1:px2]
            
            if stone_mask is not None:
                patch_stone = stone_mask[py1:py2, px1:px2]
                valid_mask = (patch_stone > 0) & (patch_exclude == 0)
            else:
                valid_mask = patch_exclude == 0
            
            valid_count = np.sum(valid_mask)
            total_pixels = valid_mask.size
            
            if valid_count < total_pixels * 0.3:
                continue
            
            valid_pixels = patch_lab[valid_mask]
            
            mean_L = np.mean(valid_pixels[:, 0])
            mean_a = np.mean(valid_pixels[:, 1])
            mean_b = np.mean(valid_pixels[:, 2])
            
            L_star, a_star, b_star = opencv_lab_to_lab_star(mean_L, mean_a, mean_b)
            
            glare_ratio = np.sum(patch_exclude) / total_pixels
            
            weight = 1.0
            weight *= max(0.1, 1.0 - glare_ratio * 1.5)
            weight *= max(0.3, valid_count / total_pixels)
            
            patches.append(PatchInfo(
                center=(cx, cy),
                grid_cell=(gi, gj),
                mean_lab=(L_star, a_star, b_star),
                weight=weight,
                glare_ratio=glare_ratio,
                valid_pixels=valid_count
            ))
    
    return patches


def cluster_patches(patches: List[PatchInfo], n_clusters: int = 2) -> Tuple[List[int], np.ndarray]:
    if len(patches) < n_clusters:
        return [0] * len(patches), np.array([[p.mean_lab[0], p.mean_lab[1], p.mean_lab[2]] for p in patches])
    
    X = np.array([[p.mean_lab[0], p.mean_lab[1], p.mean_lab[2]] for p in patches])
    weights = np.array([p.weight for p in patches])
    
    sample_indices = []
    for i, w in enumerate(weights):
        repeat = max(1, int(w * 10))
        sample_indices.extend([i] * repeat)
    
    X_weighted = X[sample_indices]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_weighted)
    
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    
    return labels.tolist(), centers


def compute_cluster_shares(patches: List[PatchInfo], labels: List[int], n_clusters: int) -> Dict[int, float]:
    cluster_weights = {i: 0.0 for i in range(n_clusters)}
    
    for patch, label in zip(patches, labels):
        cluster_weights[label] += patch.weight
    
    total_weight = sum(cluster_weights.values())
    
    if total_weight == 0:
        return {i: 1.0 / n_clusters for i in range(n_clusters)}
    
    return {i: w / total_weight for i, w in cluster_weights.items()}


def compute_cell_spread(patches: List[PatchInfo], labels: List[int], cluster_id: int) -> int:
    cells = set()
    for patch, label in zip(patches, labels):
        if label == cluster_id:
            cells.add(patch.grid_cell)
    return len(cells)


def _apply_white_field_rescue_stats(
    centers: np.ndarray,
    shares: Dict[int, float],
) -> Dict:
    """Returns lighter cluster stats for taxonomy to evaluate white field rescue."""
    lighter_idx = 0 if float(centers[0][0]) >= float(centers[1][0]) else 1
    lighter_L = float(centers[lighter_idx][0])
    lighter_a = float(centers[lighter_idx][1])
    lighter_b = float(centers[lighter_idx][2])
    lighter_chroma = math.sqrt(lighter_a ** 2 + lighter_b ** 2)
    lighter_share = shares[lighter_idx]

    return {
        "lighter_L": round(lighter_L, 2),
        "lighter_chroma": round(lighter_chroma, 2),
        "lighter_share": round(lighter_share, 3),
    }


def _compute_simple_vein_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Produce a binary vein/pattern mask using delta-E from image median.

    Self-contained: takes a BGR image and returns a uint8 mask (255=vein,
    0=field) of the same spatial dimensions.  Uses the same delta-E weights
    as cv_analyzer.detect_patterns_delta_e so results are comparable, but
    does NOT require stone_mask / inner_bbox prerequisites.

    Returns an empty (all-zero) mask if the image is too small or too bright.
    """
    h, w = img_bgr.shape[:2]
    if h < 32 or w < 32:
        return np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_img, a_img, b_img = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    glare_mask = (l_img > 240).astype(bool)
    field_mask = ~glare_mask

    if np.sum(field_mask) < 100:
        return np.zeros((h, w), dtype=np.uint8)

    bg_L = float(np.median(l_img[field_mask]))
    bg_a = float(np.median(a_img[field_mask]))
    bg_b = float(np.median(b_img[field_mask]))

    delta_e = np.sqrt(
        VEIN_MASK_L_WEIGHT * (l_img - bg_L) ** 2
        + (a_img - bg_a) ** 2
        + (b_img - bg_b) ** 2
    )

    vein_candidate = (
        (delta_e > VEIN_MASK_DELTA_E_THRESH) & field_mask
    ).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    vein_mask = cv2.morphologyEx(vein_candidate, cv2.MORPH_CLOSE, kernel)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel)

    return vein_mask





def analyze_stock_colors_from_arrays(
    images: List[np.ndarray],
    stone_masks: Optional[List[np.ndarray]] = None,
    apply_slab_crop: bool = True,
    _rescue_override: Optional[bool] = None,
    _vein_mask_override: Optional[bool] = None
) -> ColorAgentResult:
    """
    Analyze colors from pre-loaded BGR numpy arrays (in-memory images).
    No disk I/O - images are already loaded in memory.
    
    apply_slab_crop: If True (default), crops images to focus on slab region,
                     removing background edges (10% from sides/top) and 
                     label strip (20% from bottom).
    """
    if not images:
        return ColorAgentResult(
            main_L=0.0, main_a=0.0, main_b=0.0, main_share=1.0,
            second_L=0.0, second_a=0.0, second_b=0.0, second_share=0.0,
            delta_ab=0.0, second_cell_spread=0, total_cells=0, has_second=False,
            stone_lightness_mean=None, vein_coverage_pct=0.0, patch_count=0,
            debug_metrics={"error": "no_images"}
        )
    
    all_patches = []
    vein_enabled = _vein_mask_override if _vein_mask_override is not None else ENABLE_VEIN_MASK_SAMPLING
    vein_masks_applied = 0
    vein_coverage_sum = 0.0

    for i, img in enumerate(images[:3]):
        if img is None:
            continue
        
        if apply_slab_crop:
            img = crop_to_slab_region(img)
            if img is None or img.size == 0:
                continue
        
        stone_mask = stone_masks[i] if stone_masks and i < len(stone_masks) else None
        
        max_dim = 800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            if stone_mask is not None:
                stone_mask = cv2.resize(stone_mask, (img.shape[1], img.shape[0]))
        
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        vein_mask_arr: Optional[np.ndarray] = None
        vein_coverage_pct_img = 0.0
        if vein_enabled:
            vein_mask_arr = _compute_simple_vein_mask(img)
            total_px = vein_mask_arr.size
            vein_coverage_pct_img = float(np.sum(vein_mask_arr > 0)) / max(total_px, 1)

        patches = sample_patches_grid(img_lab, img_hsv, stone_mask,
                                       extra_exclude_mask=vein_mask_arr if vein_enabled else None)
        if vein_enabled and len(patches) < VEIN_MASK_MIN_FIELD_PATCHES:
            patches = sample_patches_grid(img_lab, img_hsv, stone_mask)
        elif vein_enabled:
            vein_masks_applied += 1
            vein_coverage_sum += vein_coverage_pct_img

        all_patches.extend(patches)

    vein_mask_applied = vein_masks_applied > 0
    vein_coverage_pct = round(vein_coverage_sum / max(vein_masks_applied, 1), 4) if vein_mask_applied else 0.0
    
    if len(all_patches) < 4:
        return ColorAgentResult(
            main_L=0.0, main_a=0.0, main_b=0.0, main_share=1.0,
            second_L=0.0, second_a=0.0, second_b=0.0, second_share=0.0,
            delta_ab=0.0, second_cell_spread=0, total_cells=0, has_second=False,
            stone_lightness_mean=None, vein_coverage_pct=vein_coverage_pct, patch_count=len(all_patches),
            debug_metrics={"error": "insufficient_patches", "patch_count": len(all_patches)}
        )
    
    labels, centers = cluster_patches(all_patches, n_clusters=2)
    shares = compute_cluster_shares(all_patches, labels, 2)
    
    main_cluster = max(shares.keys(), key=lambda k: shares[k])
    second_cluster = 1 - main_cluster
    
    main_share = shares[main_cluster]
    second_share = shares[second_cluster]
    
    main_center = centers[main_cluster]
    second_center = centers[second_cluster]
    
    main_L, main_a, main_b = main_center
    second_L, second_a, second_b = second_center
    
    rescue_enabled = _rescue_override if _rescue_override is not None else ENABLE_WHITE_FIELD_RESCUE
    field_rescue_stats = _apply_white_field_rescue_stats(centers, shares)
    
    total_cells = len(set(p.grid_cell for p in all_patches))
    second_cell_spread = compute_cell_spread(all_patches, labels, second_cluster)
    min_cells_required = max(MIN_CELL_SPREAD, int(total_cells * MIN_CELL_SPREAD_RATIO))
    
    has_second = (
        second_share >= SECOND_COLOR_MIN_SHARE and
        second_cell_spread >= min_cells_required
    )
    
    delta_ab = math.sqrt((main_a - second_a)**2 + (main_b - second_b)**2) if has_second else 0.0
    
    stone_L_mean = compute_stone_lightness_mean(all_patches)

    system_tags = []
    if vein_mask_applied:
        system_tags.append("vein_mask_sampling")

    debug_metrics = {
        "patch_count_used": len(all_patches),
        "images_analyzed": min(len(images), 3),
        "main_share": round(main_share, 3),
        "second_share": round(second_share, 3),
        "main_center_lab": [round(x, 2) for x in main_center.tolist()],
        "second_center_lab": [round(x, 2) for x in second_center.tolist()],
        "delta_ab": round(delta_ab, 2) if has_second else 0,
        "second_cell_spread": second_cell_spread,
        "total_cells_used": total_cells,
        "min_cells_required": min_cells_required,
        "has_second": has_second,
        "avg_glare_ratio": round(np.mean([p.glare_ratio for p in all_patches]), 3),
        "source": "in_memory",
        "vein_mask_applied": vein_mask_applied,
        "vein_coverage_pct": vein_coverage_pct,
        "system_tags": system_tags,
    }

    return ColorAgentResult(
        main_L=round(float(main_L), 2),
        main_a=round(float(main_a), 2),
        main_b=round(float(main_b), 2),
        main_share=round(float(main_share), 3),
        second_L=round(float(second_L), 2),
        second_a=round(float(second_a), 2),
        second_b=round(float(second_b), 2),
        second_share=round(float(second_share), 3),
        delta_ab=round(float(delta_ab), 2),
        second_cell_spread=second_cell_spread,
        total_cells=total_cells,
        has_second=has_second,
        stone_lightness_mean=stone_L_mean,
        vein_coverage_pct=vein_coverage_pct,
        patch_count=len(all_patches),
        rescue_lighter_L=field_rescue_stats["lighter_L"],
        rescue_lighter_share=field_rescue_stats["lighter_share"],
        rescue_lighter_chroma=field_rescue_stats["lighter_chroma"],
        rescue_enabled=rescue_enabled,
        debug_metrics=debug_metrics
    )



