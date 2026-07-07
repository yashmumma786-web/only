"""
Stock-level Aggregator for Fingerprints V3 - NEW VERSION

Aggregates multiple per-image analyses into a single stock-level fingerprint.
Uses weighted voting based on image quality.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from .schema_v3 import (
    FingerprintV3New, RepresentativeImages, DebugMetrics
)



def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
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
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    return obj


def _fingerprint_to_tag_dict(fp) -> dict:
    tags = {}
    if fp.base_color:
        tags["base_color"] = (fp.base_color, fp.base_color_conf)
        tags["main_color"] = (fp.base_color, fp.base_color_conf)
    if fp.design:
        tags["design"] = (fp.design, fp.design_conf)
        pf = {"plain": "uniform", "veiny": "veined", "spotty": "breccia"}.get(fp.design, fp.design)
        tags["pattern_family"] = (pf, fp.design_conf)
    if fp.tonality:
        tags["tonality"] = (fp.tonality, fp.tonality_conf)
    return tags


def aggregate_stock(
    company_stone_id: str,
    image_analyses: List[Dict],
    image_paths: List[str],
    image_ids: List[str]
) -> FingerprintV3New:
    """
    Aggregate multiple image analyses into one stock-level fingerprint.
    
    Uses weighted voting where weight = image_weight from quality metrics.
    """
    if not image_analyses:
        return FingerprintV3New(
            company_stone_id=company_stone_id,
            image_count=0,
            usable_image_count=0,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    weights = [a.get("quality", {}).get("image_weight", 0.5) for a in image_analyses]
    total_weight = sum(weights) + 1e-6
    
    usable_count = sum(1 for a in image_analyses if a.get("quality", {}).get("stone_area_ratio", 0) >= 0.25)
    
    design, design_conf = _weighted_vote_single(
        [a.get("design") for a in image_analyses],
        [a.get("design_conf", 0.5) for a in image_analyses],
        weights
    )
    
    base_color, base_color_conf = _weighted_vote_single(
        [a.get("base_color") for a in image_analyses],
        [a.get("base_color_conf", 0.5) for a in image_analyses],
        weights
    )
    
    is_exotic = base_color == "exotic"
    
    colors_in_stone = []
    colors_in_stone_conf = 0.0
    if is_exotic:
        colors_in_stone, colors_in_stone_conf = _weighted_vote_multi(
            [a.get("colors_in_stone", []) for a in image_analyses],
            [a.get("colors_in_stone_conf", 0.5) for a in image_analyses],
            weights
        )
    
    tonality, tonality_conf = _weighted_vote_single(
        [a.get("tonality") for a in image_analyses],
        [a.get("tonality_conf", 0.5) for a in image_analyses],
        weights
    )
    
    vein_color = []
    vein_color_conf = 0.0
    vein_direction = None
    vein_direction_conf = 0.0
    vein_distribution = None
    vein_distribution_conf = 0.0
    vein_thickness = None
    vein_thickness_conf = 0.0
    
    if design == "veiny":
        vein_analyses = [a.get("vein_analysis") for a in image_analyses if a.get("vein_analysis")]
        if vein_analyses:
            vein_weights = [weights[i] for i, a in enumerate(image_analyses) if a.get("vein_analysis")]
            
            vein_direction, vein_direction_conf = _weighted_vote_single(
                [v.get("direction") for v in vein_analyses],
                [v.get("direction_conf", 0.5) for v in vein_analyses],
                vein_weights
            )
            
            vein_distribution, vein_distribution_conf = _weighted_vote_single(
                [v.get("distribution") for v in vein_analyses],
                [v.get("distribution_conf", 0.5) for v in vein_analyses],
                vein_weights
            )
            
            vein_thickness, vein_thickness_conf = _weighted_vote_single(
                [v.get("thickness") for v in vein_analyses],
                [v.get("thickness_conf", 0.5) for v in vein_analyses],
                vein_weights
            )
            
            vein_color, vein_color_conf = _weighted_vote_multi(
                [v.get("colors", []) for v in vein_analyses],
                [v.get("colors_conf", 0.5) for v in vein_analyses],
                vein_weights
            )
    
    spot_color = []
    spot_color_conf = 0.0
    spot_distribution = None
    spot_distribution_conf = 0.0
    
    if design == "spotty":
        spot_analyses = [a.get("spot_analysis") for a in image_analyses if a.get("spot_analysis")]
        if spot_analyses:
            spot_weights = [weights[i] for i, a in enumerate(image_analyses) if a.get("spot_analysis")]
            
            spot_distribution, spot_distribution_conf = _weighted_vote_single(
                [s.get("distribution") for s in spot_analyses],
                [s.get("distribution_conf", 0.5) for s in spot_analyses],
                spot_weights
            )
            
            spot_color, spot_color_conf = _weighted_vote_multi(
                [s.get("colors", []) for s in spot_analyses],
                [s.get("colors_conf", 0.5) for s in spot_analyses],
                spot_weights
            )
    
    representative_images = _select_representative_images(
        image_analyses, image_paths, image_ids, design
    )
    
    debug_metrics = _compute_debug_metrics(image_analyses, design)
    
    fp = FingerprintV3New(
        company_stone_id=company_stone_id,
        base_color=base_color,
        base_color_conf=base_color_conf,
        colors_in_stone=colors_in_stone,
        colors_in_stone_conf=colors_in_stone_conf,
        tonality=tonality,
        tonality_conf=tonality_conf,
        design=design,
        design_conf=design_conf,
        vein_color=vein_color,
        vein_color_conf=vein_color_conf,
        vein_direction=vein_direction,
        vein_direction_conf=vein_direction_conf,
        vein_distribution=vein_distribution,
        vein_distribution_conf=vein_distribution_conf,
        vein_thickness=vein_thickness,
        vein_thickness_conf=vein_thickness_conf,
        spot_color=spot_color,
        spot_color_conf=spot_color_conf,
        spot_distribution=spot_distribution,
        spot_distribution_conf=spot_distribution_conf,
        representative_images=representative_images,
        debug_metrics=debug_metrics,
        image_count=len(image_analyses),
        usable_image_count=usable_count,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    fp.apply_applicability_rules()
    
    return fp


def _weighted_vote_single(
    values: List[Optional[str]],
    confidences: List[float],
    weights: List[float]
) -> Tuple[Optional[str], float]:
    """Weighted voting for single-select field."""
    if not values:
        return None, 0.0
    
    scores = {}
    for val, conf, weight in zip(values, confidences, weights):
        if val is not None:
            score = conf * weight
            scores[val] = scores.get(val, 0.0) + score
    
    if not scores:
        return None, 0.0
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = sorted_scores[0][0]
    top_score = sorted_scores[0][1]
    total = sum(scores.values()) + 1e-6
    
    confidence = top_score / total
    
    return winner, min(confidence, 0.99)


def _weighted_vote_multi(
    values_lists: List[List[str]],
    confidences: List[float],
    weights: List[float]
) -> Tuple[List[str], float]:
    """Weighted voting for multi-select field."""
    if not values_lists:
        return [], 0.0
    
    scores = {}
    for vals, conf, weight in zip(values_lists, confidences, weights):
        for val in vals:
            score = conf * weight
            scores[val] = scores.get(val, 0.0) + score
    
    if not scores:
        return [], 0.0
    
    max_score = max(scores.values())
    threshold = max_score * 0.3
    
    winners = [v for v, s in scores.items() if s >= threshold]
    winners = sorted(winners, key=lambda v: scores[v], reverse=True)[:5]
    
    total = sum(scores.values()) + 1e-6
    avg_conf = sum(scores.get(v, 0) for v in winners) / (len(winners) * total) if winners else 0.0
    
    return winners, min(avg_conf, 0.99)


def _select_representative_images(
    analyses: List[Dict],
    image_paths: List[str],
    image_ids: List[str],
    design: str
) -> RepresentativeImages:
    """Select 3-4 representative images for review."""
    if not analyses or not image_ids:
        return RepresentativeImages()
    
    weights = [a.get("quality", {}).get("image_weight", 0.5) for a in analyses]
    exposures = [a.get("quality", {}).get("exposure_score", 0.5) for a in analyses]
    
    rep_scores = [w * 0.7 + e * 0.3 for w, e in zip(weights, exposures)]
    rep_idx = int(np.argmax(rep_scores))
    representative = image_ids[rep_idx] if rep_idx < len(image_ids) else None
    
    tonalities = [a.get("tonality") for a in analyses]
    rep_tonality = tonalities[rep_idx] if rep_idx < len(tonalities) else None
    
    variation = None
    for i, ton in enumerate(tonalities):
        if ton != rep_tonality and i != rep_idx:
            variation = image_ids[i]
            break
    
    pattern_clarity = None
    if design == "veiny":
        vein_coverages = [a.get("vein_coverage", 0) for a in analyses]
        if vein_coverages:
            clarity_idx = int(np.argmax(vein_coverages))
            if clarity_idx != rep_idx:
                pattern_clarity = image_ids[clarity_idx] if clarity_idx < len(image_ids) else None
    elif design == "spotty":
        spot_coverages = [a.get("spot_coverage", 0) for a in analyses]
        if spot_coverages:
            clarity_idx = int(np.argmax(spot_coverages))
            if clarity_idx != rep_idx:
                pattern_clarity = image_ids[clarity_idx] if clarity_idx < len(image_ids) else None
    
    designs = [a.get("design") for a in analyses]
    outlier = None
    for i, d in enumerate(designs):
        if d != design and i != rep_idx:
            outlier = image_ids[i]
            break
    
    return RepresentativeImages(
        representative=representative,
        variation=variation,
        pattern_clarity=pattern_clarity,
        outlier=outlier
    )


def _compute_debug_metrics(analyses: List[Dict], design: str) -> DebugMetrics:
    """Compute aggregated debug metrics for calibration."""
    if not analyses:
        return DebugMetrics()
    
    qualities = [a.get("quality", {}) for a in analyses]
    
    avg_stone_area = np.mean([q.get("stone_area_ratio", 0) for q in qualities])
    avg_blur = np.mean([q.get("blur_score", 0) for q in qualities])
    avg_exposure = np.mean([q.get("exposure_score", 0) for q in qualities])
    avg_glare = np.mean([q.get("glare_score", 0) for q in qualities])
    avg_glare_ratio = np.mean([q.get("glare_ratio", 0) for q in qualities])
    avg_reflection_ratio = np.mean([q.get("reflection_ratio", 0) for q in qualities])
    
    vein_coverages = [a.get("vein_coverage", 0) for a in analyses]
    vein_coverage_mean = np.mean(vein_coverages) if vein_coverages else 0.0
    vein_coverage_std = np.std(vein_coverages) if len(vein_coverages) > 1 else 0.0
    
    spot_coverages = [a.get("spot_coverage", 0) for a in analyses]
    spot_coverage_mean = np.mean(spot_coverages) if spot_coverages else 0.0
    spot_coverage_std = np.std(spot_coverages) if len(spot_coverages) > 1 else 0.0
    
    vein_rel_widths = []
    dominant_angles = []
    coherences = []
    spot_cvs = []
    chroma_values = []
    
    for a in analyses:
        if a.get("vein_analysis"):
            vein_rel_widths.append(a["vein_analysis"].get("rel_width", 0))
            dominant_angles.append(a["vein_analysis"].get("dominant_angle", 0))
            coherences.append(a["vein_analysis"].get("coherence", 0))
        if a.get("spot_analysis"):
            spot_cvs.append(a["spot_analysis"].get("cv", 0))
        if a.get("chroma") is not None:
            chroma_values.append(a["chroma"])
    
    luminance_means = [a.get("luminance_mean", 128) for a in analyses]
    luminance_vars = [a.get("luminance_var", 100) for a in analyses]
    
    cluster_props = []
    cluster_dists = []
    color_debug_combined = {}
    for a in analyses:
        if a.get("cluster_proportions"):
            cluster_props.extend(a["cluster_proportions"])
        if a.get("cluster_distances"):
            cluster_dists.extend(a["cluster_distances"])
        if a.get("color_debug"):
            color_debug_combined = a["color_debug"]
    
    return DebugMetrics(
        avg_stone_area_ratio=float(avg_stone_area),
        avg_blur_score=float(avg_blur),
        avg_exposure_score=float(avg_exposure),
        avg_glare_score=float(avg_glare),
        avg_glare_ratio=float(avg_glare_ratio),
        avg_reflection_ratio=float(avg_reflection_ratio),
        vein_coverage_mean=float(vein_coverage_mean),
        vein_coverage_std=float(vein_coverage_std),
        vein_rel_width_mean=float(np.mean(vein_rel_widths)) if vein_rel_widths else 0.0,
        vein_rel_width_std=float(np.std(vein_rel_widths)) if len(vein_rel_widths) > 1 else 0.0,
        dominant_angle_mean=float(np.mean(dominant_angles)) if dominant_angles else 0.0,
        angle_coherence_mean=float(np.mean(coherences)) if coherences else 0.0,
        spot_coverage_mean=float(spot_coverage_mean),
        spot_coverage_std=float(spot_coverage_std),
        spot_cv_mean=float(np.mean(spot_cvs)) if spot_cvs else 0.0,
        luminance_mean=float(np.mean(luminance_means)),
        luminance_var=float(np.mean(luminance_vars)),
        chroma_mean=float(np.mean(chroma_values)) if chroma_values else 0.0,
        cluster_proportions=_convert_numpy_types(cluster_props[:10]),
        cluster_distances=_convert_numpy_types(cluster_dists[:10]),
        color_debug=_convert_numpy_types(color_debug_combined)
    )


