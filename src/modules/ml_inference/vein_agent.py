import cv2
import numpy as np
from typing import Optional
from src.modules.ml_inference import color_agent

MIN_VEIN_PIXELS_PER_IMAGE = 200
MIN_VEIN_COVERAGE_FOR_ANY = 0.005
NEUTRAL_CHROMA_GATE = 8.0
LIGHT_L_THRESH = 70.0
DARK_L_THRESH = 30.0
MIXED_HUE_SECOND_SHARE = 0.30
CHROMATIC_SHARE_GATE = 0.15
DOMINANT_BUCKET_MIN_SHARE = 0.55

def _aggregate_vein(images_bgr: list[np.ndarray]) -> dict:
    bucket_counts = {"gold": 0, "green": 0, "grey": 0, "pink": 0}
    total_vein = 0
    total_px = 0
    neutral_L: list[float] = []

    for img in images_bgr:
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800.0 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale)
            h, w = img.shape[:2]
        total_px += h * w

        mask = color_agent._compute_simple_vein_mask(img)
        vp = int(np.sum(mask > 0))
        if vp < MIN_VEIN_PIXELS_PER_IMAGE:
            continue
        total_vein += vp

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0] * (100.0 / 255.0)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0

        mb = mask > 0
        Lv, av, bv = L[mb], a[mb], b[mb]
        Cv = np.sqrt(av * av + bv * bv)
        chrom = Cv >= NEUTRAL_CHROMA_GATE
        neut = ~chrom
        if int(np.sum(neut)) > 0:
            neutral_L.extend(Lv[neut].tolist())

        if int(np.sum(chrom)) > 0:
            h_deg = (np.degrees(np.arctan2(bv[chrom], av[chrom]))) % 360.0
            for label, lo, hi in (
                ("gold", 25.0, 90.0),
                ("green", 90.0, 165.0),
                ("grey", 165.0, 250.0),
                ("pink", 250.0, 340.0),
            ):
                bucket_counts[label] += int(np.sum((h_deg >= lo) & (h_deg < hi)))
            bucket_counts["pink"] += int(np.sum((h_deg >= 340.0) | (h_deg < 25.0)))

    return {
        "total_vein": total_vein,
        "total_px": total_px,
        "bucket_counts": bucket_counts,
        "neutral_count": len(neutral_L),
        "neutral_mean_L": (float(np.mean(neutral_L)) if neutral_L else None),
    }

def classify_vein_colour(agg: dict) -> str:
    vein_share = agg["total_vein"] / max(agg["total_px"], 1)
    if vein_share < MIN_VEIN_COVERAGE_FOR_ANY:
        return "none"
    chrom_total = sum(agg["bucket_counts"].values())
    grand_total = chrom_total + agg["neutral_count"]
    if grand_total == 0:
        return "none"
    chromatic_share = chrom_total / grand_total
    L = agg["neutral_mean_L"]
    neutral_color = "grey"
    if L is not None:
        if L >= LIGHT_L_THRESH:
            neutral_color = "white"
        elif L < DARK_L_THRESH:
            neutral_color = "black"

    if chromatic_share < CHROMATIC_SHARE_GATE:
        return neutral_color
    sorted_b = sorted(agg["bucket_counts"].items(), key=lambda x: -x[1])
    top_n, top_c = sorted_b[0]
    sec_n, sec_c = sorted_b[1]
    if (top_c / chrom_total) < DOMINANT_BUCKET_MIN_SHARE:
        return neutral_color
    if (sec_c / chrom_total) >= MIXED_HUE_SECOND_SHARE:
        return "mixed"
    return top_n

def classify_vein_colour_from_images(images_bgr: list[np.ndarray]) -> Optional[str]:
    if not images_bgr:
        return None
    agg = _aggregate_vein(images_bgr)
    return classify_vein_colour(agg)
