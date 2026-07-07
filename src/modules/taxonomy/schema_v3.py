"""
Schema definitions for Fingerprints V3 - NEW VERSION

Defines all enums, data classes for stock-level fingerprints with:
- Design classification (Plain/Veiny/Spotty/Cloudy)
- Conditional Veiny branch (direction, distribution, thickness, color)
- Conditional Spotty branch (distribution, color)
- Exotic detection with colors_in_stone
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


class BaseColor(str, Enum):
    GREY = "grey"
    WHITE = "white"
    BLUE = "blue"
    BLACK = "black"
    PINK = "pink"
    RED = "red"
    ROSE = "rose"
    BROWN = "brown"
    YELLOW = "yellow"
    ORANGE = "orange"
    GREEN = "green"
    BEIGE = "beige"
    EXOTIC = "exotic"
    CHARCOAL = "charcoal"
    PURPLE = "purple"
    GOLD = "gold"
    MULTI = "multi"

class Design(str, Enum):
    PLAIN = "plain"
    VEINY = "veiny"
    SPOTTY = "spotty"
    CLOUDY = "cloudy"

@dataclass 
class RepresentativeImages:
    """3-4 representative images for manual review."""
    representative: Optional[str] = None
    variation: Optional[str] = None
    pattern_clarity: Optional[str] = None
    outlier: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "representative": self.representative,
            "variation": self.variation,
            "pattern_clarity": self.pattern_clarity,
            "outlier": self.outlier
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RepresentativeImages":
        return cls(
            representative=data.get("representative"),
            variation=data.get("variation"),
            pattern_clarity=data.get("pattern_clarity"),
            outlier=data.get("outlier")
        )
    
    def to_list(self) -> List[str]:
        return [img for img in [self.representative, self.variation, self.pattern_clarity, self.outlier] if img]


@dataclass
class DebugMetrics:
    """Aggregated debug metrics for calibration."""
    avg_stone_area_ratio: float = 0.0
    avg_blur_score: float = 0.0
    avg_exposure_score: float = 0.0
    avg_glare_score: float = 0.0
    avg_glare_ratio: float = 0.0
    avg_reflection_ratio: float = 0.0
    
    vein_coverage_mean: float = 0.0
    vein_coverage_std: float = 0.0
    vein_rel_width_mean: float = 0.0
    vein_rel_width_std: float = 0.0
    dominant_angle_mean: float = 0.0
    angle_coherence_mean: float = 0.0
    
    spot_coverage_mean: float = 0.0
    spot_coverage_std: float = 0.0
    spot_cv_mean: float = 0.0
    
    luminance_mean: float = 0.0
    luminance_var: float = 0.0
    chroma_mean: float = 0.0
    
    cluster_proportions: List[float] = field(default_factory=list)
    cluster_distances: List[float] = field(default_factory=list)
    
    color_debug: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "avg_stone_area_ratio": self.avg_stone_area_ratio,
            "avg_blur_score": self.avg_blur_score,
            "avg_exposure_score": self.avg_exposure_score,
            "avg_glare_score": self.avg_glare_score,
            "avg_glare_ratio": self.avg_glare_ratio,
            "avg_reflection_ratio": self.avg_reflection_ratio,
            "vein_coverage_mean": self.vein_coverage_mean,
            "vein_coverage_std": self.vein_coverage_std,
            "vein_rel_width_mean": self.vein_rel_width_mean,
            "vein_rel_width_std": self.vein_rel_width_std,
            "dominant_angle_mean": self.dominant_angle_mean,
            "angle_coherence_mean": self.angle_coherence_mean,
            "spot_coverage_mean": self.spot_coverage_mean,
            "spot_coverage_std": self.spot_coverage_std,
            "spot_cv_mean": self.spot_cv_mean,
            "luminance_mean": self.luminance_mean,
            "luminance_var": self.luminance_var,
            "chroma_mean": self.chroma_mean,
            "cluster_proportions": self.cluster_proportions,
            "cluster_distances": self.cluster_distances,
            "color_debug": self.color_debug
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "DebugMetrics":
        return cls(
            avg_stone_area_ratio=data.get("avg_stone_area_ratio", 0.0),
            avg_blur_score=data.get("avg_blur_score", 0.0),
            avg_exposure_score=data.get("avg_exposure_score", 0.0),
            avg_glare_score=data.get("avg_glare_score", 0.0),
            avg_glare_ratio=data.get("avg_glare_ratio", 0.0),
            avg_reflection_ratio=data.get("avg_reflection_ratio", 0.0),
            vein_coverage_mean=data.get("vein_coverage_mean", 0.0),
            vein_coverage_std=data.get("vein_coverage_std", 0.0),
            vein_rel_width_mean=data.get("vein_rel_width_mean", 0.0),
            vein_rel_width_std=data.get("vein_rel_width_std", 0.0),
            dominant_angle_mean=data.get("dominant_angle_mean", 0.0),
            angle_coherence_mean=data.get("angle_coherence_mean", 0.0),
            spot_coverage_mean=data.get("spot_coverage_mean", 0.0),
            spot_coverage_std=data.get("spot_coverage_std", 0.0),
            spot_cv_mean=data.get("spot_cv_mean", 0.0),
            luminance_mean=data.get("luminance_mean", 0.0),
            luminance_var=data.get("luminance_var", 0.0),
            chroma_mean=data.get("chroma_mean", 0.0),
            cluster_proportions=data.get("cluster_proportions", []),
            cluster_distances=data.get("cluster_distances", []),
            color_debug=data.get("color_debug", {})
        )


@dataclass
class FingerprintV3New:
    """Stock-level fingerprint with Design classification and conditional branches."""
    company_stone_id: str
    
    base_color: Optional[str] = None
    base_color_conf: float = 0.0
    
    colors_in_stone: List[str] = field(default_factory=list)
    colors_in_stone_conf: float = 0.0
    
    tonality: Optional[str] = None
    tonality_conf: float = 0.0
    
    design: Optional[str] = None
    design_conf: float = 0.0
    
    vein_color: List[str] = field(default_factory=list)
    vein_color_conf: float = 0.0
    vein_direction: Optional[str] = None
    vein_direction_conf: float = 0.0
    vein_distribution: Optional[str] = None
    vein_distribution_conf: float = 0.0
    vein_thickness: Optional[str] = None
    vein_thickness_conf: float = 0.0
    
    spot_color: List[str] = field(default_factory=list)
    spot_color_conf: float = 0.0
    spot_distribution: Optional[str] = None
    spot_distribution_conf: float = 0.0
    
    representative_images: RepresentativeImages = field(default_factory=RepresentativeImages)
    debug_metrics: DebugMetrics = field(default_factory=DebugMetrics)
    
    image_count: int = 0
    usable_image_count: int = 0
    
    ai_base_color: Optional[str] = None
    ai_base_color_conf: float = 0.0
    ai_base_color_method: Optional[str] = None
    
    ai_design: Optional[str] = None
    ai_design_conf: float = 0.0
    ai_design_method: Optional[str] = None
    
    ai_tonality: Optional[str] = None
    ai_tonality_conf: float = 0.0
    ai_tonality_method: Optional[str] = None
    
    has_manual_base_color: bool = False
    has_manual_design: bool = False
    has_manual_tonality: bool = False
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def apply_applicability_rules(self):
        """Clear fields that don't apply based on Design and BaseColor."""
        if self.design != Design.VEINY.value:
            self.vein_color = []
            self.vein_color_conf = 0.0
            self.vein_direction = None
            self.vein_direction_conf = 0.0
            self.vein_distribution = None
            self.vein_distribution_conf = 0.0
            self.vein_thickness = None
            self.vein_thickness_conf = 0.0
        
        if self.design != Design.SPOTTY.value:
            self.spot_color = []
            self.spot_color_conf = 0.0
            self.spot_distribution = None
            self.spot_distribution_conf = 0.0
        
        if self.base_color != BaseColor.EXOTIC.value:
            self.colors_in_stone = []
            self.colors_in_stone_conf = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "company_stone_id": self.company_stone_id,
            "base_color": self.base_color,
            "base_color_conf": self.base_color_conf,
            "colors_in_stone": self.colors_in_stone,
            "colors_in_stone_conf": self.colors_in_stone_conf,
            "tonality": self.tonality,
            "tonality_conf": self.tonality_conf,
            "design": self.design,
            "design_conf": self.design_conf,
            "vein_color": self.vein_color,
            "vein_color_conf": self.vein_color_conf,
            "vein_direction": self.vein_direction,
            "vein_direction_conf": self.vein_direction_conf,
            "vein_distribution": self.vein_distribution,
            "vein_distribution_conf": self.vein_distribution_conf,
            "vein_thickness": self.vein_thickness,
            "vein_thickness_conf": self.vein_thickness_conf,
            "spot_color": self.spot_color,
            "spot_color_conf": self.spot_color_conf,
            "spot_distribution": self.spot_distribution,
            "spot_distribution_conf": self.spot_distribution_conf,
            "representative_images": self.representative_images.to_dict(),
            "debug_metrics": self.debug_metrics.to_dict(),
            "image_count": self.image_count,
            "usable_image_count": self.usable_image_count,
            "ai_base_color": self.ai_base_color,
            "ai_base_color_conf": self.ai_base_color_conf,
            "ai_base_color_method": self.ai_base_color_method,
            "ai_design": self.ai_design,
            "ai_design_conf": self.ai_design_conf,
            "ai_design_method": self.ai_design_method,
            "ai_tonality": self.ai_tonality,
            "ai_tonality_conf": self.ai_tonality_conf,
            "ai_tonality_method": self.ai_tonality_method,
            "has_manual_base_color": self.has_manual_base_color,
            "has_manual_design": self.has_manual_design,
            "has_manual_tonality": self.has_manual_tonality,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "FingerprintV3New":
        return cls(
            company_stone_id=data["company_stone_id"],
            base_color=data.get("base_color"),
            base_color_conf=data.get("base_color_conf", 0.0),
            colors_in_stone=data.get("colors_in_stone", []),
            colors_in_stone_conf=data.get("colors_in_stone_conf", 0.0),
            tonality=data.get("tonality"),
            tonality_conf=data.get("tonality_conf", 0.0),
            design=data.get("design"),
            design_conf=data.get("design_conf", 0.0),
            vein_color=data.get("vein_color", []),
            vein_color_conf=data.get("vein_color_conf", 0.0),
            vein_direction=data.get("vein_direction"),
            vein_direction_conf=data.get("vein_direction_conf", 0.0),
            vein_distribution=data.get("vein_distribution"),
            vein_distribution_conf=data.get("vein_distribution_conf", 0.0),
            vein_thickness=data.get("vein_thickness"),
            vein_thickness_conf=data.get("vein_thickness_conf", 0.0),
            spot_color=data.get("spot_color", []),
            spot_color_conf=data.get("spot_color_conf", 0.0),
            spot_distribution=data.get("spot_distribution"),
            spot_distribution_conf=data.get("spot_distribution_conf", 0.0),
            representative_images=RepresentativeImages.from_dict(data.get("representative_images", {})),
            debug_metrics=DebugMetrics.from_dict(data.get("debug_metrics", {})),
            image_count=data.get("image_count", 0),
            usable_image_count=data.get("usable_image_count", 0),
            ai_base_color=data.get("ai_base_color"),
            ai_base_color_conf=data.get("ai_base_color_conf", 0.0),
            ai_base_color_method=data.get("ai_base_color_method"),
            ai_design=data.get("ai_design"),
            ai_design_conf=data.get("ai_design_conf", 0.0),
            ai_design_method=data.get("ai_design_method"),
            ai_tonality=data.get("ai_tonality"),
            ai_tonality_conf=data.get("ai_tonality_conf", 0.0),
            ai_tonality_method=data.get("ai_tonality_method"),
            has_manual_base_color=data.get("has_manual_base_color", False),
            has_manual_design=data.get("has_manual_design", False),
            has_manual_tonality=data.get("has_manual_tonality", False),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )

