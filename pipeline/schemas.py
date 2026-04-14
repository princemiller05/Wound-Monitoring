"""
schemas.py
==========
Dataclasses that define what each module returns.

Why have these? Because the three modules (segmentation, tissue,
healing) are being developed somewhat separately — having a fixed
contract for their outputs means we can swap out the internals
of any module without breaking the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SegmentationResult:
    """What the segmentation module hands back."""
    image_id: str
    mask_path: str                 # path to the saved binary mask (PNG)
    overlay_path: str              # contour drawn on the original image
    crop_path: str                 # wound cut out, background zeroed
    area_px: int                   # wound size in pixels (at 1024x1024)
    bbox: List[int] = field(default_factory=list)   # [x1, y1, x2, y2]
    yolo_conf: float = 0.0        # YOLO's confidence in the detection


@dataclass
class TissueResult:
    """What the tissue classification module hands back."""
    image_id: str
    granulation_pct: float         # % healthy granulation tissue
    slough_pct: float              # % slough (dead skin)
    necrosis_pct: float            # % necrotic (dead) tissue
    tissue_map_path: str = ""      # visualization with colored patches
    total_patches: int = 0        # how many patches we classified


@dataclass
class HealingResult:
    """What the healing prediction module hands back."""
    case_id: str
    healing_probability: float     # XGBoost prob between 0 and 1
    predicted_label: str           # "healing" or "non_healing"
    rule_label: str                # rule-based label for comparison
    key_factors: List[str] = field(default_factory=list)   # top features
    features: Optional[dict] = None   # the feature row we fed to XGBoost


@dataclass
class PipelineResult:
    """Everything combined — one object containing all three module outputs."""
    case_id: str
    image_id: str
    segmentation: SegmentationResult
    tissue: TissueResult
    healing: HealingResult
    saved_files: dict = field(default_factory=dict)   # quick reference to output paths
