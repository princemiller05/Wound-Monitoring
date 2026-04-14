"""
orchestrator.py
===============
The top-level pipeline that chains everything together:

    Image -> Segmentation -> Tissue Classification -> Healing Prediction

Usage:
    from pipeline.orchestrator import DFUPipeline
    pipe = DFUPipeline()
    result = pipe.run("data/sample_inputs/woundtst.jpg", case_id="CASE_001")

The DFUPipeline class loads all three models once (which is slow),
then each call to run() is quick. If you're only processing one image,
the convenience function run_pipeline() at the bottom also works.
"""

import cv2
import numpy as np

from pipeline.config import ensure_output_dirs
from pipeline.schemas import PipelineResult

# Segmentation module (Prince's work)
from pipeline.segmentation.model import load_segmentation_models
from pipeline.segmentation.infer import run_segmentation
from pipeline.segmentation.preprocess import load_image_rgb

# Tissue classification module (Subham's work)
from pipeline.tissue.model import load_tissue_model
from pipeline.tissue.infer import run_tissue_classification

# Healing prediction module (Varsha's work)
from pipeline.healing.model import load_healing_model
from pipeline.healing.infer import run_healing_prediction


class DFUPipeline:
    """
    Main pipeline class. Loads all three models up-front so subsequent
    predictions are fast. Think of it like turning on the machine once
    and then using it for as many images as you want.
    """

    def __init__(self, device=None):
        """
        Load everything: YOLO, MedSAM, ResNet18, XGBoost.
        device=None means "use GPU if available, else CPU".
        """
        print("=" * 50)
        print("Loading DFU Pipeline models...")
        print("=" * 50)

        # Segmentation models — YOLO for detection, MedSAM for segmentation
        self.detector, self.sam_model, self.device = load_segmentation_models(device)

        # Tissue classifier — ResNet18 fine-tuned on 3 tissue classes
        self.tissue_model, _ = load_tissue_model(self.device)

        # Healing predictor — XGBoost trained on longitudinal features
        self.healing_model = load_healing_model()

        # Make sure all output folders exist so we can write to them later
        ensure_output_dirs()

        print("=" * 50)
        print("Pipeline ready!")
        print("=" * 50)

    def run(self, image_path, case_id="CASE_001", image_id=None,
            simulate_mode="auto", seed=42, save=True) -> PipelineResult:
        """
        Run the full pipeline on a single wound image.

        Args:
            image_path:     path to the wound image (png or jpg)
            case_id:        patient/case identifier (e.g. "CASE_001")
            image_id:       specific image ID — defaults to "<case_id>_DAY0"
            simulate_mode:  "auto" | "healing" | "non_healing" (for synthetic progression)
            seed:           random seed so results are reproducible
            save:           whether to save all the intermediate files

        Returns:
            PipelineResult containing segmentation, tissue, and healing results.
        """
        # If no image_id given, default to day 0 of this case
        if image_id is None:
            image_id = f"{case_id}_DAY0"

        print(f"\n{'='*50}")
        print(f"Running pipeline: {case_id}")
        print(f"Image: {image_path}")
        print(f"{'='*50}")

        # ── Step 1: Segmentation ─────────────────────────
        # YOLO finds where the wound is, then MedSAM does the precise outline.
        print("\n[1/3] Segmentation...")
        seg_result = run_segmentation(
            image_path, self.detector, self.sam_model, self.device,
            image_id=image_id, save=save
        )
        print(f"  Area: {seg_result.area_px} px | YOLO conf: {seg_result.yolo_conf:.3f}")

        # Load the image again so the tissue module has access to it.
        # (We could pass it around in memory, but this is cleaner.)
        img_rgb = load_image_rgb(image_path)

        # Reload the mask we just saved. If for some reason it wasn't saved
        # (e.g. save=False), re-run segmentation to get the mask in memory.
        mask = cv2.imread(seg_result.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = (mask > 127).astype(np.uint8)   # binarize: anything white-ish is wound
        else:
            from pipeline.segmentation.infer import segment_wound
            mask, _, _ = segment_wound(
                img_rgb, self.detector, self.sam_model, self.device
            )

        # ── Step 2: Tissue Classification ────────────────
        # Slide a window over the wound region and classify each patch.
        print("\n[2/3] Tissue Classification...")
        tissue_result = run_tissue_classification(
            img_rgb, mask, self.tissue_model, self.device,
            image_id=image_id, save=save
        )

        # ── Step 3: Healing Prediction ───────────────────
        # Since we only have one image (Day 0), we generate a synthetic
        # progression forward in time and ask XGBoost to judge it.
        print("\n[3/3] Healing Prediction...")
        healing_result = run_healing_prediction(
            base_mask=mask,
            healing_model=self.healing_model,
            case_id=case_id,
            tissue_result=tissue_result,
            simulate_mode=simulate_mode,
            seed=seed,
            save=save,
        )

        # ── Pack results into one object ─────────────────
        saved_files = {
            "mask":       seg_result.mask_path,
            "overlay":    seg_result.overlay_path,
            "crop":       seg_result.crop_path,
            "tissue_map": tissue_result.tissue_map_path,
        }

        result = PipelineResult(
            case_id=case_id,
            image_id=image_id,
            segmentation=seg_result,
            tissue=tissue_result,
            healing=healing_result,
            saved_files=saved_files
        )

        # ── Pretty-print a summary so the user can see the results ──
        print(f"\n{'='*50}")
        print(f"RESULTS — {case_id}")
        print(f"{'='*50}")
        print(f"  Wound area:      {seg_result.area_px} px")
        print(f"  Tissue:          G={tissue_result.granulation_pct}% "
              f"S={tissue_result.slough_pct}% N={tissue_result.necrosis_pct}%")
        print(f"  Healing prob:    {healing_result.healing_probability}")
        print(f"  Predicted:       {healing_result.predicted_label}")
        print(f"  Rule baseline:   {healing_result.rule_label}")
        print(f"{'='*50}\n")

        return result


# ─── Convenience function ────────────────────────────────────────
# For quick one-off runs. If you're processing multiple images in a loop,
# prefer creating a DFUPipeline instance once (so models only load once).

def run_pipeline(image_path, case_id="CASE_001", simulate_mode="auto",
                 seed=42, device=None) -> PipelineResult:
    """One-shot helper: loads models, runs pipeline, returns results."""
    pipe = DFUPipeline(device=device)
    return pipe.run(image_path, case_id=case_id,
                    simulate_mode=simulate_mode, seed=seed)
