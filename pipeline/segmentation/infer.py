"""
The actual segmentation inference: YOLO detection -> MedSAM segmentation.

YOLO gives us a rough bounding box around the wound. MedSAM then uses
that box as a prompt to produce a precise pixel-level mask.
"""

import cv2
import numpy as np
import torch

from pipeline.config import (
    SEG_INPUT_SIZE, SEG_THRESHOLD, YOLO_CONF, BBOX_PAD,
    MASKS_DIR, OVERLAYS_DIR, CROPS_DIR
)
from pipeline.schemas import SegmentationResult
from .preprocess import load_image_rgb, resize_to_1024
from .utils import postprocess_mask, extract_wound_crop, create_overlay, compute_area


@torch.no_grad()   # inference only — no need to track gradients
def segment_wound(img_rgb, yolo_model, sam_model, device,
                  threshold=SEG_THRESHOLD, pad=BBOX_PAD):
    """
    Run the full detection -> segmentation pipeline on one image.

    Args:
        img_rgb:    H x W x 3 numpy array in RGB
        yolo_model: loaded YOLO detector
        sam_model:  loaded MedSAM model
        device:     torch device
        threshold:  probability cutoff to binarize the mask
        pad:        pixels of padding to add around the YOLO bbox

    Returns:
        mask:       1024x1024 binary mask (uint8)
        yolo_conf:  YOLO confidence (0.0 if YOLO missed and we fell back)
        bbox:       [x1, y1, x2, y2] the bbox we used (in 1024 space)
    """
    H, W = img_rgb.shape[:2]
    yolo_conf = 0.0

    # ── Step 1: YOLO finds the wound bbox ────────────────
    results = yolo_model(img_rgb, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        # YOLO didn't find anything — fall back to assuming wound is centered.
        # Better than nothing but not great, this usually means the image is
        # too far out of distribution for the detector.
        print("  [seg] YOLO found no wound — falling back to centre crop")
        bbox = np.array([W // 4, H // 4, 3 * W // 4, 3 * H // 4])
    else:
        # Pick the highest-confidence detection
        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy()
        yolo_conf = float(best.conf.item())

        # Scale the bbox from the original image size to 1024 (MedSAM's input size)
        scale_x = SEG_INPUT_SIZE / W
        scale_y = SEG_INPUT_SIZE / H
        # Expand the bbox a bit with padding so MedSAM has some context around the wound
        bbox = np.array([
            max(0, x1 * scale_x - pad),
            max(0, y1 * scale_y - pad),
            min(SEG_INPUT_SIZE, x2 * scale_x + pad),
            min(SEG_INPUT_SIZE, y2 * scale_y + pad)
        ])
        print(f"  [seg] YOLO conf={yolo_conf:.3f} bbox={bbox.astype(int)}")

    # ── Step 2: MedSAM segments the wound using the bbox as prompt ──
    img_1024 = resize_to_1024(img_rgb)
    # Convert to tensor: HWC uint8 -> 1CHW float in [0, 1]
    img_t = (torch.from_numpy(img_1024)
             .permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0)
    bbox_t = torch.as_tensor(bbox, dtype=torch.float32).unsqueeze(0).to(device)

    # MedSAM is made of three parts: image encoder, prompt encoder, mask decoder
    emb = sam_model.image_encoder(img_t)
    sparse, dense = sam_model.prompt_encoder(
        points=None, boxes=bbox_t, masks=None
    )
    logits, _ = sam_model.mask_decoder(
        image_embeddings=emb,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False   # only want one mask
    )

    # Sigmoid -> probability map -> resize -> threshold
    prob = cv2.resize(
        torch.sigmoid(logits).squeeze().cpu().numpy(),
        (SEG_INPUT_SIZE, SEG_INPUT_SIZE)
    )
    mask = (prob > threshold).astype(np.uint8)

    # Clean up stray pixels and smooth edges
    mask = postprocess_mask(mask)

    return mask, yolo_conf, bbox.astype(int).tolist()


def run_segmentation(image_path, yolo_model, sam_model, device,
                     image_id="image", save=True) -> SegmentationResult:
    """
    High-level entry point. Loads the image, runs segmentation,
    optionally saves all the outputs to disk, and returns a structured result.
    """
    img_rgb = load_image_rgb(image_path)
    mask, yolo_conf, bbox = segment_wound(img_rgb, yolo_model, sam_model, device)

    area_px = compute_area(mask)
    mask_path = ""
    overlay_path = ""
    crop_path = ""

    if save:
        # Wrapping in try/except because OneDrive sometimes locks files
        # while syncing — we don't want that to crash the whole pipeline.
        try:
            # 1. Binary mask as PNG (multiplied by 255 so it's visible)
            mask_path = str(MASKS_DIR / f"{image_id}_mask.png")
            cv2.imwrite(mask_path, mask * 255)

            # 2. Green contour drawn on the original image
            overlay = create_overlay(img_rgb, mask)
            overlay_path = str(OVERLAYS_DIR / f"{image_id}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # 3. Just the wound, cropped and background-zeroed
            crop = extract_wound_crop(img_rgb, mask)
            crop_path = str(CROPS_DIR / f"{image_id}_crop.png")
            cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        except PermissionError as e:
            print(f"  [seg] Warning: could not save some files: {e}")

    return SegmentationResult(
        image_id=image_id,
        mask_path=mask_path,
        overlay_path=overlay_path,
        crop_path=crop_path,
        area_px=area_px,
        bbox=bbox,
        yolo_conf=yolo_conf
    )
