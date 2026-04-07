# tools/general_perception/opacity_segmenter.py
# Models: TorchXRayVision DenseNet121 (presence) + Grounding DINO (bbox) + MedSAM (mask)
from tools.base import BaseTool


class OpacitySegmenter(BaseTool):
    """
    Three-stage pipeline:
      1. TorchXRayVision DenseNet121  — pathology probability score
      2. Grounding DINO               — locate finding by text query → bbox
      3. MedSAM ViT-B                 — precise segmentation from bbox
    MedSAM beats SAM2 on 2D X-ray by ~7.5 Dice points (arXiv:2408.03322).
    """
    id = "opacity_segmenter"
    name = "Opacity Segmenter"
    category = "General Perception"
    description = (
        "Detects and segments opacities, consolidations, effusions and nodules. "
        "Uses TorchXRayVision for classification, Grounding DINO for localisation, "
        "and MedSAM for precise zero-shot segmentation."
    )
    input_format = "CXR image path + args: {finding: str}"
    output_format = (
        '{ "finding": str, "detected": bool, "confidence": float, '
        '"bbox": [x1,y1,x2,y2], "mask_area_pct": float, '
        '"location": str, "distribution": str }'
    )
    example = "Use to segment and characterise any pulmonary opacity or effusion"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict, gdino_detect, medsam_segment
        from utils.model_loader import txrv_segment
        from utils.spatial_geometry import classify_distribution, mask_centroid

        finding = args.get("finding", "opacity").lower().strip()
        img = load_image_np(image_path)
        h, w = img.shape[:2]

        # ── Step 1: TorchXRayVision — pathology probability ────────────────────
        probs = txrv_predict(img, config.TXRV_WEIGHTS)
        confidence = 0.0
        for label, p in probs.items():
            if label and finding.replace(" ", "") in label.lower().replace(" ", ""):
                confidence = max(confidence, p)
        if confidence == 0.0:
            top = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            confidence = top[0][1] if top else 0.0
        detected = confidence >= config.TXRV_THRESHOLD

        # ── Step 2: Grounding DINO — get bounding box ──────────────────────────
        bbox = None
        if image_path:
            dets = gdino_detect(img, [finding],
                                box_threshold=config.GDINO_BOX_THRESHOLD,
                                text_threshold=config.GDINO_TEXT_THRESHOLD)
            if dets:
                bbox = dets[0]["bbox"]

        # ── Step 3: MedSAM — precise segmentation from bbox ───────────────────
        finding_mask = np.zeros((h, w), np.uint8)
        if bbox and image_path and config.MEDSAM_CHECKPOINT.exists():
            try:
                finding_mask = medsam_segment(img, bbox, config.MEDSAM_CHECKPOINT)
            except Exception:
                x1, y1, x2, y2 = bbox
                finding_mask[y1:y2, x1:x2] = 1
        elif bbox:
            x1, y1, x2, y2 = bbox
            finding_mask[y1:y2, x1:x2] = 1

        # ── Spatial stats ──────────────────────────────────────────────────────
        lung_masks = txrv_segment(img)
        left_lung  = lung_masks.get("Left Lung",  np.zeros((h, w), np.uint8))
        right_lung = lung_masks.get("Right Lung", np.zeros((h, w), np.uint8))
        total_lung = int(left_lung.sum() + right_lung.sum()) or (h * w)
        mask_area_pct = round(float(finding_mask.sum()) / total_lung * 100, 1)

        cy, cx = mask_centroid(finding_mask)
        h_pos = "right" if cx < w * 0.5 else "left"
        v_pos = "upper" if cy < h / 3 else ("lower" if cy > 2 * h / 3 else "middle")
        location = f"{v_pos} {h_pos} lung zone"

        dist = classify_distribution(finding_mask, left_lung, right_lung,
                                     config.PIXEL_SPACING_MM,
                                     config.PERIPHERAL_FRACTION)
        return {
            "finding": finding,
            "detected": detected,
            "confidence": round(confidence, 3),
            "bbox": bbox,
            "mask_area_pct": mask_area_pct,
            "location": location,
            "distribution": dist["full_description"],
            "laterality": dist["laterality"],
            "pattern": dist["pattern"],
        }
