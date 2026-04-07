# tools/general_perception/opacity_segmenter.py
# Models: TorchXRayVision DenseNet121 (presence gate) + Grounding DINO (bbox) + MedSAM (mask)
from tools.base import BaseTool


class OpacitySegmenter(BaseTool):
    """
    Three-stage pipeline with a hard TXRV presence gate:

      Stage 1 — TorchXRayVision DenseNet121
                Primary presence signal.  TXRV trained on CheXpert+MIMIC+NIH
                is far more reliable for CXR presence detection than GDINO
                zero-shot.  If TXRV confidence < threshold the finding is
                reported as absent and stages 2–3 are skipped entirely.

      Stage 2 — Grounding DINO  (only if TXRV says finding is present)
                Localises the finding by text query to produce a bbox.
                Uses a higher threshold than config default to reduce
                false-positive boxes.

      Stage 3 — MedSAM ViT-B  (only if GDINO found a credible bbox)
                Precise zero-shot segmentation from the GDINO bbox.
                Skipped if GDINO returned no confident detection.
    """

    id = "opacity_segmenter"
    name = "Opacity Segmenter"
    category = "General Perception"
    description = (
        "Detects and segments opacities, consolidations, effusions and nodules. "
        "TorchXRayVision provides the primary presence gate — GDINO and MedSAM "
        "only run when TXRV confidence exceeds the detection threshold."
    )
    input_format = "CXR image path + args: {finding: str}"
    output_format = (
        '{ "finding": str, "detected": bool, "confidence": float, '
        '"txrv_probability": float, '
        '"bbox": [x1,y1,x2,y2] or null, "mask_area_pct": float, '
        '"location": str, "distribution": str }'
    )
    example = "Use to detect and characterise any pulmonary opacity or effusion"

    # TXRV label fragments that map to common finding names
    _LABEL_MAP = {
        "pleural effusion": ["pleural effusion", "effusion"],
        "effusion":         ["pleural effusion", "effusion"],
        "pneumothorax":     ["pneumothorax"],
        "consolidation":    ["consolidation"],
        "atelectasis":      ["atelectasis"],
        "opacity":          ["lung opacity", "opacity", "infiltration"],
        "infiltrate":       ["infiltration", "infiltrate"],
        "edema":            ["edema"],
        "cardiomegaly":     ["cardiomegaly"],
        "nodule":           ["nodule"],
        "mass":             ["mass"],
        "pneumonia":        ["pneumonia", "consolidation", "infiltration"],
        "emphysema":        ["emphysema"],
        "fibrosis":         ["fibrosis"],
        "fracture":         ["fracture"],
    }

    def _txrv_confidence(self, probs: dict, finding: str) -> float:
        """
        Return the highest TXRV probability for the given finding name.
        Uses the label map for reliable matching; falls back to substring.
        """
        finding_lower = finding.lower().strip()

        # Try label map first
        targets = self._LABEL_MAP.get(finding_lower, [])
        if not targets:
            # build targets from any key that is a substring of the finding
            for key, vals in self._LABEL_MAP.items():
                if key in finding_lower or finding_lower in key:
                    targets.extend(vals)

        best = 0.0
        for label, p in probs.items():
            if not label:
                continue
            label_lower = label.lower()
            # Match against mapped targets
            for t in targets:
                if t in label_lower:
                    best = max(best, p)
                    break
            # Fallback: direct substring
            if finding_lower.replace(" ", "") in label_lower.replace(" ", ""):
                best = max(best, p)
        return best

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict, txrv_segment
        from utils.spatial_geometry import classify_distribution, mask_centroid

        finding = args.get("finding", "opacity").lower().strip()
        img = load_image_np(image_path)
        h, w = img.shape[:2]

        # ── Stage 1: TorchXRayVision — primary presence gate ──────────────────
        probs = txrv_predict(img, config.TXRV_WEIGHTS)
        txrv_prob = self._txrv_confidence(probs, finding)
        detected = txrv_prob >= config.TXRV_THRESHOLD

        # Return early if TXRV says absent — do not run GDINO/MedSAM
        # to avoid false positive localisation on a negative finding
        if not detected:
            return {
                "finding":        finding,
                "detected":       False,
                "confidence":     round(txrv_prob, 3),
                "txrv_probability": round(txrv_prob, 3),
                "bbox":           None,
                "mask_area_pct":  0.0,
                "location":       "not detected",
                "distribution":   "not applicable",
                "laterality":     "not applicable",
                "pattern":        "not applicable",
                "note": (
                    f"TXRV probability {txrv_prob:.1%} is below threshold "
                    f"{config.TXRV_THRESHOLD:.1%}. "
                    "Grounding DINO and MedSAM skipped to avoid false positives."
                ),
            }

        # ── Stage 2: Grounding DINO — localise finding (only if TXRV positive) ─
        bbox = None
        gdino_score = 0.0
        if image_path:
            try:
                from utils.model_loader import gdino_detect
                # Use stricter threshold than default when running post-TXRV
                # to ensure the bbox corresponds to the actual finding
                dets = gdino_detect(
                    img, [finding],
                    box_threshold=max(config.GDINO_BOX_THRESHOLD, 0.30),
                    text_threshold=max(config.GDINO_TEXT_THRESHOLD, 0.20),
                )
                if dets:
                    bbox = dets[0]["bbox"]
                    gdino_score = dets[0]["score"]
            except Exception:
                pass   # GDINO unavailable — continue without bbox

        # ── Stage 3: MedSAM — segment from bbox (only if GDINO found a bbox) ──
        finding_mask = np.zeros((h, w), np.uint8)
        if bbox and image_path and config.MEDSAM_CHECKPOINT.exists():
            try:
                from utils.model_loader import medsam_segment
                finding_mask = medsam_segment(img, bbox, config.MEDSAM_CHECKPOINT)
            except Exception:
                x1, y1, x2, y2 = bbox
                finding_mask[y1:y2, x1:x2] = 1
        elif bbox:
            x1, y1, x2, y2 = bbox
            finding_mask[y1:y2, x1:x2] = 1

        # ── Spatial statistics ─────────────────────────────────────────────────
        lung_masks = txrv_segment(img)
        left_lung  = lung_masks.get("Left Lung",  np.zeros((h, w), np.uint8))
        right_lung = lung_masks.get("Right Lung", np.zeros((h, w), np.uint8))
        total_lung = int(left_lung.sum() + right_lung.sum()) or (h * w)
        mask_area_pct = round(float(finding_mask.sum()) / total_lung * 100, 1)

        cy, cx = mask_centroid(finding_mask) if finding_mask.sum() > 0 \
                 else (h / 2, w / 2)
        h_pos  = "right" if cx < w * 0.5 else "left"
        v_pos  = "upper" if cy < h / 3 else ("lower" if cy > 2 * h / 3 else "middle")
        location = f"{v_pos} {h_pos} lung zone" if finding_mask.sum() > 0 \
                   else "detected but not localised (GDINO found no bbox)"

        dist = classify_distribution(
            finding_mask, left_lung, right_lung,
            config.PIXEL_SPACING_MM, config.PERIPHERAL_FRACTION,
        ) if finding_mask.sum() > 0 else {
            "full_description": "localisation unavailable",
            "laterality": "unknown", "pattern": "unknown",
        }

        return {
            "finding":          finding,
            "detected":         True,
            "confidence":       round(txrv_prob, 3),
            "txrv_probability": round(txrv_prob, 3),
            "gdino_score":      round(gdino_score, 3),
            "bbox":             bbox,
            "mask_area_pct":    mask_area_pct,
            "location":         location,
            "distribution":     dist["full_description"],
            "laterality":       dist["laterality"],
            "pattern":          dist["pattern"],
        }