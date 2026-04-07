# tools/geometry/tube_line_localizer.py
# Models: Grounding DINO (device detection) + MedSAM (segmentation) + spatial geometry
from tools.base import BaseTool


class TubeLineLocalizer(BaseTool):
    """
    Detects and localises medical devices using:
      1. Grounding DINO  — open-vocab detection of any device by text name
                           (no retraining needed for new device types)
      2. MedSAM ViT-B    — precise device segmentation from GDINO bbox
      3. Extremal-point  — tip extraction from the segmentation mask
      4. Grounding DINO  — locate reference structure (carina, diaphragm)
      5. Geometry        — tip-to-reference distance + clinical zone
    """
    id = "tube_line_localizer"
    name = "Tube & Line Localizer"
    category = "Geometry & Measurement"
    description = (
        "Detects ETT, NGT, CVC, pacemaker leads, chest drains. "
        "Uses Grounding DINO (open-vocabulary, no retraining needed) + "
        "MedSAM segmentation + geometric tip localisation."
    )
    input_format = "CXR image path + args: {device: str (optional)}"
    output_format = (
        '{ "devices": [{ "type": str, "detected": bool, '
        '"tip_location": str, "zone": str, '
        '"distance_from_reference_mm": float, "correct_position": bool }] }'
    )
    example = "Use post-procedure to verify ETT or CVC placement"

    CORRECT_ZONES = {
        "endotracheal tube": "above",
        "ett": "above",
        "nasogastric tube": "below",
        "ngt": "below",
        "central venous catheter": "at the level of",
        "cvc": "at the level of",
        "chest tube": "above",
    }

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import gdino_detect, medsam_segment
        from utils.spatial_geometry import find_device_tip, tip_vs_reference

        target = args.get("device", "").lower().strip()
        img = load_image_np(image_path)
        h, w = img.shape[:2]

        # ── Step 1: Grounding DINO device detection ────────────────────────────
        queries = [target] if target else [
            "endotracheal tube", "nasogastric tube",
            "central venous catheter", "chest tube",
        ]
        dets = gdino_detect(img, queries,
                            box_threshold=config.GDINO_BOX_THRESHOLD,
                            text_threshold=config.GDINO_TEXT_THRESHOLD)

        if not dets:
            return {"devices": [{"type": target or "device", "detected": False,
                                 "tip_location": "not detected", "zone": "n/a",
                                 "distance_from_reference_mm": 0.0,
                                 "correct_position": False}]}

        results = []
        for det in dets[:3]:
            dtype = det["label"]
            bbox  = det["bbox"]

            # ── Step 2: MedSAM segmentation ────────────────────────────────────
            device_mask = np.zeros((h, w), np.uint8)
            if image_path and config.MEDSAM_CHECKPOINT.exists():
                try:
                    device_mask = medsam_segment(img, bbox, config.MEDSAM_CHECKPOINT)
                except Exception:
                    x1, y1, x2, y2 = bbox
                    device_mask[y1:y2, x1:x2] = 1
            else:
                x1, y1, x2, y2 = bbox
                device_mask[y1:y2, x1:x2] = 1

            # ── Step 3: Tip extraction ─────────────────────────────────────────
            tip_yx = find_device_tip(device_mask, dtype)

            # ── Step 4: Reference structure via Grounding DINO ─────────────────
            ref_name = "carina" if "tube" in dtype else "superior vena cava"
            ref_dets = gdino_detect(img, [ref_name],
                                    box_threshold=config.GDINO_BOX_THRESHOLD,
                                    text_threshold=config.GDINO_TEXT_THRESHOLD)
            ref_mask = np.zeros((h, w), np.uint8)
            if ref_dets and config.MEDSAM_CHECKPOINT.exists():
                try:
                    ref_mask = medsam_segment(img, ref_dets[0]["bbox"],
                                              config.MEDSAM_CHECKPOINT)
                except Exception:
                    rx1, ry1, rx2, ry2 = ref_dets[0]["bbox"]
                    ref_mask[ry1:ry2, rx1:rx2] = 1
            else:
                # anatomical prior for carina (~55-60% down, near centre)
                cy, cx = int(h * 0.58), int(w * 0.5)
                ref_mask[cy - 10:cy + 10, cx - 25:cx + 25] = 1

            # ── Step 5: Tip-to-reference distance ──────────────────────────────
            tip_rel = tip_vs_reference(tip_yx, ref_mask, config.PIXEL_SPACING_MM)
            expected = self.CORRECT_ZONES.get(dtype.lower(), "above")
            correct  = tip_rel["zone"] == expected

            results.append({
                "type": dtype,
                "detected": True,
                "tip_location": tip_rel["text"],
                "zone": tip_rel["zone"],
                "distance_from_reference_mm": tip_rel["dist_mm"],
                "correct_position": correct,
            })

        return {"devices": results}
