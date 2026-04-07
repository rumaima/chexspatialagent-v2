# tools/spatial_analysis/trachea_mediastinum_analyzer.py
# Models: TorchXRayVision PSPNet (trachea + lung segmentation) + Grounding DINO
from tools.base import BaseTool


class TracheaMediastinumAnalyzer(BaseTool):
    """
    Detects tracheal deviation and mediastinal widening.
    Uses PSPNet trachea/lung masks for midline geometry;
    Grounding DINO is used when a specific reference structure is needed.
    """
    id = "trachea_mediastinum_analyzer"
    name = "Trachea & Mediastinum Analyzer"
    category = "Spatial Analysis"
    description = (
        "Detects tracheal deviation and mediastinal shift using PSPNet trachea "
        "and lung segmentation. Also detects named reference structures "
        "(carina, aortic arch) via Grounding DINO for relative position queries."
    )
    input_format = "CXR image path + args: {reference: str (optional)}"
    output_format = (
        '{ "trachea_deviation_mm": float, "direction": "midline|left|right", '
        '"mediastinum_width_cm": float, "mediastinal_widening": bool, '
        '"reference_bbox": [x1,y1,x2,y2] }'
    )
    example = "Use for tracheal deviation, mediastinal shift, or locating the carina"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_segment, gdino_detect

        img = load_image_np(image_path)
        h, w = img.shape[:2]

        masks = txrv_segment(img)
        right_mask = masks.get("Right Lung", np.zeros((h, w), np.uint8))
        left_mask  = masks.get("Left Lung",  np.zeros((h, w), np.uint8))

        right_xs = np.where(right_mask > 0)[1]
        left_xs  = np.where(left_mask  > 0)[1]

        if len(right_xs) > 0 and len(left_xs) > 0:
            inner_r = int(right_xs.max())
            inner_l = int(left_xs.min())
            med_cx  = (inner_r + inner_l) / 2
            dev_px  = med_cx - w / 2
            dev_mm  = round(dev_px * config.PIXEL_SPACING_MM, 1)
            med_w_cm = round(max(0, inner_l - inner_r) * config.PIXEL_SPACING_MM / 10, 1)
        else:
            dev_mm, med_w_cm = 0.0, 0.0

        direction = ("midline" if abs(dev_mm) < 5 else
                     "right" if dev_mm > 0 else "left")
        widening = med_w_cm > 8.0

        # Grounding DINO for reference structure (carina, aortic arch, etc.)
        reference = args.get("reference", "").strip()
        ref_bbox = None
        if reference and image_path:
            dets = gdino_detect(img, [reference],
                                box_threshold=config.GDINO_BOX_THRESHOLD,
                                text_threshold=config.GDINO_TEXT_THRESHOLD)
            if dets:
                ref_bbox = dets[0]["bbox"]

        return {
            "trachea_deviation_mm": dev_mm,
            "direction": direction,
            "mediastinum_width_cm": med_w_cm,
            "mediastinal_widening": widening,
            "reference_bbox": ref_bbox,
        }
