# tools/general_perception/lung_region_detector.py
# Model: TorchXRayVision PSPNet — 14-structure CXR segmenter, auto-downloads weights
from tools.base import BaseTool


class LungRegionDetector(BaseTool):
    """
    Segments 14 thoracic structures using TorchXRayVision PSPNet.
    Weights auto-download on first use (~100 MB).
    Returns: left/right lung masks, heart mask, key bboxes.
    """
    id = "lung_region_detector"
    name = "Lung Region Detector"
    category = "General Perception"
    description = (
        "Segments left lung, right lung, heart, mediastinum, trachea and 9 other "
        "thoracic structures using TorchXRayVision PSPNet (auto-downloads weights). "
        "Provides binary masks and bounding boxes for all structures."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "left_lung": {"bbox":[x1,y1,x2,y2], "area_px":int}, '
        '"right_lung": {"bbox":[x1,y1,x2,y2], "area_px":int}, '
        '"heart": {"bbox":[x1,y1,x2,y2], "area_px":int}, '
        '"total_lung_area_px": int, "structures_found": [str] }'
    )
    example = "Run before any tool needing lung zones, laterality or containment"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        from utils.image import load_image_np
        from utils.model_loader import txrv_segment, TXRV_LEFT_LUNG_IDX, TXRV_RIGHT_LUNG_IDX, TXRV_HEART_IDX
        from utils.spatial_geometry import mask_bbox

        img = load_image_np(image_path)
        masks = txrv_segment(img)

        left_mask  = masks.get("Left Lung",  np.zeros(img.shape[:2], np.uint8))
        right_mask = masks.get("Right Lung", np.zeros(img.shape[:2], np.uint8))
        heart_mask = masks.get("Heart",      np.zeros(img.shape[:2], np.uint8))

        return {
            "left_lung":  {"bbox": mask_bbox(left_mask),  "area_px": int(left_mask.sum())},
            "right_lung": {"bbox": mask_bbox(right_mask), "area_px": int(right_mask.sum())},
            "heart":      {"bbox": mask_bbox(heart_mask), "area_px": int(heart_mask.sum())},
            "total_lung_area_px": int(left_mask.sum() + right_mask.sum()),
            "structures_found": [k for k, v in masks.items() if v.sum() > 100],
        }
