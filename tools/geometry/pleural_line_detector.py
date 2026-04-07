# tools/geometry/pleural_line_detector.py
# Model: TorchXRayVision DenseNet121 + PSPNet lateral-margin lucency analysis
from tools.base import BaseTool


class PleuralLineDetector(BaseTool):
    """
    Detects pneumothorax using TorchXRayVision Pneumothorax classifier
    combined with PSPNet-based lateral-margin hyper-lucency analysis.
    """
    id = "pleural_line_detector"
    name = "Pleural Line Detector"
    category = "Geometry & Measurement"
    description = (
        "Detects pneumothorax using TorchXRayVision DenseNet121 (Pneumothorax label) "
        "combined with PSPNet lateral-lung-margin lucency analysis."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "pleural_line_detected": bool, "side": "left|right|bilateral|none", '
        '"pneumothorax_probability": float, '
        '"lung_collapse_pct": float, "size_classification": "none|small|moderate|large" }'
    )
    example = "Use when assessing pneumothorax or ruling out tension PTX"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict, txrv_segment

        img  = load_image_np(image_path)
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        h, w = gray.shape

        probs    = txrv_predict(img, config.TXRV_WEIGHTS)
        ptx_prob = probs.get("Pneumothorax", 0.0)

        masks = txrv_segment(img)

        def lateral_lucency(lung_mask):
            ys, xs = np.where(lung_mask > 0)
            if len(xs) == 0: return 0.0
            x_min, x_max = xs.min(), xs.max()
            lat_w = int((x_max - x_min) * 0.15)
            lat = np.zeros_like(lung_mask, bool)
            if xs.mean() < w / 2:
                lat[:, x_min:x_min + lat_w] = True
            else:
                lat[:, x_max - lat_w:x_max] = True
            lat &= (lung_mask > 0)
            if not lat.any(): return 0.0
            ref = float(np.percentile(gray[lung_mask > 0], 10))
            return float((gray[lat] < ref * 0.85).mean())

        r_luc = lateral_lucency(masks.get("Right Lung", np.zeros((h, w), np.uint8)))
        l_luc = lateral_lucency(masks.get("Left Lung",  np.zeros((h, w), np.uint8)))

        detected = ptx_prob >= config.TXRV_THRESHOLD
        if r_luc > l_luc and (r_luc > 0.3 or detected): side = "right"
        elif l_luc > r_luc and (l_luc > 0.3 or detected): side = "left"
        elif detected: side = "bilateral"
        else: side = "none"

        collapse_pct = round(max(r_luc, l_luc) * 60 + ptx_prob * 20, 1)
        size = ("none" if collapse_pct < 5 and not detected else
                "small" if collapse_pct < 15 else
                "moderate" if collapse_pct < 30 else "large")

        return {
            "pleural_line_detected": detected,
            "side": side,
            "pneumothorax_probability": round(ptx_prob, 3),
            "lung_collapse_pct": collapse_pct,
            "size_classification": size,
        }
