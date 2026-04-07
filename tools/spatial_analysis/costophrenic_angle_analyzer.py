# tools/spatial_analysis/costophrenic_angle_analyzer.py
# Model: TorchXRayVision DenseNet121 + PSPNet CP-region intensity
from tools.base import BaseTool


class CostophrenicAngleAnalyzer(BaseTool):
    """
    Estimates CP angle blunting by analysing the lower 20% of each
    PSPNet lung mask intensity, refined by TorchXRayVision effusion probability.
    """
    id = "costophrenic_angle_analyzer"
    name = "Costophrenic Angle Analyzer"
    category = "Spatial Analysis"
    description = (
        "Estimates costophrenic angle blunting using TorchXRayVision PSPNet "
        "lung segmentation and pleural effusion classifier probability."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "left_blunting": "none|mild|moderate|severe", '
        '"right_blunting": "none|mild|moderate|severe", '
        '"effusion_probability": float, "effusion_likelihood": "low|moderate|high" }'
    )
    example = "Use when assessing for pleural effusion"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict, txrv_segment

        img  = load_image_np(image_path)
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        h, w = gray.shape

        probs = txrv_predict(img, config.TXRV_WEIGHTS)
        eff_prob = max(probs.get("Pleural Effusion", 0.0), probs.get("Effusion", 0.0))

        masks = txrv_segment(img)

        def cp_blunting(lung_mask):
            ys, _ = np.where(lung_mask > 0)
            if len(ys) == 0: return "indeterminate"
            y_cp = int(ys.max() * 0.80)
            cp = np.zeros((h, w), bool)
            cp[y_cp:, :] = True
            cp &= (lung_mask > 0)
            if not cp.any(): return "none"
            aerated = float(np.percentile(gray[lung_mask > 0], 25))
            ratio = (float(gray[cp].mean()) - aerated) / (gray.max() - aerated + 1e-6)
            if ratio > 0.6 or eff_prob > 0.6: return "severe"
            if ratio > 0.35 or eff_prob > 0.4: return "moderate"
            if ratio > 0.15 or eff_prob > 0.25: return "mild"
            return "none"

        left_b  = cp_blunting(masks.get("Left Lung",  np.zeros((h, w), np.uint8)))
        right_b = cp_blunting(masks.get("Right Lung", np.zeros((h, w), np.uint8)))
        likelihood = "high" if eff_prob > 0.5 else ("moderate" if eff_prob > 0.25 else "low")

        return {
            "left_blunting": left_b,
            "right_blunting": right_b,
            "effusion_probability": round(eff_prob, 3),
            "effusion_likelihood": likelihood,
        }
