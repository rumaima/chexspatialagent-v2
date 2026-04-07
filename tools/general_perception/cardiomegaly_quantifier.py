# tools/general_perception/cardiomegaly_quantifier.py
# Model: TorchXRayVision DenseNet121 + PSPNet geometry
from tools.base import BaseTool


class CardiomegalyQuantifier(BaseTool):
    """
    Combines TorchXRayVision classifier probability with geometric CTR
    measurement derived from PSPNet lung/heart segmentation masks.
    CTR > 0.5 on a PA film = cardiomegaly.
    """
    id = "cardiomegaly_quantifier"
    name = "Cardiomegaly Quantifier"
    category = "General Perception"
    description = (
        "Measures cardiothoracic ratio (CTR) using TorchXRayVision DenseNet121 "
        "and PSPNet lung/heart geometry. CTR > 0.5 = cardiomegaly."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "cardiomegaly_prob": float, "CTR_estimate": float, '
        '"label": "normal|borderline|cardiomegaly", "confidence": float }'
    )
    example = "Use when assessing cardiac size or signs of congestive heart failure"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict, txrv_segment

        img = load_image_np(image_path)
        probs = txrv_predict(img, config.TXRV_WEIGHTS)
        cxr_prob = probs.get("Cardiomegaly", 0.0)

        masks = txrv_segment(img)
        right_mask = masks.get("Right Lung", np.zeros(img.shape[:2], np.uint8))
        left_mask  = masks.get("Left Lung",  np.zeros(img.shape[:2], np.uint8))

        right_xs = np.where(right_mask > 0)[1]
        left_xs  = np.where(left_mask  > 0)[1]

        if len(right_xs) > 0 and len(left_xs) > 0:
            thoracic_w = int(left_xs.max())  - int(right_xs.min())
            heart_w    = max(0, int(left_xs.min()) - int(right_xs.max()))
            ctr = heart_w / thoracic_w if thoracic_w > 0 else 0.5
        else:
            ctr = 0.5

        ctr_est = round(0.5 * ctr + 0.5 * (0.4 + cxr_prob * 0.2), 3)
        if cxr_prob > 0.5 or ctr_est > 0.55:
            label = "cardiomegaly"
        elif cxr_prob > 0.3 or ctr_est > 0.5:
            label = "borderline"
        else:
            label = "normal"

        return {
            "cardiomegaly_prob": round(cxr_prob, 3),
            "CTR_estimate": ctr_est,
            "label": label,
            "confidence": round(max(cxr_prob, 1 - cxr_prob), 3),
        }
