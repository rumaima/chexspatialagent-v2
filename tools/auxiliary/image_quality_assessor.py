# tools/auxiliary/image_quality_assessor.py
# No external model needed — pixel heuristics only
from tools.base import BaseTool


class ImageQualityAssessor(BaseTool):
    """Evaluates rotation, inspiration, and penetration using pixel heuristics."""
    id = "image_quality_assessor"
    name = "Image Quality Assessor"
    category = "Auxiliary"
    description = (
        "Evaluates CXR technical quality: rotation (L/R symmetry), "
        "inspiration (diaphragm height), penetration (intensity histogram). "
        "Always run first — poor quality can mimic or mask pathology."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "rotation": "adequate|rotated", "inspiration": "adequate|poor", '
        '"penetration": "adequate|over|under", '
        '"quality_score": float, "warnings": [str] }'
    )
    example = "Always the first step in any plan"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        from utils.image import load_image_np

        img  = load_image_np(image_path)
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        h, w = gray.shape
        warnings = []

        upper  = gray[:int(h * 0.35), :]
        sym    = abs(upper[:, :w//2].mean() - upper[:, w//2:].mean())
        sym   /= max(upper.mean(), 1e-6)
        rotation = "rotated" if sym > 0.12 else "adequate"
        if rotation == "rotated":
            warnings.append("Image appears rotated — asymmetric upper lung fields")

        lower_q  = gray[int(h * 0.75):, :]
        dia_row  = int(h * 0.75) + int(lower_q.mean(axis=1).argmax())
        inspiration = "adequate" if dia_row / h > 0.65 else "poor"
        if inspiration == "poor":
            warnings.append("Poor inspiration — diaphragm appears high")

        med = float(np.median(gray))
        if   med < 60:  penetration = "under"; warnings.append("Under-penetrated (too bright)")
        elif med > 180: penetration = "over";  warnings.append("Over-penetrated (too dark)")
        else:           penetration = "adequate"

        penalties     = (rotation == "rotated") * 0.2 + \
                        (inspiration == "poor") * 0.15 + \
                        (penetration != "adequate") * 0.15
        quality_score = round(max(0.0, 1.0 - penalties), 2)

        return {"rotation": rotation, "inspiration": inspiration,
                "penetration": penetration, "quality_score": quality_score,
                "warnings": warnings}
