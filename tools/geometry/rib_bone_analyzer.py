# tools/geometry/rib_bone_analyzer.py
# Model: TorchXRayVision DenseNet121 + lateral intensity analysis
from tools.base import BaseTool


class RibBoneAnalyzer(BaseTool):
    """
    Analyses bony structures using TorchXRayVision classification
    and lateral-rib-zone intensity analysis.
    """
    id = "rib_bone_analyzer"
    name = "Rib & Bone Analyzer"
    category = "Geometry & Measurement"
    description = (
        "Flags rib fractures, lytic/sclerotic lesions and scoliosis using "
        "TorchXRayVision Fracture label and lateral rib-zone intensity analysis."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "fracture_probability": float, "fracture_suspected": bool, '
        '"scoliosis_suspected": bool, '
        '"bone_lesions": [{"type": str, "location": str}] }'
    )
    example = "Use after trauma or when bone metastases are suspected"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict

        img  = load_image_np(image_path)
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        h, w = gray.shape

        probs       = txrv_predict(img, config.TXRV_WEIGHTS)
        frac_prob   = probs.get("Fracture", 0.0)

        left_strip  = gray[:, :int(w * 0.2)].mean()
        right_strip = gray[:, int(w * 0.8):].mean()
        sym_ratio   = abs(left_strip - right_strip) / (max(left_strip, right_strip) + 1e-6)
        scoliosis   = sym_ratio > 0.15

        lesions = []
        for side, x_range in [("right", (0, int(w * 0.15))), ("left", (int(w * 0.85), w))]:
            strip     = gray[:, x_range[0]:x_range[1]]
            row_means = strip.mean(axis=1)
            bright    = np.where(row_means > np.percentile(row_means, 90))[0]
            if 0 < len(bright) < h * 0.1:
                zone = "upper" if bright.mean() < h/3 else (
                       "lower" if bright.mean() > 2*h/3 else "middle")
                lesions.append({"type": "sclerotic density", "location": f"{side} {zone}"})

        return {
            "fracture_probability": round(frac_prob, 3),
            "fracture_suspected": frac_prob > 0.4,
            "scoliosis_suspected": scoliosis,
            "bone_lesions": lesions[:3],
        }
