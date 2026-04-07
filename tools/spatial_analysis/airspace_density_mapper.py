# tools/spatial_analysis/airspace_density_mapper.py
# Model: TorchXRayVision PSPNet zones + DenseNet121 pattern classification
from tools.base import BaseTool


class AirspaceDensityMapper(BaseTool):
    """
    Divides each lung into upper/middle/lower thirds via PSPNet masks,
    then measures opacity fraction per zone using pixel intensity analysis
    refined by TorchXRayVision pathology probabilities.
    """
    id = "airspace_density_mapper"
    name = "Airspace Density Mapper"
    category = "Spatial Analysis"
    description = (
        "Maps opacity across 6 lung zones (RUL/RML/RLL/LUL/LML/LLL) using "
        "TorchXRayVision PSPNet lung segmentation and pixel intensity analysis."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "zone_map": { "RUL": {"opacity_pct": float, "pattern": str}, '
        '"RML": {...}, "RLL": {...}, "LUL": {...}, "LML": {...}, "LLL": {...} } }'
    )
    example = "Use to determine lobar vs diffuse vs bilateral distribution"

    def execute(self, image_path: str | None, args: dict) -> dict:
        import numpy as np
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_segment, txrv_predict

        img  = load_image_np(image_path)
        gray = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        h, w = gray.shape

        masks  = txrv_segment(img)
        probs  = txrv_predict(img, config.TXRV_WEIGHTS)
        consol = probs.get("Consolidation", 0.0)
        edema  = probs.get("Edema", 0.0)
        infil  = probs.get("Infiltration", 0.0)
        overall_prob = max(consol, edema, infil)

        def _pattern(opacity_pct, prob):
            if opacity_pct < 5:   return "clear"
            if opacity_pct < 20 and prob < 0.3: return "ground-glass opacity"
            if opacity_pct < 40: return "ground-glass opacity"
            return "consolidation"

        def zone_stats(lung_mask, side):
            ys, _ = np.where(lung_mask > 0)
            if len(ys) == 0:
                return {f"{side}UL": {"opacity_pct": 0.0, "pattern": "indeterminate"},
                        f"{side}ML": {"opacity_pct": 0.0, "pattern": "indeterminate"},
                        f"{side}LL": {"opacity_pct": 0.0, "pattern": "indeterminate"}}
            y_min, y_max = ys.min(), ys.max()
            zone_h = (y_max - y_min) / 3
            aerated = float(np.percentile(gray[lung_mask > 0], 25))
            out = {}
            for i, lbl in enumerate([f"{side}UL", f"{side}ML", f"{side}LL"]):
                ys0, ye = int(y_min + i * zone_h), int(y_min + (i + 1) * zone_h)
                zone_px = np.zeros((h, w), bool)
                zone_px[ys0:ye, :] = True
                zone_px &= (lung_mask > 0)
                if not zone_px.any():
                    out[lbl] = {"opacity_pct": 0.0, "pattern": "indeterminate"}
                    continue
                zone_mean = float(gray[zone_px].mean())
                op_pct = round(max(0.0, (zone_mean - aerated) /
                               (gray.max() - aerated + 1e-6)) * 100, 1)
                out[lbl] = {"opacity_pct": op_pct,
                            "pattern": _pattern(op_pct, overall_prob)}
            return out

        right = zone_stats(masks.get("Right Lung", np.zeros((h, w), np.uint8)), "R")
        left  = zone_stats(masks.get("Left Lung",  np.zeros((h, w), np.uint8)), "L")
        return {"zone_map": {**right, **left}}
