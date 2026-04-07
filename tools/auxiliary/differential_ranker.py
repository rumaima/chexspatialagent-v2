# tools/auxiliary/differential_ranker.py
# Model: TorchXRayVision DenseNet121 full probability vector
from tools.base import BaseTool


class DifferentialRanker(BaseTool):
    """Ranks differentials using TorchXRayVision's 18-label probability vector."""
    id = "differential_ranker"
    name = "Differential Diagnosis Ranker"
    category = "Auxiliary"
    description = (
        "Ranks differential diagnoses using TorchXRayVision DenseNet121 "
        "full pathology probability vector."
    )
    input_format = "CXR image path"
    output_format = (
        '{ "differentials": [{ "diagnosis": str, "probability": "high|moderate|low", '
        '"supporting_findings": [str], "against_findings": [str] }] }'
    )
    example = "Run after all imaging tools, before Terminate"

    _GROUPS = {
        "Pneumonia / consolidation": ["Consolidation", "Infiltration", "Pneumonia"],
        "Pleural effusion":          ["Pleural Effusion", "Effusion"],
        "Pneumothorax":              ["Pneumothorax"],
        "Pulmonary oedema":          ["Edema"],
        "Atelectasis":               ["Atelectasis"],
        "Cardiomegaly":              ["Cardiomegaly"],
        "Pulmonary mass / nodule":   ["Mass", "Nodule"],
        "Emphysema / COPD":          ["Emphysema"],
        "Pulmonary fibrosis":        ["Fibrosis"],
        "Rib fracture":              ["Fracture"],
    }

    def execute(self, image_path: str | None, args: dict) -> dict:
        import config
        from utils.image import load_image_np
        from utils.model_loader import txrv_predict

        img   = load_image_np(image_path)
        probs = txrv_predict(img, config.TXRV_WEIGHTS)

        diffs = []
        for diagnosis, labels in self._GROUPS.items():
            p = max((probs.get(lbl, 0.0) for lbl in labels), default=0.0)
            if p < 0.10: continue
            prob_str = "high" if p > 0.5 else ("moderate" if p > 0.25 else "low")
            supporting = [f"{lbl} {probs.get(lbl,0):.0%}" for lbl in labels if probs.get(lbl,0) > 0.2]
            against    = [f"{lbl} low" for lbl in labels if probs.get(lbl,0) < 0.1]
            diffs.append({"diagnosis": diagnosis, "probability": prob_str,
                          "supporting_findings": supporting[:3],
                          "against_findings": against[:2]})

        diffs.sort(key=lambda d: {"high":3,"moderate":2,"low":1}[d["probability"]], reverse=True)
        return {"differentials": diffs[:6]}
