# utils/model_loader.py — lazy singleton loaders for all DL models
"""
All models load once and are cached for the process lifetime.
No API keys required — all models run fully locally.

Models:
  get_txrv_classifier()   TorchXRayVision DenseNet121  (pathology classification)
  txrv_predict()          Run TXRV → {label: probability}
  get_txrv_segmenter()    TorchXRayVision PSPNet        (14-structure lung seg)
  txrv_segment()          Run PSPNet → dict of masks
  get_medsam()            MedSAM ViT-B                  (bbox-prompted segmentation)
  medsam_segment()        Run MedSAM → binary mask
  get_gdino()             Grounding DINO tiny            (open-vocab detection)
  gdino_detect()          Run GDINO → list of {label, score, bbox}
  get_chexagent()         CheXagent-2-3b                 (CXR planner / tool sim)
  chexagent_ask()         Run CheXagent → answer string
  get_qwen()              Qwen2.5-VL-7B-Instruct         (clinical summarizer)
  qwen_ask()              Run Qwen → answer string
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)
_CACHE: dict[str, object] = {}


# ── 1. TorchXRayVision classifier ─────────────────────────────────────────────

def get_txrv_classifier(weights: str = "densenet121-res224-all"):
    key = f"txrv_cls_{weights}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch, torchxrayvision as xrv
        model = xrv.models.DenseNet(weights=weights)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE[key] = model.to(device)
        logger.info("TorchXRayVision classifier loaded on %s", device)
        return _CACHE[key]
    except ImportError:
        raise ImportError("pip install torchxrayvision")


def txrv_predict(image_np: np.ndarray,
                 weights: str = "densenet121-res224-all") -> dict[str, float]:
    """Return {pathology: sigmoid_prob} for all 18 labels."""
    import torch, torchvision, torchxrayvision as xrv

    model = get_txrv_classifier(weights)
    device = next(model.parameters()).device

    # Preprocess: grayscale → [-1024, 1024] → 224×224
    img = image_np.mean(axis=2) if image_np.ndim == 3 else image_np.copy()
    img = img.astype(np.float32)
    img = xrv.datasets.normalize(img, maxval=255 if img.max() <= 255 else 65535,
                                   reshape=True)
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img = transform(img)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0].cpu().numpy()

    return {lbl: float(p) for lbl, p in zip(model.pathologies, probs) if lbl}


# ── 2. TorchXRayVision segmenter ──────────────────────────────────────────────

# PSPNet structure index map
TXRV_SEG_LABELS = [
    "Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula",
    "Left Lung", "Right Lung",
    "Left Hilus Pulmonis", "Right Hilus Pulmonis",
    "Heart", "Aorta", "Facies Diaphragmatica",
    "Mediastinum", "Weasand", "Spine",
]
TXRV_LEFT_LUNG_IDX  = 4
TXRV_RIGHT_LUNG_IDX = 5
TXRV_HEART_IDX      = 8
TXRV_MEDIASTINUM_IDX = 11
TXRV_TRACHEA_IDX    = 12


def get_txrv_segmenter():
    key = "txrv_seg"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch, torchxrayvision as xrv
        model = xrv.baseline_models.chestx_det.PSPNet()
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHE[key] = model.to(device)
        logger.info("TorchXRayVision PSPNet segmenter loaded on %s", device)
        return _CACHE[key]
    except ImportError:
        raise ImportError("pip install torchxrayvision")


def txrv_segment(image_np: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """
    Run PSPNet on a CXR image.
    Returns dict: {structure_name: binary_mask (H, W, uint8)}.
    Input: numpy array (H, W) or (H, W, 3).
    """
    import torch, torchvision, torchxrayvision as xrv
    from skimage.transform import resize

    model = get_txrv_segmenter()
    device = next(model.parameters()).device
    H_orig, W_orig = image_np.shape[:2]

    img = image_np.mean(axis=2) if image_np.ndim == 3 else image_np.copy()
    img = img.astype(np.float32)
    img = xrv.datasets.normalize(img, maxval=255 if img.max() <= 255 else 65535,
                                   reshape=True)
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(512),
    ])
    img = transform(img)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)[0].cpu().numpy()   # (14, 512, 512)

    masks = {}
    for idx, name in enumerate(TXRV_SEG_LABELS):
        m = (output[idx] > threshold).astype(np.uint8)
        m = (resize(m.astype(float), (H_orig, W_orig),
                    order=0, anti_aliasing=False) > 0.5).astype(np.uint8)
        masks[name] = m
    return masks


# ── 3. MedSAM ─────────────────────────────────────────────────────────────────

def get_medsam(checkpoint: str | Path | None = None):
    key = f"medsam_{checkpoint}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        from segment_anything import sam_model_registry

        ckpt = str(checkpoint or Path(__file__).parent.parent / "model_weights" / "medsam_vit_b.pth")
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"MedSAM checkpoint not found: {ckpt}\n"
                "Run: python scripts/download_models.py"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sam_model_registry["vit_b"](checkpoint=ckpt)
        model = model.to(device).eval()
        _CACHE[key] = model
        logger.info("MedSAM loaded from %s on %s", ckpt, device)
        return model
    except ImportError:
        raise ImportError(
            "MedSAM not installed.\n"
            "  git clone https://github.com/bowang-lab/MedSAM && cd MedSAM && pip install -e ."
        )


def medsam_segment(image_np: np.ndarray,
                   bbox: list[int],
                   checkpoint: str | Path | None = None) -> np.ndarray:
    """
    Segment a region given a bounding box [x1, y1, x2, y2].
    Returns binary mask (H, W) uint8.
    Beats SAM2 on 2D X-ray by ~7.5 Dice points (arXiv:2408.03322).
    """
    import torch
    from skimage.transform import resize as sk_resize

    model = get_medsam(checkpoint)
    device = next(model.parameters()).device
    H, W = image_np.shape[:2]

    # Convert to RGB uint8
    if image_np.dtype != np.uint8:
        img = ((image_np - image_np.min()) /
               (image_np.max() - image_np.min() + 1e-6) * 255).astype(np.uint8)
    else:
        img = image_np.copy()
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)

    # Resize to 1024×1024 (MedSAM's expected input size)
    img_1024 = sk_resize(img, (1024, 1024), order=3,
                         preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_norm = img_1024.astype(np.float32)
    img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min() + 1e-8)
    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # Scale bbox from original image to 1024×1024
    x1, y1, x2, y2 = bbox
    box_1024 = np.array([[
        x1 / W * 1024, y1 / H * 1024,
        x2 / W * 1024, y2 / H * 1024,
    ]])
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)[:, None, :]

    with torch.no_grad():
        img_embed = model.image_encoder(tensor)
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )
        logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = torch.sigmoid(logits)
        pred = torch.nn.functional.interpolate(
            pred, size=(H, W), mode="bilinear", align_corners=False
        )

    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


# ── 4. Grounding DINO ─────────────────────────────────────────────────────────

def get_gdino(model_id: str = "IDEA-Research/grounding-dino-tiny"):
    key = f"gdino_{model_id}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        model.eval()
        _CACHE[key] = (processor, model, device)
        logger.info("Grounding DINO loaded: %s on %s", model_id, device)
        return _CACHE[key]
    except ImportError:
        raise ImportError("pip install transformers torch")


def gdino_detect(image_np: np.ndarray,
                 text_queries: list[str],
                 model_id: str = "IDEA-Research/grounding-dino-tiny",
                 box_threshold: float = 0.20,
                 text_threshold: float = 0.15) -> list[dict]:
    """
    Detect named structures/devices in a CXR by text query.
    text_queries: list of strings e.g. ["carina", "endotracheal tube tip"]
    Returns list of {label, score, bbox:[x1,y1,x2,y2]}.
    Use low thresholds (0.15–0.25) for medical images.
    """
    import torch
    from PIL import Image as PILImage

    processor, model, device = get_gdino(model_id)

    # Convert to PIL RGB
    if image_np.dtype != np.uint8:
        img_u8 = ((image_np - image_np.min()) /
                  (image_np.max() - image_np.min() + 1e-6) * 255).astype(np.uint8)
    else:
        img_u8 = image_np.copy()
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8] * 3, axis=2)
    pil_img = PILImage.fromarray(img_u8)

    # Grounding DINO requires period-separated lowercase queries
    text = ". ".join(q.lower().strip() for q in text_queries) + "."

    inputs = processor(images=pil_img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_img.size[::-1]],
    )

    detections = []
    for score, label, box in zip(
        results[0]["scores"], results[0]["text_labels"], results[0]["boxes"]
    ):
        detections.append({
            "label": label,
            "score": float(score),
            "bbox": [int(x) for x in box.tolist()],
        })
    return sorted(detections, key=lambda d: d["score"], reverse=True)


# ── 5. CheXagent-2-3b (planner + tool simulator) ─────────────────────────────

def get_chexagent(model_name: str = "StanfordAIMI/CheXagent-2-3b"):
    key = f"chexagent_{model_name}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
        ).to(torch.bfloat16).eval()
        _CACHE[key] = (tokenizer, model)
        logger.info("CheXagent-2-3b loaded")
        return _CACHE[key]
    except ImportError:
        raise ImportError("pip install transformers accelerate")


def chexagent_ask(prompt: str,
                  image_path: str | None = None,
                  max_new_tokens: int = 512,
                  model_name: str = "StanfordAIMI/CheXagent-2-3b") -> str:
    """Ask CheXagent a question, optionally with a CXR image."""
    import torch

    tokenizer, model = get_chexagent(model_name)
    device = next(model.parameters()).device

    content_parts = []
    if image_path:
        content_parts.append({"image": image_path})
    content_parts.append({"text": prompt})

    query = tokenizer.from_list_format(content_parts)
    conv = [
        {"from": "system", "value": "You are a helpful radiology assistant."},
        {"from": "human",  "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )[0]

    return tokenizer.decode(output[input_ids.size(1):-1]).strip()


# ── 6. Qwen2.5-VL-7B-Instruct (summarizer) ───────────────────────────────────

def get_qwen(model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
             load_in_4bit: bool = True):
    key = f"qwen_{model_name}_{load_in_4bit}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **quant_kwargs,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        _CACHE[key] = (processor, model)
        logger.info("Qwen2.5-VL-7B loaded (4bit=%s)", load_in_4bit)
        return _CACHE[key]
    except ImportError:
        raise ImportError(
            "pip install transformers>=4.45.0 accelerate bitsandbytes qwen-vl-utils"
        )


def qwen_ask(system_prompt: str,
             user_prompt: str,
             image_path: str | None = None,
             max_new_tokens: int = 512,
             model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
             load_in_4bit: bool = True) -> str:
    """Ask Qwen2.5-VL a clinical reasoning question."""
    import torch

    processor, model = get_qwen(model_name, load_in_4bit)

    user_content = []
    if image_path:
        user_content.append({"type": "image", "image": image_path})
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
    except ImportError:
        # Fallback without qwen_vl_utils
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False)
    trimmed = output[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()
