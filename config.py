# config.py — CXR SpatialAgent configuration

import os
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
MODEL_DIR = ROOT / "model_weights"
MODEL_DIR.mkdir(exist_ok=True)

# ── Execution mode ─────────────────────────────────────────────────────────────
# "real"      — runs DL models (requires weights + GPU)
# "simulated" — CheXagent-2 acts as every tool (GPU recommended, no weights needed)
DEFAULT_MODE = "real"

# ── 1. Pathology classification — TorchXRayVision DenseNet121 ──────────────────
# pip install torchxrayvision
# Weights auto-download on first use (~120 MB to ~/.torchxrayvision/)
TXRV_WEIGHTS   = "densenet121-res224-all"   # NIH+CheXpert+MIMIC+PadChest
TXRV_THRESHOLD = 0.3                         # probability threshold for "present"

# ── 2. Lung segmentation — TorchXRayVision PSPNet ─────────────────────────────
# Same pip install; 14-structure segmenter weights auto-download
# Index mapping: 4=Left Lung, 5=Right Lung, 8=Heart, 11=Mediastinum, 12=Trachea
TXRV_SEG_SIZE = 512   # PSPNet expects 512×512

# ── 3. Bbox-prompted segmentation — MedSAM ────────────────────────────────────
# git clone https://github.com/bowang-lab/MedSAM && pip install -e .
# Download checkpoint: see scripts/download_models.py
# Outperforms SAM2 on 2D X-ray by ~7.5 Dice points (arXiv:2408.03322)
MEDSAM_CHECKPOINT = MODEL_DIR / "medsam_vit_b.pth"   # ~357 MB
MEDSAM_IMAGE_SIZE = 1024

# ── 4. Open-vocabulary detection — Grounding DINO ─────────────────────────────
# pip install transformers
# Detects ANY anatomical structure or device by text name, no retraining needed
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"   # 172M params, ~3 GB VRAM
# Use "IDEA-Research/grounding-dino-base" for better accuracy (341M, ~6 GB VRAM)
GDINO_BOX_THRESHOLD  = 0.20   # lower for medical images (structures are subtle)
GDINO_TEXT_THRESHOLD = 0.15

# ── 5. Planner — CheXagent-2-3b (local, no API key) ──────────────────────────
# pip install torch transformers accelerate
# HuggingFace: StanfordAIMI/CheXagent-2-3b  (~6 GB bfloat16)
# Instruction-tuned on 8.5M+ CXR samples from 28 datasets (CheXinstruct)
CHEXAGENT_MODEL  = "StanfordAIMI/CheXagent-2-3b"
CHEXAGENT_DTYPE  = "bfloat16"   # or "float16" for older GPUs
CHEXAGENT_DEVICE = "auto"       # uses device_map="auto"

# ── 6. Summarizer — Qwen2.5-VL-7B-Instruct (local, no API key) ───────────────
# pip install transformers>=4.45.0 accelerate bitsandbytes qwen-vl-utils
# HuggingFace: Qwen/Qwen2.5-VL-7B-Instruct (~5 GB at 4-bit)
# Best open-source VLM for structured clinical reasoning; Apache 2.0
QWEN_MODEL        = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_LOAD_IN_4BIT = True    # ~5 GB VRAM; set False for full bfloat16 (~16 GB)
QWEN_MAX_PIXELS   = 512 * 28 * 28   # limit image tokens to avoid OOM

# ── Spatial constants ──────────────────────────────────────────────────────────
PIXEL_SPACING_MM    = 0.28   # standard PA CXR at 512×512 (≈0.28 mm/px)
PERIPHERAL_FRACTION = 0.3    # outer 30% of lung = peripheral

# ── Pipeline ───────────────────────────────────────────────────────────────────
MAX_PLAN_STEPS       = 7
TOOL_TIMEOUT_SECONDS = 60

# ── Image ──────────────────────────────────────────────────────────────────────
IMAGE_MAX_SIZE = (1024, 1024)
IMAGE_QUALITY  = 85

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL        = "INFO"
LOG_TOOL_OUTPUTS = True
