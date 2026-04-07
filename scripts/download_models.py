#!/usr/bin/env python3
# scripts/download_models.py — download MedSAM checkpoint
"""
Run once before first use in real mode:
    python scripts/download_models.py

What this downloads:
  medsam_vit_b.pth  (~357 MB) — MedSAM ViT-B segmentation checkpoint

What auto-downloads on first use (no action needed):
  TorchXRayVision weights  (~120 MB → ~/.torchxrayvision/)
  Grounding DINO weights   (~700 MB → ~/.cache/huggingface/)
  CheXagent-2-3b weights   (~6 GB  → ~/.cache/huggingface/)
  Qwen2.5-VL-7B weights    (~16 GB → ~/.cache/huggingface/)
"""
import urllib.request
from pathlib import Path

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model_weights"
MODEL_DIR.mkdir(exist_ok=True)

MEDSAM_URL  = "https://drive.usercontent.google.com/download?id=1UAmWL88roYR7wKlnApw5Bcuzns2-tDOm&confirm=t"
MEDSAM_DEST = MODEL_DIR / "medsam_vit_b.pth"
MEDSAM_MB   = 357


def _progress(count, block, total):
    pct = min(100, count * block * 100 // max(total, 1))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download_medsam():
    if MEDSAM_DEST.exists():
        print(f"✓ MedSAM already present ({MEDSAM_DEST.stat().st_size // 1_000_000} MB)")
        return
    print(f"Downloading MedSAM checkpoint ({MEDSAM_MB} MB)…")
    print("  Source: bowang-lab/MedSAM (Nature Communications 2024)")
    print(f"  Destination: {MEDSAM_DEST}\n")
    print("  Note: if the Google Drive link fails, download medsam_vit_b.pth manually")
    print("  from https://github.com/bowang-lab/MedSAM and place it in model_weights/\n")
    try:
        urllib.request.urlretrieve(MEDSAM_URL, str(MEDSAM_DEST), reporthook=_progress)
        print(f"\n✓ Saved → {MEDSAM_DEST}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("  Please download manually from https://github.com/bowang-lab/MedSAM")


def check_packages():
    print("\nChecking packages…")
    checks = [
        ("torchxrayvision",   "torchxrayvision"),
        ("transformers",      "transformers"),
        ("torch",             "torch"),
        ("bitsandbytes",      "bitsandbytes"),
        ("segment-anything",  "segment_anything"),
        ("qwen-vl-utils",     "qwen_vl_utils"),
    ]
    for name, mod in checks:
        try:
            __import__(mod)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}  →  pip install {name}")


if __name__ == "__main__":
    print("=" * 55)
    print("  CXR SpatialAgent — Model Setup")
    print("=" * 55)
    check_packages()
    print()
    download_medsam()
    print("\nDone. Run: python pipeline.py --image cxr.jpg --question '...'")
