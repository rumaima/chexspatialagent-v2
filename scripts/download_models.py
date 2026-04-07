#!/usr/bin/env python3
# scripts/download_models.py — download MedSAM checkpoint
"""
Run once before first use in real mode:
    python scripts/download_models.py

What this downloads:
  medsam_vit_b.pth  (~2.4 GB) — MedSAM ViT-B segmentation checkpoint

What auto-downloads on first use (no action needed):
  TorchXRayVision weights  (~120 MB → ~/.torchxrayvision/)
  Grounding DINO weights   (~700 MB → ~/.cache/huggingface/)
  CheXagent-2-3b weights   (~6 GB  → ~/.cache/huggingface/)
  Qwen2.5-VL-7B weights    (~16 GB → ~/.cache/huggingface/)
"""
import urllib.request
import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model_weights"
MODEL_DIR.mkdir(exist_ok=True)

# Correct Google Drive file ID from bowang-lab/MedSAM
MEDSAM_GDRIVE_ID = "1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_"
MEDSAM_URL       = f"https://drive.usercontent.google.com/download?id={MEDSAM_GDRIVE_ID}&confirm=t"
MEDSAM_DEST      = MODEL_DIR / "medsam_vit_b.pth"
MEDSAM_MB        = 2400


def _progress(count, block, total):
    pct = min(100, count * block * 100 // max(total, 1))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%", end="", flush=True)


def download_medsam():
    if MEDSAM_DEST.exists():
        print(f"✓ MedSAM already present ({MEDSAM_DEST.stat().st_size // 1_000_000} MB)")
        return
    print(f"Downloading MedSAM checkpoint (~{MEDSAM_MB} MB)…")
    print("  Source: bowang-lab/MedSAM (Nature Communications 2024)")
    print(f"  Destination: {MEDSAM_DEST}\n")

    # Try gdown first (handles Google Drive large-file confirmation properly)
    try:
        import gdown
        print("  Using gdown for reliable Google Drive download…\n")
        gdown.download(id=MEDSAM_GDRIVE_ID, output=str(MEDSAM_DEST), quiet=False)
        if MEDSAM_DEST.exists() and MEDSAM_DEST.stat().st_size > 100_000_000:
            print(f"\n✓ Saved → {MEDSAM_DEST}")
            return
    except ImportError:
        print("  gdown not found, trying urllib (install gdown for more reliability)…\n")
    except Exception as e:
        print(f"\n  gdown failed: {e}, falling back to urllib…\n")

    # Fallback: urllib
    try:
        urllib.request.urlretrieve(MEDSAM_URL, str(MEDSAM_DEST), reporthook=_progress)
        print(f"\n✓ Saved → {MEDSAM_DEST}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("  Please download manually:")
        print(f"    gdown --id {MEDSAM_GDRIVE_ID} -O {MEDSAM_DEST}")
        print("  or visit: https://github.com/bowang-lab/MedSAM")


def check_packages():
    print("\nChecking packages…")
    checks = [
        ("torchxrayvision",   "torchxrayvision"),
        ("transformers",      "transformers"),
        ("torch",             "torch"),
        ("bitsandbytes",      "bitsandbytes"),
        ("segment-anything",  "segment_anything"),
        ("qwen-vl-utils",     "qwen_vl_utils"),
        ("gdown",             "gdown"),
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