# utils/image.py — CXR image loading and preprocessing

import base64
import io
from pathlib import Path

from PIL import Image

import config


def load_and_resize(image_path: str) -> Image.Image:
    """Load a CXR image and resize to config.IMAGE_MAX_SIZE."""
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail(config.IMAGE_MAX_SIZE, Image.LANCZOS)
    return img


def encode_image_b64(image_path: str) -> tuple[str, str]:
    """
    Encode an image to base64 for the Anthropic API.

    Returns:
        (base64_string, media_type)  e.g. ("...", "image/jpeg")
    """
    suffix = Path(image_path).suffix.lower()
    format_map = {
        ".jpg": ("JPEG", "image/jpeg"),
        ".jpeg": ("JPEG", "image/jpeg"),
        ".png": ("PNG", "image/png"),
        ".gif": ("GIF", "image/gif"),
        ".webp": ("WEBP", "image/webp"),
    }
    pil_format, media_type = format_map.get(suffix, ("JPEG", "image/jpeg"))

    img = load_and_resize(image_path)
    buf = io.BytesIO()
    img.save(buf, format=pil_format, quality=config.IMAGE_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, media_type


def is_valid_image(image_path: str) -> bool:
    """Check if the path points to a readable image."""
    try:
        Image.open(image_path).verify()
        return True
    except Exception:
        return False


# ── Numpy helpers (used by real tool implementations) ─────────────────────────

def load_image_np(image_path: str | None):
    """
    Load a CXR image as a numpy array (H, W, 3) uint8.
    Returns a blank 512×512 array if image_path is None or missing.
    """
    import numpy as np
    if not image_path:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    try:
        img = load_and_resize(image_path).convert("RGB")
        return np.array(img)
    except Exception:
        return np.zeros((512, 512, 3), dtype=np.uint8)


def to_rgb_uint8(image_np) -> "np.ndarray":
    """Ensure a numpy array is (H, W, 3) uint8 — required by SAM2 and YOLO."""
    import numpy as np
    img = image_np.copy()
    if img.dtype != np.uint8:
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    return img
