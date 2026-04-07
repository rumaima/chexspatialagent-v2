# utils/spatial_geometry.py — low-level spatial computations for CXR analysis
"""
All geometric operations used by the spatial tools:
  - Mask centroid, bounding box, area
  - Inter-structure distance (centroid, nearest-surface / Hausdorff)
  - Cardinal direction (radiological convention)
  - Lung zone assignment (upper / middle / lower)
  - Distribution classification (focal / diffuse / bilateral / central / peripheral)
  - Containment ratio (within vs beyond pleural surface)
  - Border proximity (silhouette sign)
  - Device tip extraction
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import label as nd_label
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpatialRelation:
    direction: str
    distance_px: float
    distance_mm: float
    relation_text: str
    centroid_a: tuple[float, float]
    centroid_b: tuple[float, float]
    nearest_surface_dist_px: float
    overlap_iou: float


# ── Mask primitives ───────────────────────────────────────────────────────────

def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (cy, cx) centroid of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return float(mask.shape[0] / 2), float(mask.shape[1] / 2)
    return float(ys.mean()), float(xs.mean())


def mask_bbox(mask: np.ndarray) -> list[int]:
    """Return [x1, y1, x2, y2] tight bounding box of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return [0, 0, w, h]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter) / float(union + 1e-6)


def nearest_surface_distance(mask_a: np.ndarray, mask_b: np.ndarray,
                              max_pts: int = 500) -> float:
    """Minimum pixel distance between the surfaces of two binary masks."""
    from scipy.ndimage import binary_erosion

    def boundary(m):
        eroded = binary_erosion(m > 0)
        edge = (m > 0) & ~eroded
        ys, xs = np.where(edge)
        pts = np.stack([ys, xs], axis=1).astype(float)
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
        return pts

    pts_a, pts_b = boundary(mask_a), boundary(mask_b)
    if len(pts_a) == 0 or len(pts_b) == 0:
        cy_a, cx_a = mask_centroid(mask_a)
        cy_b, cx_b = mask_centroid(mask_b)
        return float(np.sqrt((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2))
    return float(cdist(pts_a, pts_b).min())


# ── Cardinal direction ────────────────────────────────────────────────────────

def cardinal_direction(cy_a, cx_a, cy_b, cx_b, h, w) -> str:
    """Direction of A with respect to B in PA CXR convention."""
    dy = cy_a - cy_b   # negative → A is superior
    dx = cx_a - cx_b   # positive → A is to the image-right (patient's left)
    threshold = 0.05
    parts = []
    if abs(dy) / h > threshold:
        parts.append("superior" if dy < 0 else "inferior")
    if abs(dx) / w > threshold:
        parts.append("to the right" if dx > 0 else "to the left")
    return " and ".join(parts) if parts else "at approximately the same level"


# ── Relative position (main) ──────────────────────────────────────────────────

def compute_spatial_relation(mask_a: np.ndarray, mask_b: np.ndarray,
                              pixel_spacing_mm: float = 0.28) -> SpatialRelation:
    h, w = mask_a.shape
    cy_a, cx_a = mask_centroid(mask_a)
    cy_b, cx_b = mask_centroid(mask_b)
    centroid_dist = float(np.sqrt((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2))
    centroid_dist_mm = centroid_dist * pixel_spacing_mm
    surf_dist = nearest_surface_distance(mask_a, mask_b)
    surf_dist_mm = surf_dist * pixel_spacing_mm
    direction = cardinal_direction(cy_a, cx_a, cy_b, cx_b, h, w)
    iou = mask_iou(mask_a, mask_b)

    if iou > 0.1:
        text = f"overlapping with the reference structure (IoU={iou:.2f})"
    elif surf_dist_mm < 10:
        text = f"immediately adjacent ({surf_dist_mm:.1f} mm), {direction}"
    elif surf_dist_mm < 30:
        text = f"near the reference ({surf_dist_mm:.1f} mm), {direction}"
    else:
        text = f"{direction} of the reference (centroid ≈{centroid_dist_mm:.0f} mm)"

    return SpatialRelation(
        direction=direction,
        distance_px=centroid_dist,
        distance_mm=centroid_dist_mm,
        relation_text=text,
        centroid_a=(cy_a, cx_a),
        centroid_b=(cy_b, cx_b),
        nearest_surface_dist_px=surf_dist,
        overlap_iou=iou,
    )


# ── Device tip ────────────────────────────────────────────────────────────────

def find_device_tip(mask: np.ndarray, device_type: str = "") -> tuple[float, float]:
    """Estimate device tip as the most inferior point of its mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask_centroid(mask)
    max_y = ys.max()
    return float(max_y), float(xs[ys == max_y].mean())


def tip_vs_reference(tip_yx: tuple, ref_mask: np.ndarray,
                     pixel_spacing_mm: float = 0.28) -> dict:
    """Clinical zone of a device tip relative to a reference structure."""
    cy_r, cx_r = mask_centroid(ref_mask)
    bbox = mask_bbox(ref_mask)  # [x1,y1,x2,y2]
    ty, tx = tip_yx
    dy = ty - cy_r
    dist_mm = float(np.sqrt(dy ** 2 + (tx - cx_r) ** 2)) * pixel_spacing_mm
    dist_above_mm = -dy * pixel_spacing_mm

    if ty < bbox[1]:
        zone = "above"
    elif ty > bbox[3]:
        zone = "below"
    else:
        zone = "at the level of"

    return {
        "zone": zone,
        "dist_mm": round(dist_mm, 1),
        "dist_above_reference_mm": round(dist_above_mm, 1),
        "text": (
            f"tip is {zone} the reference "
            f"(≈{abs(dist_above_mm):.1f} mm "
            f"{'superior' if dist_above_mm > 0 else 'inferior'})"
        ),
    }


# ── Lung zone assignment ──────────────────────────────────────────────────────

def assign_lung_zone(mask_finding: np.ndarray, mask_lung: np.ndarray,
                     n_zones: int = 3) -> dict[str, float]:
    """Fraction of finding area in each horizontal lung zone."""
    ys_lung, _ = np.where(mask_lung > 0)
    if len(ys_lung) == 0:
        return {}
    y_min, y_max = int(ys_lung.min()), int(ys_lung.max())
    zone_h = (y_max - y_min) / n_zones
    labels = ["upper", "middle", "lower"][:n_zones]
    total = float(mask_finding.sum()) + 1e-6
    result = {}
    for i, lbl in enumerate(labels):
        ys_start = int(y_min + i * zone_h)
        ys_end = int(y_min + (i + 1) * zone_h)
        zone_mask = np.zeros_like(mask_finding, dtype=bool)
        zone_mask[ys_start:ys_end, :] = True
        zone_mask &= (mask_lung > 0)
        result[lbl] = float(np.logical_and(mask_finding > 0, zone_mask).sum()) / total
    return result


# ── Distribution classification ───────────────────────────────────────────────

def classify_distribution(mask_finding: np.ndarray,
                           mask_left_lung: np.ndarray,
                           mask_right_lung: np.ndarray,
                           pixel_spacing_mm: float = 0.28,
                           peripheral_fraction: float = 0.3) -> dict:
    from scipy.ndimage import binary_erosion
    total_lung = float((mask_left_lung + mask_right_lung).clip(0, 1).sum())
    finding_area = float(mask_finding.sum())
    fraction = finding_area / (total_lung + 1e-6)

    left_ov  = float(np.logical_and(mask_finding > 0, mask_left_lung > 0).sum())
    right_ov = float(np.logical_and(mask_finding > 0, mask_right_lung > 0).sum())
    total_ov = left_ov + right_ov + 1e-6
    left_frac, right_frac = left_ov / total_ov, right_ov / total_ov

    if left_frac > 0.85:
        laterality = "unilateral (left)"
    elif right_frac > 0.85:
        laterality = "unilateral (right)"
    else:
        laterality = "bilateral"

    labeled, n_comp = nd_label(mask_finding > 0)
    if fraction > 0.5:
        pattern = "diffuse"
    elif n_comp > 4:
        pattern = "multilobular"
    elif n_comp > 1:
        pattern = "multifocal"
    elif fraction < 0.15:
        pattern = "focal"
    else:
        pattern = "localized"

    total_lung_mask = (mask_left_lung + mask_right_lung).clip(0, 1).astype(bool)
    radius = int(np.sqrt(total_lung / np.pi))
    inner = binary_erosion(total_lung_mask, iterations=max(1, int(radius * peripheral_fraction)))
    central_ov   = float(np.logical_and(mask_finding > 0, inner).sum())
    peripheral_ov = float(np.logical_and(mask_finding > 0, ~inner & total_lung_mask).sum())
    if central_ov > peripheral_ov * 2:
        cp = "central"
    elif peripheral_ov > central_ov * 2:
        cp = "peripheral"
    else:
        cp = "central and peripheral"

    return {
        "pattern": pattern,
        "laterality": laterality,
        "central_peripheral": cp,
        "fraction_of_lung": round(fraction, 3),
        "n_components": n_comp,
        "full_description": f"{pattern}, {laterality}, {cp}",
        "confidence": round(min(0.95, 0.5 + fraction * 0.5), 3),
    }


# ── Containment ───────────────────────────────────────────────────────────────

def check_containment(mask_finding: np.ndarray, mask_lung: np.ndarray) -> dict:
    in_lung = float(np.logical_and(mask_finding > 0, mask_lung > 0).sum())
    total   = float(mask_finding.sum()) + 1e-6
    ratio   = in_lung / total
    if ratio > 0.95:
        verdict = "fully contained within the lung field"
    elif ratio > 0.70:
        verdict = "predominantly within the lung field, with minor extension beyond the pleural surface"
    else:
        verdict = "extends significantly beyond the pleural surface"
    return {"contained": ratio > 0.95, "containment_ratio": round(ratio, 3), "verdict": verdict}


# ── Border / silhouette ───────────────────────────────────────────────────────

def check_border_involvement(mask_finding: np.ndarray, border_mask: np.ndarray,
                              proximity_px: int = 10) -> dict:
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(border_mask > 0, iterations=proximity_px)
    overlap  = float(np.logical_and(mask_finding > 0, border_mask > 0).sum())
    adjacent = float(np.logical_and(mask_finding > 0, dilated & ~(border_mask > 0)).sum())
    total    = float(mask_finding.sum()) + 1e-6
    ov_ratio = overlap / total
    if ov_ratio > 0.05:
        verdict = "directly involves and obliterates the border (silhouette sign positive)"
        positive = True
    elif adjacent / total > 0.10:
        verdict = "adjacent to the border without clearly obliterating it"
        positive = False
    else:
        verdict = "does not involve the border (silhouette sign negative)"
        positive = False
    return {
        "silhouette_positive": positive,
        "overlap_ratio": round(ov_ratio, 3),
        "verdict": verdict,
    }
