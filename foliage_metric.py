import cv2
import numpy as np
from typing import Literal, Tuple

VegMethod = Literal["hsv", "exg", "hybrid"]


def _veg_mask_hsv(image_bgr: np.ndarray, tree_mask: np.ndarray) -> np.ndarray:
    """
    Vegetation mask using HSV hue thresholds (green-ish).
    Returns boolean mask where vegetation is detected inside tree_mask.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Green hue range (tune if needed)
    veg = (h > 35) & (h < 85)

    # Optional guardrails (helps shadows / low-signal pixels)
    veg &= (s >= 40)      # avoid grey-ish pixels
    veg &= (v >= 30)      # avoid very dark pixels

    veg &= (tree_mask > 0)
    return veg


def _veg_mask_exg(image_bgr: np.ndarray, tree_mask: np.ndarray) -> np.ndarray:
    """
    Vegetation mask using Excess Green index:
      ExG = 2G - R - B
    Threshold is adaptive (Otsu) computed inside the tree mask region.
    Returns boolean mask where vegetation is detected inside tree_mask.
    """
    b, g, r = cv2.split(image_bgr.astype(np.int16))
    exg = (2 * g - r - b).astype(np.int16)

    # Work only inside the tree mask for thresholding
    region = exg[tree_mask > 0]
    if region.size == 0:
        return np.zeros(tree_mask.shape[:2], dtype=bool)

    # Normalize to 0..255 so Otsu works robustly
    rmin, rmax = int(region.min()), int(region.max())
    if rmax <= rmin:
        return np.zeros(tree_mask.shape[:2], dtype=bool)

    exg_norm = ((exg - rmin) * 255.0 / (rmax - rmin)).clip(0, 255).astype(np.uint8)

    # Otsu threshold computed on masked pixels only:
    masked_vals = exg_norm[tree_mask > 0].reshape(-1, 1)  # ensure Nx1
    thr_val, _ = cv2.threshold(masked_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    veg = (exg_norm > int(thr_val)) & (tree_mask > 0)
    return veg



def foliage_metric(
    image_bgr: np.ndarray,
    tree_mask: np.ndarray,
    method: VegMethod = "hsv",
) -> float:
    """
    Compute a vegetation metric inside tree_mask.

    metric = mean((S/255)*(V/255)) * veg_pixel_count

    method:
      - "hsv": vegetation pixels come from HSV green thresholds
      - "exg": vegetation pixels come from Excess Green index (adaptive threshold)
      - "hybrid": intersection of hsv AND exg (more conservative / fewer false positives)
    """
    if method not in ("hsv", "exg", "hybrid"):
        raise ValueError("method must be one of: 'hsv', 'exg', 'hybrid'")

    if method == "hsv":
        veg = _veg_mask_hsv(image_bgr, tree_mask)
    elif method == "exg":
        veg = _veg_mask_exg(image_bgr, tree_mask)
    else:
        veg = _veg_mask_hsv(image_bgr, tree_mask) & _veg_mask_exg(image_bgr, tree_mask)

    veg_count = int(veg.sum())
    if veg_count == 0:
        return 0.0

    # Weight by saturation and brightness (your original idea)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    weights = (s[veg] / 255.0) * (v[veg] / 255.0)
    return float(np.mean(weights) * veg_count)
