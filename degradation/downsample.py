"""Downsampling degradation component."""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def apply_downsample(
    img: np.ndarray,
    scale: int = 2,
    method: str = "gaussian_prefilter",
) -> np.ndarray:
    """Apply 2x downsampling with optional Gaussian prefilter.

    Args:
        img: Input image (H, W) float32.
        scale: Downsampling factor.
        method: Downsampling method ('gaussian_prefilter' or 'direct').

    Returns:
        Downsampled image (H//scale, W//scale).
    """
    if method == "gaussian_prefilter":
        # Anti-aliasing Gaussian filter before subsampling
        prefilter_sigma = 0.5 * scale / np.pi
        filtered = gaussian_filter(img, sigma=prefilter_sigma)
        # Subsample
        downsampled = filtered[::scale, ::scale]
    elif method == "direct":
        downsampled = img[::scale, ::scale]
    elif method == "bicubic":
        downsampled = zoom(img, 1.0 / scale, order=3)
    else:
        raise ValueError(f"Unknown downsampling method: {method}")

    return downsampled.astype(np.float32)
