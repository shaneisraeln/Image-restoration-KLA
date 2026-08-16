"""Gaussian blur degradation component."""

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_blur(
    img: np.ndarray,
    rng: np.random.Generator,
    sigma_min: float = 0.7,
    sigma_max: float = 1.3,
) -> np.ndarray:
    """Apply Gaussian blur with random sigma.

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator for reproducibility.
        sigma_min: Minimum blur sigma.
        sigma_max: Maximum blur sigma.

    Returns:
        Blurred image.
    """
    sigma = rng.uniform(sigma_min, sigma_max)
    return gaussian_filter(img, sigma=sigma).astype(np.float32)
