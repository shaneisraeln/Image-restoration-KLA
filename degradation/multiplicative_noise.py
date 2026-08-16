"""Multiplicative / speckle-like noise."""

import numpy as np


def multiplicative_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    alpha_min: float = 0.02,
    alpha_max: float = 0.12,
) -> np.ndarray:
    """Apply multiplicative (speckle-like) noise.

    Model: noise = x * alpha * epsilon, epsilon ~ N(0,1)

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator.
        alpha_min: Minimum multiplicative factor.
        alpha_max: Maximum multiplicative factor.

    Returns:
        Noisy image (not clipped).
    """
    alpha = rng.uniform(alpha_min, alpha_max)
    epsilon = rng.normal(0, 1, img.shape).astype(np.float32)
    noise = img * alpha * epsilon
    return (img + noise).astype(np.float32)
