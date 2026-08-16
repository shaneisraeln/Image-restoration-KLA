"""Signal-dependent Gaussian noise."""

import numpy as np


def signal_dependent_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    scale_min: float = 0.3,
    scale_max: float = 2.0,
) -> np.ndarray:
    """Apply signal-dependent noise based on empirical model.

    Model: sigma(x) = scale * (0.11 * sqrt(max(x, 0)) + 0.008)

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator.
        scale_min: Minimum noise scale factor.
        scale_max: Maximum noise scale factor.

    Returns:
        Noisy image (not clipped - raw values preserved).
    """
    scale = rng.uniform(scale_min, scale_max)
    # Signal-dependent sigma map
    sigma_map = scale * (0.11 * np.sqrt(np.maximum(img, 0)) + 0.008)
    noise = rng.normal(0, 1, img.shape).astype(np.float32) * sigma_map
    return (img + noise).astype(np.float32)


def additive_gaussian_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    sigma_min: float = 0.002,
    sigma_max: float = 0.05,
) -> np.ndarray:
    """Apply additive Gaussian noise with random sigma.

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator.
        sigma_min: Minimum noise standard deviation.
        sigma_max: Maximum noise standard deviation.

    Returns:
        Noisy image (not clipped).
    """
    sigma = rng.uniform(sigma_min, sigma_max)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return (img + noise).astype(np.float32)
