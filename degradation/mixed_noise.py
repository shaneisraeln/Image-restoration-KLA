"""Mixed noise combining additive and multiplicative components."""

import numpy as np
from .gaussian_noise import additive_gaussian_noise
from .multiplicative_noise import multiplicative_noise


def mixed_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    additive_sigma_min: float = 0.002,
    additive_sigma_max: float = 0.05,
    mult_alpha_min: float = 0.02,
    mult_alpha_max: float = 0.12,
) -> np.ndarray:
    """Apply mixed additive + multiplicative noise.

    Model: noise = additive_noise + multiplicative_noise

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator.
        additive_sigma_min: Min additive sigma.
        additive_sigma_max: Max additive sigma.
        mult_alpha_min: Min multiplicative alpha.
        mult_alpha_max: Max multiplicative alpha.

    Returns:
        Noisy image (not clipped).
    """
    # Apply multiplicative noise first
    noisy = multiplicative_noise(img, rng, mult_alpha_min, mult_alpha_max)
    # Then add additive component
    sigma = rng.uniform(additive_sigma_min, additive_sigma_max)
    additive = rng.normal(0, sigma, img.shape).astype(np.float32)
    return (noisy + additive).astype(np.float32)
