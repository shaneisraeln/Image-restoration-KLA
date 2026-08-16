"""Heavy-tailed noise components (Student-t, Laplace, outlier mixture)."""

import numpy as np
from scipy.stats import t as student_t


def heavy_tail_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.05,
    df_min: float = 3.0,
    df_max: float = 10.0,
    distribution: str = "student_t",
) -> np.ndarray:
    """Apply heavy-tailed noise.

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator.
        sigma: Noise scale.
        df_min: Min degrees of freedom (Student-t).
        df_max: Max degrees of freedom (Student-t).
        distribution: 'student_t' or 'laplace'.

    Returns:
        Noisy image (not clipped).
    """
    if distribution == "student_t":
        df = rng.uniform(df_min, df_max)
        # Generate Student-t samples
        noise = student_t.rvs(df, size=img.shape, random_state=int(rng.integers(2**31)))
        noise = noise.astype(np.float32) * sigma
    elif distribution == "laplace":
        noise = rng.laplace(0, sigma / np.sqrt(2), img.shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return (img + noise).astype(np.float32)


def outlier_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.01,
    magnitude: float = 0.3,
) -> np.ndarray:
    """Add sparse outlier noise (salt-and-pepper like).

    Args:
        img: Input image (H, W) float32.
        rng: NumPy random generator.
        probability: Probability of each pixel being an outlier.
        magnitude: Magnitude of outlier values.

    Returns:
        Image with outlier noise.
    """
    mask = rng.random(img.shape) < probability
    outlier_values = rng.uniform(-magnitude, magnitude, img.shape).astype(np.float32)
    result = img.copy()
    result[mask] += outlier_values[mask]
    return result
