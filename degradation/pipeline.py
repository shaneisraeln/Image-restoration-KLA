"""Complete degradation pipeline combining all components.

Pipeline:
    GT 256x256 -> blur -> 2x downsample -> noise -> synthetic 128x128 LR
"""

import numpy as np
from typing import Dict, Any, Optional

from .blur import apply_blur
from .downsample import apply_downsample
from .gaussian_noise import signal_dependent_noise, additive_gaussian_noise
from .multiplicative_noise import multiplicative_noise
from .mixed_noise import mixed_noise
from .heavy_tail import heavy_tail_noise, outlier_noise


def degradation_pipeline(
    gt: np.ndarray,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> np.ndarray:
    """Apply full degradation pipeline to a GT image.

    Args:
        gt: Clean GT image (H, W) float32, range [0, 1].
        rng: NumPy random generator for reproducibility.
        config: Degradation configuration dictionary.

    Returns:
        Degraded LR image (H/2, W/2) float32, raw values (NOT clipped).
    """
    degradation_cfg = config.get("degradation", config)
    img = gt.copy()

    # Step 1: Blur
    blur_cfg = degradation_cfg.get("blur", {})
    if blur_cfg.get("enabled", True):
        img = apply_blur(
            img, rng,
            sigma_min=blur_cfg.get("sigma_min", 0.7),
            sigma_max=blur_cfg.get("sigma_max", 1.3),
        )

    # Step 2: Downsample
    ds_cfg = degradation_cfg.get("downsample", {})
    scale = ds_cfg.get("scale", 2)
    method = ds_cfg.get("method", "gaussian_prefilter")
    img = apply_downsample(img, scale=scale, method=method)

    # Step 3: Noise
    noise_cfg = degradation_cfg.get("noise", {})
    img = _apply_noise(img, rng, noise_cfg)

    return img


def _apply_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    noise_cfg: Dict[str, Any],
) -> np.ndarray:
    """Apply noise according to configuration with random type selection."""

    # Determine noise type based on mix weights
    mix_weights = noise_cfg.get("mix_weights", {})
    if mix_weights:
        types = list(mix_weights.keys())
        weights = np.array([mix_weights[t] for t in types], dtype=np.float64)
        weights /= weights.sum()
        noise_type = rng.choice(types, p=weights)
    else:
        noise_type = "signal_dependent"

    if noise_type == "signal_dependent":
        sd_cfg = noise_cfg.get("signal_dependent", {})
        if sd_cfg.get("enabled", True):
            img = signal_dependent_noise(
                img, rng,
                scale_min=sd_cfg.get("scale_min", 0.3),
                scale_max=sd_cfg.get("scale_max", 2.0),
            )

    elif noise_type == "additive":
        add_cfg = noise_cfg.get("additive", {})
        if add_cfg.get("enabled", True):
            img = additive_gaussian_noise(
                img, rng,
                sigma_min=add_cfg.get("sigma_min", 0.002),
                sigma_max=add_cfg.get("sigma_max", 0.05),
            )

    elif noise_type == "multiplicative":
        mult_cfg = noise_cfg.get("multiplicative", {})
        if mult_cfg.get("enabled", True):
            img = multiplicative_noise(
                img, rng,
                alpha_min=mult_cfg.get("alpha_min", 0.02),
                alpha_max=mult_cfg.get("alpha_max", 0.12),
            )

    elif noise_type == "mixed":
        add_cfg = noise_cfg.get("additive", {})
        mult_cfg = noise_cfg.get("multiplicative", {})
        img = mixed_noise(
            img, rng,
            additive_sigma_min=add_cfg.get("sigma_min", 0.002),
            additive_sigma_max=add_cfg.get("sigma_max", 0.05),
            mult_alpha_min=mult_cfg.get("alpha_min", 0.02),
            mult_alpha_max=mult_cfg.get("alpha_max", 0.12),
        )

    elif noise_type == "low_noise":
        # Very low noise - almost clean
        sigma = rng.uniform(0.001, 0.005)
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = img + noise

    # Optional heavy-tail component
    ht_cfg = noise_cfg.get("heavy_tail", {})
    if ht_cfg.get("enabled", False):
        if rng.random() < ht_cfg.get("probability", 0.2):
            # Apply as additional perturbation
            scale = rng.uniform(0.005, 0.02)
            img = heavy_tail_noise(
                img, rng,
                sigma=scale,
                df_min=ht_cfg.get("df_min", 3.0),
                df_max=ht_cfg.get("df_max", 10.0),
            )

    return img


def create_degradation_fn(config: Dict[str, Any]):
    """Create a degradation function closure from config.

    Returns:
        Callable that takes (gt, rng, config) and returns degraded LR.
    """
    def degrade(gt: np.ndarray, rng: np.random.Generator, cfg: Optional[Dict] = None):
        return degradation_pipeline(gt, rng, cfg or config)
    return degrade
