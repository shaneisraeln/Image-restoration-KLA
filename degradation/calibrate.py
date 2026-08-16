"""Real-vs-Synthetic degradation calibration.

Compares statistics of real NoisyLR images against synthetic degraded images
to tune degradation parameters.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.ndimage import zoom


def compute_image_statistics(img: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive statistics for a single image."""
    return {
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "median": float(np.median(img)),
        "std": float(img.std()),
        "p1": float(np.percentile(img, 1)),
        "p5": float(np.percentile(img, 5)),
        "p95": float(np.percentile(img, 95)),
        "p99": float(np.percentile(img, 99)),
        "frac_below_0": float((img < 0).mean()),
        "frac_above_1": float((img > 1).mean()),
    }


def compute_frequency_stats(img: np.ndarray) -> Dict[str, float]:
    """Compute frequency-domain statistics."""
    fft = np.fft.fft2(img)
    mag = np.abs(np.fft.fftshift(fft))
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    radius = np.sqrt(x**2 + y**2)
    max_r = max(cy, cx)

    low = radius <= max_r * 0.2
    mid = (radius > max_r * 0.2) & (radius <= max_r * 0.6)
    high = radius > max_r * 0.6

    total_energy = mag.sum() + 1e-10
    return {
        "low_freq_energy": float(mag[low].sum() / total_energy),
        "mid_freq_energy": float(mag[mid].sum() / total_energy),
        "high_freq_energy": float(mag[high].sum() / total_energy),
        "high_freq_mean": float(mag[high].mean()),
    }


def compute_structural_stats(img: np.ndarray) -> Dict[str, float]:
    """Compute structural/edge statistics."""
    grad_x = np.abs(np.diff(img, axis=1))
    grad_y = np.abs(np.diff(img, axis=0))
    grad_mag = np.sqrt(grad_x[:, :-1]**2 + grad_y[:-1, :]**2)

    # Local variance (3x3 patches)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(img, size=3)
    local_sq_mean = uniform_filter(img**2, size=3)
    local_var = local_sq_mean - local_mean**2

    return {
        "edge_density": float((grad_mag > 0.05).mean()),
        "mean_gradient": float(grad_mag.mean()),
        "max_gradient": float(grad_mag.max()),
        "mean_local_variance": float(local_var.mean()),
    }


def estimate_noise_level(img: np.ndarray) -> float:
    """Estimate noise level using MAD of wavelet coefficients."""
    # Simple approach: high-frequency component
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(img, sigma=1.0)
    residual = img - smoothed
    # MAD estimator
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = mad / 0.6745  # Gaussian noise sigma estimator
    return float(sigma)


def calibrate_degradation(
    gt_dir: str,
    lr_dir: str,
    output_dir: str = "reports/degradation_calibration",
    n_samples: int = 100,
    seed: int = 42,
) -> Dict:
    """Run full calibration comparing real NoisyLR vs synthetic.

    Returns calibration report.
    """
    from degradation.pipeline import degradation_pipeline

    gt_path = Path(gt_dir)
    lr_path = Path(lr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_path.glob("*.npy"))
    lr_files = sorted(lr_path.glob("*.npy"))

    rng = np.random.default_rng(seed)

    # Compute real NoisyLR statistics
    print("Computing real NoisyLR statistics...")
    real_stats = []
    real_freq_stats = []
    real_noise_levels = []
    for f in lr_files:
        img = np.load(f).astype(np.float32)
        real_stats.append(compute_image_statistics(img))
        real_freq_stats.append(compute_frequency_stats(img))
        real_noise_levels.append(estimate_noise_level(img))

    # Generate synthetic samples and compute statistics
    print("Generating synthetic samples...")
    default_config = {
        "degradation": {
            "blur": {"enabled": True, "sigma_min": 0.7, "sigma_max": 1.3},
            "downsample": {"scale": 2, "method": "gaussian_prefilter"},
            "noise": {
                "signal_dependent": {"enabled": True, "scale_min": 0.3, "scale_max": 2.0},
                "additive": {"enabled": True, "sigma_min": 0.002, "sigma_max": 0.05},
                "multiplicative": {"enabled": True, "alpha_min": 0.02, "alpha_max": 0.12},
                "heavy_tail": {"enabled": True, "probability": 0.2, "df_min": 3.0, "df_max": 10.0},
                "mix_weights": {
                    "signal_dependent": 0.34,
                    "additive": 0.22,
                    "multiplicative": 0.16,
                    "mixed": 0.26,
                    "low_noise": 0.02,
                },
            },
        }
    }

    sample_indices = rng.choice(len(gt_files), size=min(n_samples, len(gt_files)), replace=False)
    synth_stats = []
    synth_freq_stats = []
    synth_noise_levels = []

    for idx in sample_indices:
        gt = np.load(gt_files[idx]).astype(np.float32)
        synth_lr = degradation_pipeline(gt, rng, default_config)
        synth_stats.append(compute_image_statistics(synth_lr))
        synth_freq_stats.append(compute_frequency_stats(synth_lr))
        synth_noise_levels.append(estimate_noise_level(synth_lr))

    # Compare distributions
    report = {
        "real": {
            "pixel_stats": {
                "mean_of_means": float(np.mean([s["mean"] for s in real_stats])),
                "std_of_means": float(np.std([s["mean"] for s in real_stats])),
                "mean_of_stds": float(np.mean([s["std"] for s in real_stats])),
                "mean_min": float(np.mean([s["min"] for s in real_stats])),
                "mean_max": float(np.mean([s["max"] for s in real_stats])),
                "mean_frac_below_0": float(np.mean([s["frac_below_0"] for s in real_stats])),
                "mean_frac_above_1": float(np.mean([s["frac_above_1"] for s in real_stats])),
            },
            "frequency_stats": {
                "mean_low_freq": float(np.mean([s["low_freq_energy"] for s in real_freq_stats])),
                "mean_mid_freq": float(np.mean([s["mid_freq_energy"] for s in real_freq_stats])),
                "mean_high_freq": float(np.mean([s["high_freq_energy"] for s in real_freq_stats])),
            },
            "noise_stats": {
                "mean_noise_level": float(np.mean(real_noise_levels)),
                "std_noise_level": float(np.std(real_noise_levels)),
                "min_noise_level": float(np.min(real_noise_levels)),
                "max_noise_level": float(np.max(real_noise_levels)),
            },
        },
        "synthetic": {
            "pixel_stats": {
                "mean_of_means": float(np.mean([s["mean"] for s in synth_stats])),
                "std_of_means": float(np.std([s["mean"] for s in synth_stats])),
                "mean_of_stds": float(np.mean([s["std"] for s in synth_stats])),
                "mean_min": float(np.mean([s["min"] for s in synth_stats])),
                "mean_max": float(np.mean([s["max"] for s in synth_stats])),
                "mean_frac_below_0": float(np.mean([s["frac_below_0"] for s in synth_stats])),
                "mean_frac_above_1": float(np.mean([s["frac_above_1"] for s in synth_stats])),
            },
            "frequency_stats": {
                "mean_low_freq": float(np.mean([s["low_freq_energy"] for s in synth_freq_stats])),
                "mean_mid_freq": float(np.mean([s["mid_freq_energy"] for s in synth_freq_stats])),
                "mean_high_freq": float(np.mean([s["high_freq_energy"] for s in synth_freq_stats])),
            },
            "noise_stats": {
                "mean_noise_level": float(np.mean(synth_noise_levels)),
                "std_noise_level": float(np.std(synth_noise_levels)),
                "min_noise_level": float(np.min(synth_noise_levels)),
                "max_noise_level": float(np.max(synth_noise_levels)),
            },
        },
    }

    with open(output_path / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Calibration report saved to {output_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate degradation")
    parser.add_argument("--gt_dir", default="train")
    parser.add_argument("--lr_dir", default="NoisyLR")
    parser.add_argument("--output_dir", default="reports/degradation_calibration")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    calibrate_degradation(args.gt_dir, args.lr_dir, args.output_dir, args.n_samples)
