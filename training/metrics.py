"""Evaluation metrics for image restoration.

Primary: PSNR, SSIM, MAE, RMSE
Supplementary: Edge preservation, frequency error
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from skimage.metrics import structural_similarity as ski_ssim


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Structural Similarity Index."""
    return ski_ssim(pred, target, data_range=1.0)


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_edge_preservation(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Edge Preservation Score.

    EPS = 1 - normalized_gradient_error
    """
    # Compute gradients
    pred_dx = np.abs(np.diff(pred, axis=1))
    pred_dy = np.abs(np.diff(pred, axis=0))
    target_dx = np.abs(np.diff(target, axis=1))
    target_dy = np.abs(np.diff(target, axis=0))

    # Gradient magnitude (trim to common shape)
    pred_grad = np.sqrt(pred_dx[:-1, :] ** 2 + pred_dy[:, :-1] ** 2)
    target_grad = np.sqrt(target_dx[:-1, :] ** 2 + target_dy[:, :-1] ** 2)

    # Normalized error
    denom = target_grad.sum() + 1e-8
    error = np.abs(pred_grad - target_grad).sum() / denom

    return float(1.0 - min(error, 1.0))


def compute_frequency_error(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute frequency reconstruction error by band."""
    pred_fft = np.fft.fft2(pred)
    target_fft = np.fft.fft2(target)

    pred_mag = np.abs(np.fft.fftshift(pred_fft))
    target_mag = np.abs(np.fft.fftshift(target_fft))

    h, w = pred.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    radius = np.sqrt(x ** 2 + y ** 2)
    max_r = max(cy, cx)

    # Band masks
    low = radius <= max_r * 0.2
    mid = (radius > max_r * 0.2) & (radius <= max_r * 0.6)
    high = radius > max_r * 0.6

    def band_error(mask):
        p = pred_mag[mask]
        t = target_mag[mask]
        if t.sum() < 1e-8:
            return 0.0
        return float(np.abs(p - t).sum() / (t.sum() + 1e-8))

    return {
        "low_freq_error": band_error(low),
        "mid_freq_error": band_error(mid),
        "high_freq_error": band_error(high),
        "total_freq_error": band_error(low | mid | high),
    }


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all metrics for a single image pair."""
    metrics = {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "mae": compute_mae(pred, target),
        "rmse": compute_rmse(pred, target),
        "edge_preservation": compute_edge_preservation(pred, target),
    }
    freq = compute_frequency_error(pred, target)
    metrics.update(freq)
    return metrics


def compute_batch_metrics(
    preds: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """Compute average metrics over a batch of tensors.

    Args:
        preds: (B, 1, H, W) predictions
        targets: (B, 1, H, W) ground truth

    Returns:
        Dictionary of averaged metrics.
    """
    batch_size = preds.shape[0]
    all_metrics = []

    for i in range(batch_size):
        pred_np = preds[i, 0].cpu().numpy()
        target_np = targets[i, 0].cpu().numpy()
        m = compute_all_metrics(pred_np, target_np)
        all_metrics.append(m)

    # Average
    avg = {}
    for key in all_metrics[0]:
        avg[key] = np.mean([m[key] for m in all_metrics])

    return avg
