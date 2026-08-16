"""Loss functions for evidence-preserving image restoration.

Combined loss:
    L_total = λ1*L_charbonnier + λ2*L_ssim + λ3*L_gradient + λ4*L_frequency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class CharbonnierLoss(nn.Module):
    """Charbonnier loss - robust to outliers, smoother than L1.

    L = sqrt((pred - target)^2 + epsilon^2)
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


class MSELoss(nn.Module):
    """MSE loss - directly optimizes PSNR."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss.

    L_ssim = 1 - SSIM(pred, target)
    """

    def __init__(self, window_size: int = 11, channels: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.window = self._create_window(window_size, channels)

    def _create_window(self, size: int, channels: int) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        sigma = 1.5
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size)
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force float32 for numerical stability with AMP
        pred_f32 = pred.float()
        target_f32 = target.float()
        window = self.window.to(pred.device, torch.float32)
        channels = pred_f32.shape[1]

        mu1 = F.conv2d(pred_f32, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target_f32, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred_f32 * pred_f32, window, padding=self.window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target_f32 * target_f32, window, padding=self.window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred_f32 * target_f32, window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        # Clamp to avoid negative variances from numerical precision
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class GradientLoss(nn.Module):
    """Gradient loss to preserve edges and thin structures.

    Compares spatial gradients (dx, dy) between prediction and target.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force float32 for stability
        pred_f32 = pred.float()
        target_f32 = target.float()
        # Compute gradients
        pred_dx = pred_f32[:, :, :, 1:] - pred_f32[:, :, :, :-1]
        pred_dy = pred_f32[:, :, 1:, :] - pred_f32[:, :, :-1, :]
        target_dx = target_f32[:, :, :, 1:] - target_f32[:, :, :, :-1]
        target_dy = target_f32[:, :, 1:, :] - target_f32[:, :, :-1, :]

        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)

        return loss_dx + loss_dy


class FrequencyLoss(nn.Module):
    """Frequency-domain loss with band weighting.

    Compares radial power spectra with conservative high-frequency weighting.
    """

    def __init__(
        self,
        low_weight: float = 1.0,
        mid_weight: float = 1.0,
        high_weight: float = 0.3,
    ):
        super().__init__()
        self.low_weight = low_weight
        self.mid_weight = mid_weight
        self.high_weight = high_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT requires float32 — cast up from float16 if needed (AMP safety)
        pred_f32 = pred.float()
        target_f32 = target.float()
        pred_fft = torch.fft.fft2(pred_f32, norm="ortho")
        target_fft = torch.fft.fft2(target_f32, norm="ortho")

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        # Create frequency band masks
        b, c, h, w = pred.shape
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=pred.device).float() - cy
        x = torch.arange(w, device=pred.device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx ** 2 + yy ** 2)
        max_r = max(cy, cx)

        # Band masks
        low_mask = (radius <= max_r * 0.2).float()
        mid_mask = ((radius > max_r * 0.2) & (radius <= max_r * 0.6)).float()
        high_mask = (radius > max_r * 0.6).float()

        # Weighted frequency loss
        diff = (pred_mag - target_mag) ** 2
        loss_low = (diff * low_mask).mean() * self.low_weight
        loss_mid = (diff * mid_mask).mean() * self.mid_weight
        loss_high = (diff * high_mask).mean() * self.high_weight

        return loss_low + loss_mid + loss_high


class CombinedLoss(nn.Module):
    """Combined loss function with configurable weights."""

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 0.15,
        gradient_weight: float = 0.05,
        frequency_weight: float = 0.02,
        mse_weight: float = 0.0,
        charbonnier_eps: float = 1e-6,
        ssim_window_size: int = 11,
    ):
        super().__init__()
        self.weights = {
            "charbonnier": charbonnier_weight,
            "ssim": ssim_weight,
            "gradient": gradient_weight,
            "frequency": frequency_weight,
            "mse": mse_weight,
        }
        self.charbonnier = CharbonnierLoss(charbonnier_eps)
        self.mse = MSELoss()
        self.ssim = SSIMLoss(ssim_window_size)
        self.gradient = GradientLoss()
        self.frequency = FrequencyLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        if self.weights["charbonnier"] > 0:
            l = self.charbonnier(pred, target)
            losses["charbonnier"] = l
            total = total + self.weights["charbonnier"] * l

        if self.weights["mse"] > 0:
            l = self.mse(pred, target)
            losses["mse"] = l
            total = total + self.weights["mse"] * l

        if self.weights["ssim"] > 0:
            l = self.ssim(pred, target)
            losses["ssim"] = l
            total = total + self.weights["ssim"] * l

        if self.weights["gradient"] > 0:
            l = self.gradient(pred, target)
            losses["gradient"] = l
            total = total + self.weights["gradient"] * l

        if self.weights["frequency"] > 0:
            l = self.frequency(pred, target)
            losses["frequency"] = l
            total = total + self.weights["frequency"] * l

        losses["total"] = total
        return losses
