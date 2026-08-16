"""Test-Time Augmentation for inference.

Applies geometric transforms, runs inference, inverts transforms, and averages.
Supported: identity, hflip, vflip, hflip+vflip, rot90, rot180, rot270
"""

import torch
import torch.nn as nn
from typing import List


def apply_tta(
    model: nn.Module,
    x: torch.Tensor,
    transforms: List[str] = None,
) -> torch.Tensor:
    """Apply TTA and return averaged prediction.

    Args:
        model: Trained model.
        x: Input tensor (B, 1, H, W).
        transforms: List of transform names. If None, uses all 8.

    Returns:
        Averaged prediction (B, 1, 2H, 2W).
    """
    if transforms is None:
        transforms = [
            "identity", "hflip", "vflip", "hvflip",
            "rot90", "rot180", "rot270",
        ]

    predictions = []

    for t in transforms:
        # Apply forward transform
        x_aug = _forward_transform(x, t)
        # Predict
        pred = model(x_aug)
        # Invert transform on prediction
        pred_inv = _inverse_transform(pred, t)
        predictions.append(pred_inv)

    # Average predictions
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0)


def _forward_transform(x: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply geometric transform."""
    if transform == "identity":
        return x
    elif transform == "hflip":
        return torch.flip(x, dims=[3])
    elif transform == "vflip":
        return torch.flip(x, dims=[2])
    elif transform == "hvflip":
        return torch.flip(x, dims=[2, 3])
    elif transform == "rot90":
        return torch.rot90(x, k=1, dims=[2, 3])
    elif transform == "rot180":
        return torch.rot90(x, k=2, dims=[2, 3])
    elif transform == "rot270":
        return torch.rot90(x, k=3, dims=[2, 3])
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _inverse_transform(x: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply inverse geometric transform."""
    if transform == "identity":
        return x
    elif transform == "hflip":
        return torch.flip(x, dims=[3])
    elif transform == "vflip":
        return torch.flip(x, dims=[2])
    elif transform == "hvflip":
        return torch.flip(x, dims=[2, 3])
    elif transform == "rot90":
        return torch.rot90(x, k=3, dims=[2, 3])  # inverse of rot90 is rot270
    elif transform == "rot180":
        return torch.rot90(x, k=2, dims=[2, 3])
    elif transform == "rot270":
        return torch.rot90(x, k=1, dims=[2, 3])  # inverse of rot270 is rot90
    else:
        raise ValueError(f"Unknown transform: {transform}")
