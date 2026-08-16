"""Dataset classes for GT and NoisyLR image loading."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class GTDataset(Dataset):
    """Dataset for clean GT images (256x256 float32)."""

    def __init__(
        self,
        gt_dir: str,
        indices: Optional[List[int]] = None,
        patch_size: Optional[int] = None,
        augment: bool = False,
        transform=None,
    ):
        self.gt_dir = Path(gt_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.transform = transform

        # Discover all .npy files
        all_files = sorted(self.gt_dir.glob("*.npy"))
        if indices is not None:
            self.files = [all_files[i] for i in indices]
        else:
            self.files = all_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = np.load(self.files[idx]).astype(np.float32)

        if self.patch_size is not None:
            img = self._random_crop(img, self.patch_size)

        if self.augment:
            img = self._augment(img)

        if self.transform:
            img = self.transform(img)

        # Add channel dimension: (H, W) -> (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0)
        return img

    def _random_crop(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape
        top = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return img[top:top + size, left:left + size]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
        # Random vertical flip
        if np.random.random() < 0.5:
            img = np.flip(img, axis=0).copy()
        # Random 90 degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k).copy()
        return img


class NoisyLRDataset(Dataset):
    """Dataset for NoisyLR images (128x128 float32, raw values preserved)."""

    def __init__(self, lr_dir: str):
        self.lr_dir = Path(lr_dir)
        self.files = sorted(self.lr_dir.glob("*.npy"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        filepath = self.files[idx]
        # NEVER clip - preserve raw range
        img = np.load(filepath).astype(np.float32)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        return img_tensor, filepath.stem


class SyntheticPairDataset(Dataset):
    """Dataset that generates synthetic LR from GT using degradation pipeline."""

    def __init__(
        self,
        gt_dir: str,
        indices: Optional[List[int]] = None,
        patch_size_gt: int = 128,
        degradation_fn=None,
        augment: bool = True,
        config: Optional[Dict] = None,
    ):
        self.gt_dir = Path(gt_dir)
        self.patch_size_gt = patch_size_gt
        self.degradation_fn = degradation_fn
        self.augment = augment
        self.config = config or {}

        all_files = sorted(self.gt_dir.glob("*.npy"))
        if indices is not None:
            self.files = [all_files[i] for i in indices]
        else:
            self.files = all_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt = np.load(self.files[idx]).astype(np.float32)

        # Random crop GT patch
        if self.patch_size_gt and self.patch_size_gt < gt.shape[0]:
            h, w = gt.shape
            top = np.random.randint(0, h - self.patch_size_gt + 1)
            left = np.random.randint(0, w - self.patch_size_gt + 1)
            gt = gt[top:top + self.patch_size_gt, left:left + self.patch_size_gt]

        # Augmentation
        if self.augment:
            if np.random.random() < 0.5:
                gt = np.flip(gt, axis=1).copy()
            if np.random.random() < 0.5:
                gt = np.flip(gt, axis=0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                gt = np.rot90(gt, k).copy()

        # Generate synthetic degraded LR
        if self.degradation_fn is not None:
            rng = np.random.default_rng()
            lr = self.degradation_fn(gt, rng, self.config)
        else:
            # Fallback: simple bicubic downsampling
            from scipy.ndimage import zoom
            lr = zoom(gt, 0.5, order=3)

        # Convert to tensors: (1, H, W)
        gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0)
        lr_tensor = torch.from_numpy(lr.copy()).unsqueeze(0)

        return lr_tensor, gt_tensor


def verify_dataset_integrity(gt_dir: str, lr_dir: str) -> Dict:
    """Verify dataset integrity and return statistics."""
    gt_path = Path(gt_dir)
    lr_path = Path(lr_dir)

    gt_files = sorted(gt_path.glob("*.npy"))
    lr_files = sorted(lr_path.glob("*.npy"))

    results = {
        "gt_count": len(gt_files),
        "lr_count": len(lr_files),
        "gt_shape": None,
        "lr_shape": None,
        "gt_dtype": None,
        "lr_dtype": None,
        "gt_range": [None, None],
        "lr_range": [None, None],
        "issues": [],
    }

    # Check GT
    if len(gt_files) > 0:
        sample_gt = np.load(gt_files[0])
        results["gt_shape"] = list(sample_gt.shape)
        results["gt_dtype"] = str(sample_gt.dtype)

        gt_min, gt_max = float("inf"), float("-inf")
        for f in gt_files:
            img = np.load(f)
            if img.shape != (256, 256):
                results["issues"].append(f"GT {f.name}: unexpected shape {img.shape}")
            if img.dtype != np.float32:
                results["issues"].append(f"GT {f.name}: unexpected dtype {img.dtype}")
            gt_min = min(gt_min, img.min())
            gt_max = max(gt_max, img.max())
        results["gt_range"] = [float(gt_min), float(gt_max)]

    # Check LR
    if len(lr_files) > 0:
        sample_lr = np.load(lr_files[0])
        results["lr_shape"] = list(sample_lr.shape)
        results["lr_dtype"] = str(sample_lr.dtype)

        lr_min, lr_max = float("inf"), float("-inf")
        for f in lr_files:
            img = np.load(f)
            if img.shape != (128, 128):
                results["issues"].append(f"LR {f.name}: unexpected shape {img.shape}")
            if img.dtype != np.float32:
                results["issues"].append(f"LR {f.name}: unexpected dtype {img.dtype}")
            lr_min = min(lr_min, img.min())
            lr_max = max(lr_max, img.max())
        results["lr_range"] = [float(lr_min), float(lr_max)]

    return results
