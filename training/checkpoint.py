"""Checkpoint management for training experiments."""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any


class CheckpointManager:
    """Manages saving/loading model checkpoints and tracking best metrics."""

    def __init__(self, save_dir: str, metrics: list = ["psnr", "ssim"]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.metrics = metrics
        self.best_values = {m: -float("inf") for m in metrics}
        self.best_epochs = {m: -1 for m in metrics}

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict,
        is_last: bool = True,
    ) -> Dict[str, bool]:
        """Save checkpoint and track best metrics.

        Returns:
            Dict indicating which 'best' checkpoints were updated.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {k: float(v) for k, v in metrics.items()},
            "config": config,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        updated = {}

        # Save last
        if is_last:
            torch.save(state, self.checkpoints_dir / "last.pth")

        # Check and save best for each tracked metric
        for m in self.metrics:
            if m in metrics and float(metrics[m]) > self.best_values[m]:
                self.best_values[m] = float(metrics[m])
                self.best_epochs[m] = epoch
                torch.save(state, self.checkpoints_dir / f"best_{m}.pth")
                updated[m] = True
            else:
                updated[m] = False

        # Save metadata
        meta = {
            "best_values": {k: float(v) for k, v in self.best_values.items()},
            "best_epochs": self.best_epochs,
            "last_epoch": epoch,
            "last_metrics": {k: float(v) for k, v in metrics.items()},
        }
        with open(self.save_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return updated

    def load_checkpoint(
        self, path: str, model: torch.nn.Module, device: str = "cpu"
    ) -> Dict:
        """Load a checkpoint into model."""
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        return state

    def get_best_checkpoint_path(self, metric: str = "psnr") -> str:
        """Get path to best checkpoint for a given metric."""
        return str(self.checkpoints_dir / f"best_{metric}.pth")
