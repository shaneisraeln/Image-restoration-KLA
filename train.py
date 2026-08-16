"""Main training script for the restoration system.

Usage:
    python train.py --config configs/noise_aware.yaml
"""

import argparse
import json
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from data.dataset import SyntheticPairDataset
from data.split import load_split, create_split
from degradation.pipeline import degradation_pipeline
from models.nafnet import NAFNet, NoiseAwareNAFNet
from models.unet import SmallUNet
from models.losses import CombinedLoss
from training.metrics import compute_batch_metrics
from training.checkpoint import CheckpointManager


def load_config(path: str) -> Dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(config: Dict) -> torch.device:
    """Determine compute device."""
    device_cfg = config.get("device", "auto")
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_model(config: Dict) -> nn.Module:
    """Build model from configuration."""
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name == "nafnet":
        return NAFNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            width=model_cfg.get("width", 48),
            num_blocks=model_cfg.get("num_blocks", [2, 4, 8, 8]),
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
        )
    elif name == "noise_aware_nafnet":
        nc = model_cfg.get("noise_conditioning", {})
        return NoiseAwareNAFNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            width=model_cfg.get("width", 48),
            num_blocks=model_cfg.get("num_blocks", [2, 4, 8, 8]),
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            noise_mode=nc.get("mode", "spatial"),
        )
    elif name == "small_unet":
        return SmallUNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            base_features=model_cfg.get("base_features", 32),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def build_loss(config: Dict) -> CombinedLoss:
    """Build loss function from configuration."""
    loss_cfg = config.get("loss", {})
    return CombinedLoss(
        charbonnier_weight=loss_cfg.get("charbonnier", {}).get("weight", 1.0),
        ssim_weight=loss_cfg.get("ssim", {}).get("weight", 0.15),
        gradient_weight=loss_cfg.get("gradient", {}).get("weight", 0.05),
        frequency_weight=loss_cfg.get("frequency", {}).get("weight", 0.02),
        mse_weight=loss_cfg.get("mse", {}).get("weight", 0.0),
        charbonnier_eps=loss_cfg.get("charbonnier", {}).get("epsilon", 1e-6),
    )


def build_optimizer(model: nn.Module, config: Dict):
    """Build optimizer from configuration."""
    opt_cfg = config["training"]["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 0.0002),
        weight_decay=opt_cfg.get("weight_decay", 0.0001),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(optimizer, config: Dict):
    """Build learning rate scheduler."""
    sched_cfg = config["training"].get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")
    epochs = config["training"]["epochs"]
    warmup = sched_cfg.get("warmup_epochs", 5)

    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup,
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
        return scheduler
    return None


def degradation_fn(gt: np.ndarray, rng: np.random.Generator, config: Dict) -> np.ndarray:
    """Wrapper for degradation pipeline."""
    return degradation_pipeline(gt, rng, config)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    n_batches = 0

    for lr_batch, gt_batch in tqdm(dataloader, desc="Training", leave=False):
        lr_batch = lr_batch.to(device, non_blocking=True)
        gt_batch = gt_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast('cuda'):
                pred = model(lr_batch)
                losses = criterion(pred, gt_batch)
            # Skip NaN losses to prevent model corruption
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                continue
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr_batch)
            losses = criterion(pred, gt_batch)
            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                continue
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0.0
            epoch_losses[k] += v.item()
        n_batches += 1

    # Average
    if n_batches == 0:
        return {"total": float("nan")}
    return {k: v / n_batches for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation."""
    model.eval()
    all_metrics = []
    total_loss = 0.0
    n_batches = 0

    for lr_batch, gt_batch in tqdm(dataloader, desc="Validation", leave=False):
        lr_batch = lr_batch.to(device, non_blocking=True)
        gt_batch = gt_batch.to(device, non_blocking=True)

        pred = model(lr_batch)
        losses = criterion(pred, gt_batch)
        total_loss += losses["total"].item()

        # Compute metrics
        metrics = compute_batch_metrics(pred, gt_batch)
        all_metrics.append(metrics)
        n_batches += 1

    # Average metrics
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    avg_metrics["val_loss"] = total_loss / max(n_batches, 1)
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train restoration model")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--finetune", type=str, default=None, help="Load model weights only (no optimizer/epoch state)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = get_device(config)
    seed = config.get("seed", 42)

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # Dataset paths
    ds_cfg = config["dataset"]
    gt_dir = ds_cfg["gt_dir"]
    split_dir = ds_cfg.get("split_dir", "splits")

    # Ensure split exists
    split_file = Path(split_dir) / "split_metadata.json"
    if not split_file.exists():
        print("Creating train/val split...")
        create_split(gt_dir, split_dir, seed=seed)

    train_indices, val_indices = load_split(split_dir)
    print(f"Train: {len(train_indices)} images, Val: {len(val_indices)} images")

    # Build datasets
    train_cfg = config["training"]
    train_dataset = SyntheticPairDataset(
        gt_dir=gt_dir,
        indices=train_indices,
        patch_size_gt=train_cfg.get("patch_size_gt", 128),
        degradation_fn=degradation_fn,
        augment=True,
        config=config,
    )
    val_dataset = SyntheticPairDataset(
        gt_dir=gt_dir,
        indices=val_indices,
        patch_size_gt=256,  # Full image for validation
        degradation_fn=degradation_fn,
        augment=False,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("validation", {}).get("batch_size", 4),
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    # Build model
    model = build_model(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['name']}, Parameters: {param_count:,}")

    # Build optimizer, scheduler, loss
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_loss(config).to(device)

    # AMP
    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler('cuda') if use_amp else None

    # Checkpoint manager
    ckpt_cfg = config.get("checkpointing", {})
    save_dir = ckpt_cfg.get("save_dir", f"experiments/{config['model']['name']}")
    ckpt_manager = CheckpointManager(save_dir, metrics=ckpt_cfg.get("metrics", ["psnr", "ssim"]))

    # Save config
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(os.path.join(save_dir, "seed.txt"), "w") as f:
        f.write(str(seed))
    with open(os.path.join(save_dir, "model_summary.txt"), "w") as f:
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Parameters: {param_count:,}\n")
        f.write(str(model))

    # Resume or finetune
    start_epoch = 0
    if args.resume:
        state = ckpt_manager.load_checkpoint(args.resume, model, str(device))
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    elif args.finetune:
        state = torch.load(args.finetune, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded weights from {args.finetune} (epoch {state.get('epoch', '?')})")

    # Training loop
    epochs = train_cfg["epochs"]
    val_interval = config.get("validation", {}).get("interval", 5)
    early_stop_cfg = train_cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 30) if early_stop_cfg.get("enabled", False) else float("inf")
    patience_counter = 0
    best_metric = -float("inf")

    train_log = []

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)

        # Step scheduler
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_losses["total"],
            "time": epoch_time,
        }

        # Validate
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device)
            log_entry.update({f"val_{k}": v for k, v in val_metrics.items()})

            # Checkpoint
            updated = ckpt_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, config
            )

            # Early stopping
            metric_val = val_metrics.get("psnr", 0)
            if metric_val > best_metric:
                best_metric = metric_val
                patience_counter = 0
            else:
                patience_counter += val_interval

            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_losses['total']:.4f} | "
                  f"PSNR: {val_metrics.get('psnr', 0):.2f} | "
                  f"SSIM: {val_metrics.get('ssim', 0):.4f} | "
                  f"LR: {lr:.6f} | Time: {epoch_time:.1f}s"
                  + (" [BEST]" if updated.get("psnr", False) else ""))

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_losses['total']:.4f} | "
                  f"LR: {lr:.6f} | Time: {epoch_time:.1f}s")

        train_log.append(log_entry)

    # Save training log
    import pandas as pd
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(os.path.join(save_dir, "train_log.csv"), index=False)

    print(f"\nTraining complete. Best PSNR: {best_metric:.2f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
