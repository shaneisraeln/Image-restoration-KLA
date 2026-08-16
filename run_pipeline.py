"""Full pipeline runner: split -> calibrate -> train -> validate -> inference.

Usage:
    python run_pipeline.py --config configs/noise_aware.yaml
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_split(config: dict):
    """Phase 2: Create train/val split."""
    from data.split import create_split
    ds_cfg = config["dataset"]
    split_dir = ds_cfg.get("split_dir", "splits")
    if not (Path(split_dir) / "split_metadata.json").exists():
        print("\n=== Phase 2: Creating train/val split ===")
        create_split(ds_cfg["gt_dir"], split_dir, seed=config.get("seed", 42))
    else:
        print("\n=== Phase 2: Split already exists ===")


def run_calibration(config: dict):
    """Phase 3: Degradation calibration."""
    from degradation.calibrate import calibrate_degradation
    ds_cfg = config["dataset"]
    print("\n=== Phase 3: Degradation Calibration ===")
    calibrate_degradation(
        ds_cfg["gt_dir"],
        ds_cfg.get("noisy_lr_dir", "NoisyLR"),
        output_dir="reports/degradation_calibration",
    )


def run_baselines(config: dict):
    """Phase 4: Run baselines (bicubic)."""
    from data.split import load_split
    from training.metrics import compute_all_metrics
    from scipy.ndimage import zoom

    print("\n=== Phase 4: Bicubic Baseline ===")
    ds_cfg = config["dataset"]
    _, val_indices = load_split(ds_cfg.get("split_dir", "splits"))
    gt_files = sorted(Path(ds_cfg["gt_dir"]).glob("*.npy"))

    # Generate synthetic LR from validation GT, upscale with bicubic
    rng = np.random.default_rng(config.get("seed", 42))
    all_metrics = []

    from degradation.pipeline import degradation_pipeline
    for idx in val_indices[:50]:  # Quick eval on subset
        gt = np.load(gt_files[idx]).astype(np.float32)
        lr = degradation_pipeline(gt, rng, config)
        # Bicubic upscale
        bicubic = zoom(lr, 2.0, order=3)
        bicubic = np.clip(bicubic, 0.0, 1.0)
        m = compute_all_metrics(bicubic, gt)
        all_metrics.append(m)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print(f"Bicubic baseline: PSNR={avg['psnr']:.2f}, SSIM={avg['ssim']:.4f}")

    # Save
    os.makedirs("reports/baselines", exist_ok=True)
    with open("reports/baselines/bicubic.json", "w") as f:
        json.dump(avg, f, indent=2)


def run_training(config: dict):
    """Phase 5-6: Training."""
    print("\n=== Phase 5-6: Training ===")
    # Import and run training main
    import train
    sys.argv = ["train.py", "--config", config_path]
    train.main()


def run_inference(config: dict):
    """Phase 9: Final inference on real NoisyLR."""
    print("\n=== Phase 9: Final Inference ===")
    ds_cfg = config["dataset"]
    ckpt_cfg = config.get("checkpointing", {})
    save_dir = ckpt_cfg.get("save_dir", f"experiments/{config['model']['name']}")
    checkpoint = os.path.join(save_dir, "checkpoints", "best_psnr.pth")

    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        print("Run training first.")
        return

    sys.argv = [
        "evaluate.py",
        "--input", ds_cfg.get("noisy_lr_dir", "NoisyLR"),
        "--output", "outputs/restored",
        "--checkpoint", checkpoint,
    ]
    import evaluate
    evaluate.main()


def main():
    global config_path
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "split", "calibrate", "baselines", "train", "inference"])
    args = parser.parse_args()
    config_path = args.config

    config = load_config(args.config)

    if args.phase == "all":
        run_split(config)
        run_calibration(config)
        run_baselines(config)
        run_training(config)
        run_inference(config)
    elif args.phase == "split":
        run_split(config)
    elif args.phase == "calibrate":
        run_calibration(config)
    elif args.phase == "baselines":
        run_baselines(config)
    elif args.phase == "train":
        run_training(config)
    elif args.phase == "inference":
        run_inference(config)


if __name__ == "__main__":
    main()
