"""Standalone evaluation script for KLA AI-Based Image Restoration Challenge.

Usage:
    python evaluate.py --input /path/to/test_images --output /path/to/output_dir

    Optionally specify checkpoint:
    python evaluate.py --input /path/to/test_images --output /path/to/output_dir --checkpoint /path/to/model.pth

This script:
    1. Loads the trained Noise-Aware NAFNet model
    2. Processes all .npy files in the input directory
    3. Produces 256x256 float32 restored outputs in [0, 1]
    4. Saves outputs to the specified output directory
    5. Reports inference time statistics
    6. Exits with non-zero status on failure

NO manual edits required. Runs on CUDA if available, CPU fallback otherwise.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple

# Default checkpoint path (relative to script location)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = SCRIPT_DIR / "experiments" / "wide_nafnet_64" / "checkpoints" / "best_psnr.pth"


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint. Returns (model, config)."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "noise_aware_nafnet")

    # Add script directory to path for model imports
    sys.path.insert(0, str(SCRIPT_DIR))

    if name == "noise_aware_nafnet":
        from models.nafnet import NoiseAwareNAFNet
        nc = model_cfg.get("noise_conditioning", {})
        model = NoiseAwareNAFNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            width=model_cfg.get("width", 48),
            num_blocks=model_cfg.get("num_blocks", [2, 4, 8, 8]),
            dropout_rate=0.0,
            noise_mode=nc.get("mode", "spatial"),
        )
    elif name == "nafnet":
        from models.nafnet import NAFNet
        model = NAFNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            width=model_cfg.get("width", 48),
            num_blocks=model_cfg.get("num_blocks", [2, 4, 8, 8]),
            dropout_rate=0.0,
        )
    elif name == "small_unet":
        from models.unet import SmallUNet
        model = SmallUNet(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            base_features=model_cfg.get("base_features", 32),
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {name}, Parameters: {param_count:,}")
    return model, config


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 8,
) -> Dict:
    """Run inference on all .npy files in input directory.

    CRITICAL: Input values are preserved (NOT clipped before inference).
    Only final predictions are clamped to [0, 1].
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover input files
    input_files = sorted(input_path.glob("*.npy"))
    n_files = len(input_files)

    if n_files == 0:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    print(f"Processing {n_files} images...")

    # Process in batches
    total_time = 0.0
    processed = 0

    for i in range(0, n_files, batch_size):
        batch_files = input_files[i:i + batch_size]
        batch_imgs = []

        for f in batch_files:
            # Load and preserve raw values - NEVER clip before inference
            img = np.load(f).astype(np.float32)
            batch_imgs.append(img)

        # Stack batch
        batch_np = np.stack(batch_imgs)[:, np.newaxis, :, :]  # (B, 1, H, W)
        batch_tensor = torch.from_numpy(batch_np).to(device, non_blocking=True)

        # Inference with timing
        t0 = time.time()
        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                predictions = model(batch_tensor)
            torch.cuda.synchronize()
        else:
            predictions = model(batch_tensor)
        batch_time = time.time() - t0
        total_time += batch_time

        # Save outputs
        preds_np = predictions.cpu().numpy()
        for j, f in enumerate(batch_files):
            # Final clamp to [0, 1] - only place clipping is allowed
            output = np.clip(preds_np[j, 0], 0.0, 1.0).astype(np.float32)
            np.save(output_path / f.name, output)

        processed += len(batch_files)
        if processed % 50 == 0 or processed == n_files:
            print(f"  Processed {processed}/{n_files} ({batch_time:.3f}s for batch)")

    stats = {
        "total_images": n_files,
        "total_time_seconds": round(total_time, 3),
        "avg_time_per_image_ms": round(total_time / n_files * 1000, 1),
        "throughput_images_per_second": round(n_files / total_time, 1) if total_time > 0 else 0,
        "device": str(device),
        "batch_size": batch_size,
    }
    return stats


def validate_outputs(output_dir: str, expected_count: int) -> Dict:
    """Validate all output files meet the submission contract."""
    output_path = Path(output_dir)
    files = sorted(output_path.glob("*.npy"))

    manifest = {
        "count": len(files),
        "expected_count": expected_count,
        "shape": [256, 256],
        "dtype": "float32",
        "min": float("inf"),
        "max": float("-inf"),
        "nan_count": 0,
        "inf_count": 0,
        "issues": [],
    }

    for f in files:
        img = np.load(f)
        if img.shape != (256, 256):
            manifest["issues"].append(f"{f.name}: shape {img.shape}")
        if img.dtype != np.float32:
            manifest["issues"].append(f"{f.name}: dtype {img.dtype}")
        if np.isnan(img).any():
            manifest["nan_count"] += int(np.isnan(img).sum())
            manifest["issues"].append(f"{f.name}: contains NaN")
        if np.isinf(img).any():
            manifest["inf_count"] += int(np.isinf(img).sum())
            manifest["issues"].append(f"{f.name}: contains Inf")
        manifest["min"] = min(manifest["min"], float(img.min()))
        manifest["max"] = max(manifest["max"], float(img.max()))

    if len(files) != expected_count:
        manifest["issues"].append(f"Expected {expected_count} files, found {len(files)}")
    if manifest["min"] < 0:
        manifest["issues"].append(f"Min value {manifest['min']:.6f} < 0")
    if manifest["max"] > 1:
        manifest["issues"].append(f"Max value {manifest['max']:.6f} > 1")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="KLA AI Image Restoration - Evaluation Script"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Directory containing input .npy test images"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Directory for restored output .npy images"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Path to model checkpoint (default: best model in experiments/)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto/cuda/cpu (default: auto)"
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    # Count input files
    input_files = list(Path(args.input).glob("*.npy"))
    expected_count = len(input_files)
    print(f"Input files: {expected_count}")

    if expected_count == 0:
        print("ERROR: No .npy files found in input directory")
        sys.exit(1)

    # Run inference
    print("\n--- Running Inference ---")
    stats = run_inference(model, args.input, args.output, device, args.batch_size)

    print(f"\n--- Inference Statistics ---")
    print(f"Total images: {stats['total_images']}")
    print(f"Total time: {stats['total_time_seconds']:.3f}s")
    print(f"Avg time/image: {stats['avg_time_per_image_ms']:.1f}ms")
    print(f"Throughput: {stats['throughput_images_per_second']:.1f} img/s")

    # Validate outputs
    print(f"\n--- Validating Outputs ---")
    manifest = validate_outputs(args.output, expected_count)

    if manifest["issues"]:
        print("ISSUES FOUND:")
        for issue in manifest["issues"]:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("All outputs PASS integrity checks.")
        print(f"  Count: {manifest['count']}")
        print(f"  Shape: {manifest['shape']}")
        print(f"  Range: [{manifest['min']:.6f}, {manifest['max']:.6f}]")
        print(f"  NaN: {manifest['nan_count']}, Inf: {manifest['inf_count']}")

    # Save manifest
    output_path = Path(args.output)
    with open(output_path / "inference_manifest.json", "w") as f:
        json.dump({**manifest, **stats}, f, indent=2)

    # Model info
    param_count = sum(p.numel() for p in model.parameters())
    ckpt_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    print(f"\n--- Model Info ---")
    print(f"Parameters: {param_count:,}")
    print(f"Checkpoint size: {ckpt_size_mb:.1f} MB")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
