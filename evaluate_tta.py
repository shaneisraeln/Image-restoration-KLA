"""Evaluate with TTA (Test-Time Augmentation) for improved results.

Usage:
    python evaluate_tta.py --input NoisyLR --output outputs/restored_tta --checkpoint experiments/noise_aware_nafnet/checkpoints/best_psnr.pth
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    config = state.get("config", {})
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "noise_aware_nafnet")

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
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_with_tta(model, x):
    """TTA: predict with 8 geometric transforms, average results."""
    predictions = []

    # All 8 geometric transforms for grayscale
    transforms = [
        lambda t: t,                                    # identity
        lambda t: torch.flip(t, [3]),                   # hflip
        lambda t: torch.flip(t, [2]),                   # vflip
        lambda t: torch.flip(t, [2, 3]),                # hvflip
        lambda t: torch.rot90(t, 1, [2, 3]),            # rot90
        lambda t: torch.rot90(t, 2, [2, 3]),            # rot180
        lambda t: torch.rot90(t, 3, [2, 3]),            # rot270
        lambda t: torch.flip(torch.rot90(t, 1, [2, 3]), [3]),  # rot90+hflip
    ]

    inverse_transforms = [
        lambda t: t,
        lambda t: torch.flip(t, [3]),
        lambda t: torch.flip(t, [2]),
        lambda t: torch.flip(t, [2, 3]),
        lambda t: torch.rot90(t, 3, [2, 3]),
        lambda t: torch.rot90(t, 2, [2, 3]),
        lambda t: torch.rot90(t, 1, [2, 3]),
        lambda t: torch.flip(torch.rot90(t, 3, [2, 3]), [3]),
    ]

    for fwd, inv in zip(transforms, inverse_transforms):
        x_aug = fwd(x)
        pred = model(x_aug)
        pred_inv = inv(pred)
        predictions.append(pred_inv)

    return torch.stack(predictions).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.npy"))
    print(f"Processing {len(files)} images with 8x TTA...")

    total_time = 0
    for i, f in enumerate(files):
        img = np.load(f).astype(np.float32)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        pred = predict_with_tta(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0

        output = np.clip(pred[0, 0].cpu().numpy(), 0.0, 1.0).astype(np.float32)
        np.save(output_path / f.name, output)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} done ({total_time:.1f}s)")

    print(f"\nDone! Total: {total_time:.1f}s, Avg: {total_time/len(files)*1000:.1f}ms/img")

    # Validate
    from data.validation import validate_outputs
    result = validate_outputs(str(output_path), len(files))
    if result["pass"]:
        print("All outputs PASS integrity checks.")
    else:
        print("ISSUES:", result["issues"])


if __name__ == "__main__":
    main()
