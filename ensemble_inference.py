"""Ensemble inference: average predictions from multiple models with TTA."""
import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '.')


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
    """8x TTA prediction."""
    transforms = [
        lambda t: t,
        lambda t: torch.flip(t, [3]),
        lambda t: torch.flip(t, [2]),
        lambda t: torch.flip(t, [2, 3]),
        lambda t: torch.rot90(t, 1, [2, 3]),
        lambda t: torch.rot90(t, 2, [2, 3]),
        lambda t: torch.rot90(t, 3, [2, 3]),
        lambda t: torch.flip(torch.rot90(t, 1, [2, 3]), [3]),
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

    predictions = []
    for fwd, inv in zip(transforms, inverse_transforms):
        pred = model(fwd(x))
        predictions.append(inv(pred))
    return torch.stack(predictions).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all models
    models = []
    for ckpt in args.checkpoints:
        print(f"Loading: {ckpt}")
        model = load_model(ckpt, device)
        models.append(model)
    print(f"Ensemble of {len(models)} models")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.npy"))
    print(f"Processing {len(files)} images...")

    total_time = 0
    for i, f in enumerate(files):
        img = np.load(f).astype(np.float32)
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        preds = []
        for model in models:
            if args.tta:
                pred = predict_with_tta(model, x)
            else:
                pred = model(x)
            preds.append(pred)

        # Average across models
        ensemble_pred = torch.stack(preds).mean(dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0

        output = np.clip(ensemble_pred[0, 0].cpu().numpy(), 0.0, 1.0).astype(np.float32)
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
