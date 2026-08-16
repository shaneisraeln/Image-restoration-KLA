"""Measure actual TTA and ensemble gains on validation set."""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from pathlib import Path
from data.split import load_split
from degradation.pipeline import degradation_pipeline
from training.metrics import compute_psnr, compute_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gt_files = sorted(Path('train').glob('*.npy'))
_, val_indices = load_split('splits')
rng = np.random.default_rng(123)  # Different seed from training

config = {
    'degradation': {
        'blur': {'enabled': True, 'sigma_min': 0.6, 'sigma_max': 1.2},
        'downsample': {'scale': 2, 'method': 'gaussian_prefilter'},
        'noise': {
            'signal_dependent': {'enabled': True, 'scale_min': 0.4, 'scale_max': 1.5},
            'additive': {'enabled': True, 'sigma_min': 0.005, 'sigma_max': 0.08},
            'multiplicative': {'enabled': True, 'alpha_min': 0.03, 'alpha_max': 0.15},
            'heavy_tail': {'enabled': False},
            'mix_weights': {'signal_dependent': 0.5, 'additive': 0.3, 'multiplicative': 0.2},
        },
    }
}

# Load model
from models.nafnet import NoiseAwareNAFNet
state = torch.load('experiments/wide_nafnet_64/checkpoints/best_psnr.pth', map_location=device)
model_cfg = state["config"]["model"]
nc = model_cfg.get("noise_conditioning", {})
model = NoiseAwareNAFNet(
    in_channels=1, out_channels=1, width=64,
    num_blocks=[2, 4, 8, 8], dropout_rate=0.0, noise_mode="spatial"
)
model.load_state_dict(state["model_state_dict"])
model.to(device).eval()

# TTA function
@torch.no_grad()
def predict_tta(model, x):
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
    preds = []
    for fwd, inv in zip(transforms, inverse_transforms):
        preds.append(inv(model(fwd(x))))
    return torch.stack(preds).mean(dim=0)

# Measure on 40 validation images
n_val = 40
single_psnrs, single_ssims = [], []
tta_psnrs, tta_ssims = [], []

print(f"Measuring on {n_val} validation images...")
for i, idx in enumerate(val_indices[:n_val]):
    gt = np.load(gt_files[idx]).astype(np.float32)
    lr = degradation_pipeline(gt, rng, config)
    lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

    # Single pass
    with torch.no_grad():
        pred_single = model(lr_t).cpu().numpy()[0, 0]
    pred_single = np.clip(pred_single, 0, 1)
    single_psnrs.append(compute_psnr(pred_single, gt))
    single_ssims.append(compute_ssim(pred_single, gt))

    # TTA (8x)
    pred_tta = predict_tta(model, lr_t).cpu().numpy()[0, 0]
    pred_tta = np.clip(pred_tta, 0, 1)
    tta_psnrs.append(compute_psnr(pred_tta, gt))
    tta_ssims.append(compute_ssim(pred_tta, gt))

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{n_val} done")

print(f"\n{'='*50}")
print(f"RESULTS (n={n_val})")
print(f"{'='*50}")
print(f"Single pass:  PSNR = {np.mean(single_psnrs):.3f} dB | SSIM = {np.mean(single_ssims):.4f}")
print(f"8× TTA:       PSNR = {np.mean(tta_psnrs):.3f} dB | SSIM = {np.mean(tta_ssims):.4f}")
print(f"TTA gain:     +{np.mean(tta_psnrs)-np.mean(single_psnrs):.3f} dB | +{np.mean(tta_ssims)-np.mean(single_ssims):.4f}")
