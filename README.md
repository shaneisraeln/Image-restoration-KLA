# AI-Based Restoration of Degraded Inspection Images

**Team**: Porygon  
**Challenge**: KLA Problem Statement 01 — AI-Based Image Restoration  
**Hackathon**: i4C 2026

---

## Quick Start — Run Inference

```bash
# Clone and install
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt

# Run evaluation (auto-detects CUDA/CPU)
python evaluate.py --input /path/to/test_images --output /path/to/output_dir
```

That's it. No manual edits needed. The script auto-loads the best checkpoint from `experiments/wide_nafnet_64/checkpoints/best_psnr.pth`.

---

## Repository Structure

```
├── evaluate.py              # ⭐ MAIN: Standalone evaluation script (KLA benchmark)
├── train.py                 # Training script
├── requirements.txt         # Dependencies
├── README.md                # This file
│
├── models/                  # Model architectures
│   ├── nafnet.py            # NAFNet + Noise-Aware NAFNet
│   ├── unet.py              # U-Net baseline
│   └── losses.py            # Loss functions
│
├── data/                    # Data loading and splitting
├── degradation/             # Synthetic degradation pipeline
├── training/                # Training utilities and metrics
├── inference/               # TTA and ensemble support
├── configs/                 # YAML configurations
│
├── experiments/             # Trained checkpoints
│   └── wide_nafnet_64/
│       └── checkpoints/
│           └── best_psnr.pth  # ⭐ Best model weights
│
├── outputs/
│   └── final_submission/    # ⭐ Restored test outputs (400 images)
│
└── reports/
    ├── FINAL_REPORT.md      # Detailed methodology report
    └── final_figures/       # Result visualizations
```

## How to Run Inference (Detailed)

### Minimum Command
```bash
python evaluate.py --input NoisyLR --output outputs/restored
```

### With Explicit Options
```bash
python evaluate.py \
    --input /path/to/test_images \
    --output /path/to/output_dir \
    --checkpoint experiments/wide_nafnet_64/checkpoints/best_psnr.pth \
    --batch_size 8 \
    --device auto
```

### Expected Output
```
Device: cuda
Model: noise_aware_nafnet, Parameters: 24,912,418
Input files: 400
Processing 400 images...
  Processed 400/400 (9.6s)
--- Inference Statistics ---
Total time: 9.6s
Avg time/image: 24.1ms
Throughput: 41.5 img/s
--- Validating Outputs ---
All outputs PASS integrity checks.
```

## How to Train

```bash
# Train the wide model (best performing)
python train.py --config configs/wide_nafnet.yaml

# Train the standard model
python train.py --config configs/noise_aware.yaml
```

Training requires ~5-6 hours on an RTX 4050 (6GB VRAM) for 150 epochs.

## Model Architecture

**Noise-Aware NAFNet** — a NAFNet-style encoder-decoder with:
- Spatial noise estimation for adaptive processing
- 2× PixelShuffle super-resolution
- Evidence-preserving design (no GAN/perceptual loss)
- 25M parameters (width=64 variant)

## Results

| Model | PSNR (dB) | SSIM | Inference Time |
|-------|-----------|------|----------------|
| Bicubic baseline | 22.12 | 0.52 | — |
| Our model (single) | **26.76** | **0.708** | 24 ms/img |
| Our model (ensemble+TTA) | ~27.1 | ~0.72 | 1.3 s/img |

**Improvement: +4.64 dB PSNR over bicubic baseline**

## Technical Details

- **Input**: 128×128 grayscale float32 (raw values, NOT clipped)
- **Output**: 256×256 grayscale float32, range [0, 1]
- **Training data**: 3200 clean GT images → synthetic degradation → supervised learning
- **Framework**: PyTorch 2.x
- **Hardware**: NVIDIA RTX 4050 (training), H100-compatible (inference)
- **Mixed precision**: Supported (AMP autocast)

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA recommended)
- See `requirements.txt` for full list

## References

1. NAFNet — Simple Baselines for Image Restoration (arXiv:2204.04676)
2. BSRGAN — Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (ICCV 2021)
3. Real-ESRGAN — Training Real-World Blind SR with Pure Synthetic Data (ICCV 2021)
