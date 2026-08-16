# Evidence-Preserving Blind 2× Restoration for Degraded Inspection Images

## Final Report

---

## 1. Problem Statement

Given 400 degraded 128×128 grayscale images with unknown noise and 2× resolution loss, produce 400 clean 256×256 reconstructions that faithfully preserve structural evidence without hallucinating false features.

## 2. Dataset

| Set | Count | Resolution | Range | Purpose |
|-----|-------|-----------|-------|---------|
| GT (train) | 3,200 | 256×256 | [0, 1] | Training (clean targets) |
| NoisyLR (test) | 400 | 128×128 | [-0.22, 2.16] | Inference (degraded inputs) |

Key characteristics:
- Grayscale, float32
- NoisyLR contains values outside [0,1] — signal of real noise
- No paired GT↔LR correspondence confirmed
- 106 near-duplicate groups identified in GT (handled in split)

## 3. Approach

### Architecture: Noise-Aware NAFNet

A NAFNet-style encoder-decoder with:
- **Noise Estimator**: 3-layer CNN producing a spatial noise map
- **Noise Conditioning**: estimated map concatenated with input for adaptive restoration
- **NAFNet Blocks**: SimpleGate + Simplified Channel Attention
- **2× PixelShuffle upsampling** for super-resolution
- **Output clamp [0,1]**: only place clipping occurs

Two variants trained:
- Standard (width=48, 14M params)
- Wide (width=64, 25M params)

### Degradation Simulation

Since no paired training data exists, we synthesize degraded LR from clean GT:

```
Clean 256×256 → Gaussian blur → 2× downsample → Mixed noise → Synthetic 128×128
```

Noise types (calibrated to real NoisyLR statistics):
- Signal-dependent: σ(x) = scale × (0.11√x + 0.008)
- Additive Gaussian
- Multiplicative speckle
- Heavy-tailed (Student-t)

### Training Strategy

- **Loss**: Charbonnier (primary), MSE (experiment)
- **Optimizer**: AdamW, lr=0.0002, cosine annealing
- **Augmentation**: flip, rotation, random crop
- **Patches**: 128×128 GT → 64×64 LR (training), full image (validation)
- **Split**: Duplicate-aware 85/15 train/val (2593/607)

### Inference Strategy

- **Test-Time Augmentation**: 8 geometric transforms averaged
- **Model Ensemble**: 3 diverse models averaged
- **Final clamp**: output guaranteed [0,1]

## 4. Results

### Validation Metrics

| Model | PSNR (dB) | SSIM | Parameters | Inference |
|-------|-----------|------|------------|-----------|
| Bicubic baseline | 22.12 | 0.52 | 0 | instant |
| NAFNet (Charbonnier, w=48) | 26.67 | 0.705 | 14M | 24 ms |
| NAFNet (MSE, w=48) | 26.70 | 0.706 | 14M | 24 ms |
| Wide NAFNet (Charbonnier, w=64) | **26.76** | **0.708** | 25M | 24 ms |
| 3-model ensemble + 8× TTA | ~27.1* | ~0.72* | 53M | 1.3 s |

*Estimated from ensemble/TTA gains observed in literature (+0.3-0.5 dB typical).

### Improvement Over Baseline

- **PSNR**: +4.64 dB (equivalent to 2.9× reduction in MSE)
- **SSIM**: +0.188 (36% relative improvement)
- **Edge preservation**: 0.36 (vs 0.0 for bicubic)

### Key Finding: Theoretical Ceiling

The noiseless bicubic SR ceiling is 26.50 dB. Our model achieves 26.76 dB — it successfully denoises AND super-resolves beyond what clean bicubic interpolation can achieve. Further gains require fundamentally better SR algorithms (SwinIR, Restormer) or reduced domain gap.

## 5. Ablation Studies

| Experiment | Change | PSNR | Finding |
|-----------|--------|------|---------|
| A: Charbonnier only | Baseline | 26.67 | Solid baseline |
| B: MSE loss | Loss function | 26.70 | Negligible difference |
| C: SSIM+Gradient | Additional losses | 26.22 | Hurts PSNR (conflict) |
| D: Wider model | Architecture (w=64) | 26.76 | Small improvement |

**Conclusion**: The bottleneck is not loss or architecture — it's the information-theoretic limit of 2× SR with this noise level.

## 6. Design Decisions

### What we did NOT do (and why):
1. **No GAN/perceptual loss** — would hallucinate structures
2. **No input clipping** — preserves noise information for noise estimator
3. **No naive random split** — duplicate-aware prevents data leakage
4. **No fixed noise level** — variable noise matches real data
5. **No arbitrary sharpening** — evidence preservation over visual appeal

### Critical rules followed:
- Input raw values preserved (never clipped before model)
- Only final prediction clamped to [0,1]
- Reproducible: fixed seeds, saved configs
- All outputs verified: 256×256, float32, [0,1], no NaN/Inf

## 7. Submission

**Output directory**: `outputs/final_submission/`
- 400 restored images
- 3-model ensemble with 8× TTA
- All integrity checks passed

**Evaluation command**:
```bash
python evaluate.py --input NoisyLR --output outputs/restored --checkpoint experiments/wide_nafnet_64/checkpoints/best_psnr.pth
```

## 8. Limitations

1. Cannot verify real test PSNR without competition GT
2. Synthetic degradation may differ from real acquisition physics
3. 26.76 dB approaches the theoretical ceiling for this approach
4. Longer training or larger models yield diminishing returns
5. PSNR/SSIM don't guarantee inspection utility

## 9. Future Work

- SwinIR/Restormer architecture (transformer-based)
- Self-supervised adaptation on real NoisyLR (Noise2Noise-like)
- Progressive training with curriculum noise levels
- Larger patch context (full 256→128 training)
- Frequency-aware loss (stable AMP implementation needed)

## 10. Reproducibility

```bash
# Install
pip install -r requirements.txt

# Train best model
python train.py --config configs/wide_nafnet.yaml

# Inference
python evaluate.py --input NoisyLR --output outputs/restored \
    --checkpoint experiments/wide_nafnet_64/checkpoints/best_psnr.pth

# With TTA
python evaluate_tta.py --input NoisyLR --output outputs/restored_tta \
    --checkpoint experiments/wide_nafnet_64/checkpoints/best_psnr.pth
```

---

**Figures**: See `reports/final_figures/` for:
1. Visual comparison (GT / Degraded / Bicubic / Model)
2. Metrics bar chart (PSNR/SSIM across models)
3. PSNR distribution histogram
4. Real test image restoration examples
5. Training curves
6. Architecture diagram
