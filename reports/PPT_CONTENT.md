# PPT Content — Slide by Slide

**File name**: TeamName_KLA_PS01.pdf

---

## SLIDE 1: Team Details

**Team Name**: [Your Team Name]

| Name | Role | Contact |
|------|------|---------|
| [Member 1] | Model Architecture & Training | [email] |
| [Member 2] | Degradation Pipeline & Data | [email] |
| [Member 3] | Evaluation & Deployment | [email] |

**College**: [Your College Name]

---

## SLIDE 2: Problem Statement Addressed

**Problem**: AI-Based Restoration of Degraded Images (KLA PS01)

**Why It Matters**:

In semiconductor manufacturing, inspection images captured by optical systems suffer from:
- **Speckle noise** — coherent imaging artifacts that corrupt pixel intensities
- **Gaussian noise** — sensor-level thermal/electronic noise
- **Resolution loss** — optical diffraction limits, fast acquisition reducing pixel count

These degradations reduce defect detection accuracy. A missed 5nm defect on a chip die can cause entire wafer lots to fail.

**Our task**: Take a degraded 128×128 inspection image and produce a clean, sharp 256×256 restoration — removing all three noise types while faithfully preserving real structures.

---

## SLIDE 3: Idea Description

**Core Concept**: Noise-Aware Blind Restoration Network

We chose a **NAFNet** (Nonlinear Activation Free Network) architecture because:
- Designed specifically for image restoration (not classification)
- Efficient: no heavy attention like transformers
- State-of-the-art on denoising/deblurring benchmarks
- Fast inference suitable for production

**How we address all 3 degradation types**:

| Degradation | How We Handle It |
|-------------|-----------------|
| Speckle noise | Noise estimator produces a spatial noise map — model learns speckle structure is signal-dependent |
| Gaussian noise | Trained on variable noise levels (σ from 0.005 to 0.08) — generalizes across intensities |
| 2× Super-resolution | PixelShuffle upsampling layer reconstructs 256×256 from 128×128 features |

**Key insight**: A single network jointly handles all three — no separate denoising then upscaling stages.

---

## SLIDE 4: Proposed Solution

**Architecture**: Noise-Aware NAFNet (25M parameters)

```
128×128 input (raw, unclipped)
       │
       ├────────────────┐
       │                │
       ▼                ▼
 Noise Estimator   Feature Encoder
 (spatial σ map)    (Conv 3×3)
       │                │
       └───────┬────────┘
               ▼
       NAFNet Encoder-Decoder
       [2,4,8,8] blocks + skip connections
               │
               ▼
       2× PixelShuffle Upsampling
               │
               ▼
       Clamp [0,1] → 256×256 output
```

**Training Strategy**:
- 3200 clean GT images → synthetic degradation → 128×128 noisy LR
- On-the-fly random degradation (different noise per sample per epoch)
- AdamW optimizer, cosine LR schedule, 150 epochs
- Duplicate-aware train/val split (no data leakage)

**Loss Function**: Charbonnier loss (robust L1 variant)
- Directly optimizes pixel-level accuracy
- Less sensitive to outliers than MSE
- No GAN/perceptual loss = no hallucinated structures

**Data Augmentation**: Random flip, rotation, crop + degradation randomization (blur σ, noise type, noise level all randomized per sample)

---

## SLIDE 5: Innovation & Uniqueness

**1. Spatial Noise Conditioning**
- Most models are "blind" — ours SEES the noise
- A mini-network estimates per-pixel noise level
- Main network uses this map to adaptively filter: aggressive on noisy regions, gentle on clean ones

**2. Calibrated Degradation Pipeline**
- We analyzed real test image statistics (mean, std, noise floor, frequency spectrum)
- Tuned our synthetic noise to match real data distribution
- Includes: signal-dependent noise, multiplicative speckle, additive Gaussian, heavy-tailed outliers

**3. Evidence Preservation Design**
- No GAN loss → no fabricated structures
- No perceptual loss → no texture hallucination
- Philosophy: "reconstruct what the data supports, don't invent what it doesn't"

**4. Multi-Model Ensemble + TTA**
- 3 models (different widths/losses) averaged
- 8 geometric transforms per image → self-ensembling
- Free +0.3-0.5 dB improvement at inference time

---

## SLIDE 6: Results

**Quantitative Results (Validation Set)**:

| Method | PSNR ↑ | SSIM ↑ | Inference |
|--------|--------|--------|-----------|
| Bicubic (no AI) | 22.12 dB | 0.520 | — |
| **Our Model (single)** | **26.76 dB** | **0.708** | **16 ms** |
| Our Model (ensemble+TTA) | ~27.1 dB | ~0.72 | 1.3 s |

**Improvement**: +4.64 dB PSNR = 2.9× reduction in error

**Visual Results**:

[INSERT: `reports/final_figures/01_visual_comparison.png`]
- Shows: Ground Truth → Degraded → Bicubic → Our Model
- Clear noise removal + detail recovery

[INSERT: `reports/final_figures/04_real_test_restoration.png`]
- Shows: Real test inputs → Our restored outputs

---

## SLIDE 7: Technology & Feasibility

| Component | Details |
|-----------|---------|
| Framework | PyTorch 2.6 |
| Language | Python 3.11 |
| Training Hardware | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| Training Time | ~5 hours (150 epochs) |
| Model Size | 25M parameters / 286 MB checkpoint |
| Inference Time | **16.4 ms/image** (CUDA, batch=8) |
| Throughput | **61 images/second** |
| H100 Compatible | Yes — supports AMP, batch processing, CUDA |

**Deployment-ready**:
- Single Python script, no notebooks
- Auto-detects GPU/CPU
- Batch processing for throughput
- Mixed precision for speed
- No internet/manual edits required at inference

---

## SLIDE 8: GitHub & Video Link

**GitHub**: [INSERT YOUR PUBLIC REPO LINK]

Repository contains:
- ✅ `evaluate.py` — standalone evaluation script
- ✅ `train.py` — full training reproduction
- ✅ Trained model weights (286 MB)
- ✅ 400 restored test outputs
- ✅ `requirements.txt`
- ✅ Complete README with setup instructions

**Run command** (for KLA benchmarking):
```bash
python evaluate.py --input /path/to/test --output /path/to/output
```

**Video**: [Optional — INSERT LINK if recorded]

---

## SLIDE 9: References

1. Chen, L. et al. "Simple Baselines for Image Restoration" (NAFNet), arXiv:2204.04676, 2022.

2. Zhang, K. et al. "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution" (BSRGAN), ICCV 2021.

3. Wang, X. et al. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCV Workshop 2021.

4. Blau, Y. & Michaeli, T. "The Perception-Distortion Tradeoff", CVPR 2018.

5. KLA Corporation. "Defect Inspection and Review Portfolio", 2024. https://ir.kla.com

---

## NOTES FOR CREATING THE PPT

**Figures to include** (all in `reports/final_figures/`):
- Slide 4: Use `06_architecture.png` as pipeline diagram
- Slide 6: Use `01_visual_comparison.png` and `02_metrics_comparison.png`
- Slide 6: Use `04_real_test_restoration.png` for real test examples

**Design tips**:
- Use dark/professional theme (KLA is a semiconductor company)
- Keep text minimal, let figures speak
- Highlight the numbers: +4.64 dB, 16ms, 25M params
- Save as PDF: TeamName_KLA_PS01.pdf
