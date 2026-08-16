# PRD.md — Evidence-Preserving Blind 2× Restoration for Degraded Inspection Images

**Status:** Implementation-ready  
**Audience:** Autonomous coding / engineering agent  
**Primary objective:** Build, train, validate, benchmark, and package a fast restoration system for the challenge-provided grayscale GT + NoisyLR dataset.

---

# 0. Executive Product Definition

## Product

Build an AI restoration system that takes a degraded **128×128 grayscale image** containing unknown noise and 2× spatial-resolution loss and produces a **256×256 grayscale reconstruction**.

The system must optimize for:

1. reconstruction fidelity,
2. structural fidelity,
3. robustness to unknown degradation,
4. minimal hallucination,
5. preservation of fine-scale evidence,
6. fast inference,
7. reproducibility.

This is **not** a generic image enhancer.

The central engineering question is:

> **Can AI faithfully reconstruct degraded inspection imagery while preserving defect-relevant structures without inventing structures that were not supported by the observation?**

---

# 1. Problem Definition

The challenge concerns AI-based restoration of degraded grayscale inspection images.

The degraded input may contain:

- signal-dependent noise,
- multiplicative / speckle-like noise,
- additive noise,
- mixed noise,
- high-frequency noise contamination,
- spatial-resolution reduction,
- blur associated with the degradation process.

The provided challenge data contains:

- clean high-resolution GT images,
- degraded low-resolution NoisyLR images,
- no confirmed explicit GT↔NoisyLR correspondence.

The target spatial scale is:

```text
128×128 → 256×256
```

The model therefore solves a combined:

```text
denoising
+
blind super-resolution
+
degradation inversion
```

problem.

---

# 2. Industrial Framing

The system must NOT be presented as replacing semiconductor inspection.

Modern semiconductor inspection already uses sophisticated optical inspection, signal processing, defect detection, review and high-resolution characterization workflows.

Our project addresses a different layer:

```text
Inspection / acquisition
        ↓
degraded image
        ↓
AI restoration
        ↓
downstream analysis / review
```

The proposed value is:

> computationally reconstructing a more useful representation of a degraded observation while preserving the evidence contained in the observation.

Potential downstream applications include:

- improved visualization,
- improved defect localization,
- improved automated defect detection,
- improved classification,
- improved review prioritization.

These downstream benefits must be **experimentally validated** and must not be claimed merely because PSNR/SSIM improves.

---

# 3. Inspection vs Restoration

Treat the following as separate tasks.

| Task | Question |
|---|---|
| Inspection | Is there a defect? |
| Detection | Where is it? |
| Review | What is it? |
| Classification | What type is it? |
| Restoration | Can the degraded observation be reconstructed more faithfully? |

The challenge is primarily a **restoration** problem.

If defect-labeled data becomes available, downstream detection/classification experiments should be implemented as secondary validation.

---

# 4. Core Scientific Formulation

Represent the observed image approximately as:

```text
Y = D(H(X))
```

where:

- `X` = unknown clean high-resolution image,
- `H` = blur / optical / sampling degradation,
- `D` = noise process,
- `Y` = observed degraded low-resolution image.

The model estimates:

```text
X_hat = f_theta(Y)
```

Important:

> The inverse is ill-posed. A 128×128 observation cannot uniquely determine every pixel of a 256×256 image.

Therefore the model must not be treated as a machine that literally recovers all physically lost information.

Instead, it learns a reconstruction prior from the clean training distribution.

The engineering objective is:

> **produce the most faithful reconstruction supported by the observation and learned prior, while minimizing unsupported structure.**

---

# 5. Non-Negotiable Safety/Fidelity Principle

The primary principle is:

# Evidence preservation over visual enhancement.

The model must avoid:

## False-negative restoration

Do not:

- erase real small structures,
- suppress thin edges,
- remove small defect-like features,
- merge separate structures,
- destroy periodic geometry.

## False-positive restoration

Do not:

- invent particles,
- invent scratches,
- invent edges,
- invent periodic structures,
- create ringing that resembles defects,
- hallucinate high-frequency texture.

## Geometry distortion

Do not unnecessarily change:

- feature position,
- feature spacing,
- orientation,
- width,
- periodicity,
- boundary location.

A visually sharper image is **not automatically a better restoration**.

---

# 6. Dataset Facts

The following facts are accepted from the supplied raw `.npy` dataset analysis.

## 6.1 Counts

```text
GT      = 3200
NoisyLR = 400
```

## 6.2 Resolution

```text
GT      = 256×256
NoisyLR = 128×128
scale   = 2×
```

## 6.3 Channels

```text
grayscale / single channel
```

## 6.4 dtype

```text
float32
```

## 6.5 GT range

GT images are per-image normalized to approximately:

```text
[0, 1]
```

with observed:

```text
min = 0
max = 1
```

## 6.6 NoisyLR range

NoisyLR values can extend beyond the GT range.

Observed approximate range:

```text
-0.225 → 2.158
```

Therefore the model input must retain the original values.

---

# 7. Critical Input Rule

## NEVER clip NoisyLR before inference or training.

Forbidden:

```python
x = np.clip(x, 0, 1)
```

before the model.

This would destroy information about the actual degradation.

Only the final prediction may be constrained:

```python
prediction = np.clip(prediction, 0.0, 1.0)
```

---

# 8. Critical Pairing Rule

The 3200 GT images and 400 NoisyLR images must **not** be assumed to be paired by filename.

Do not perform:

```text
GT/000001.npy ↔ NoisyLR/000001.npy
```

unless official dataset documentation explicitly confirms this correspondence.

The supplied analysis indicates that the NoisyLR set does not have explicit corresponding GT targets.

Therefore supervised training must use:

```text
GT → synthetic degradation → synthetic LR
```

---

# 9. Important Dataset-Domain Risk

The provided dataset analysis shows imagery that appears broader than obvious semiconductor-specific imagery, including generic textures and natural/architectural scenes.

Do NOT silently label the entire GT dataset as confirmed semiconductor imagery.

Use the terminology:

> **challenge-provided grayscale restoration dataset**

unless official challenge documentation establishes the exact image domain.

Treat potential domain mismatch as a major OOD/generalization risk.

The model must therefore be evaluated for:

- structural OOD,
- noise OOD,
- brightness OOD,
- frequency OOD,
- degradation OOD.

---

# 10. Observed Noise Characteristics

The supplied analysis indicates that the NoisyLR degradation is not adequately described as simple fixed Gaussian noise.

Observed characteristics include:

- signal-dependent noise,
- multiplicative / speckle-like behavior,
- additive components,
- mixed noise,
- variable per-image noise level,
- heavy-tailed behavior at low signal,
- high-frequency noise floor,
- values outside `[0,1]`.

Approximate empirical variance model:

```text
Var(noise | X)
≈
0.0094 X² + 0.0092 X - 0.0003
```

The analysis reported a very strong empirical fit.

A practical approximate parameterization is:

```text
sigma(x) ≈ 0.11 * sqrt(max(x,0)) + 0.008
```

with image-dependent scaling.

These are **empirical dataset statistics**, not a claim about the physical sensor mechanism.

Do not claim that the noise is definitively photon shot noise without physical acquisition documentation.

---

# 11. Noise Distribution

The supplied analysis approximately characterized:

| Category | Approx. fraction |
|---|---:|
| Dominant multiplicative | 34% |
| Dominant additive | 22% |
| Mixed | 16% |
| Low noise | 2% |
| Other / unclassified | 26% |

These values are initial priors for degradation generation.

They must remain configurable.

---

# 12. Heavy-Tailed Noise

At low signal levels, the observed noise is strongly non-Gaussian.

The supplied analysis reported approximately:

```text
skewness ≈ 1.31
excess kurtosis ≈ 7.37
```

Therefore the degradation generator must support heavy-tailed noise.

Supported candidates:

```text
Gaussian
Laplace
Student-t
outlier mixture
```

Do not assume that one distribution is universally correct.

---

# 13. Noise-Level Variation

Estimated NoisyLR noise standard deviation spans approximately:

```text
0.002 → 0.124
```

with high variation between images.

Therefore:

```text
fixed sigma
```

is forbidden as the only training strategy.

The training generator must sample a distribution of noise levels.

---

# 14. Signal-Dependent Noise

The analysis shows a strong relationship between image brightness and estimated noise.

The degradation generator should therefore implement:

```text
sigma(x) = scale * base_sigma(x)
```

where:

```text
base_sigma(x) =
    0.11 * sqrt(max(x,0))
    + 0.008
```

and:

```text
scale
```

is sampled from a configurable distribution.

Initial configuration:

```yaml
noise:
  signal_dependent:
    enabled: true
    scale_min: 0.3
    scale_max: 2.0
```

These values must be calibrated against real NoisyLR statistics.

---

# 15. Multiplicative Noise

Implement:

```text
noise = x * alpha * epsilon
```

where:

```text
epsilon ~ N(0,1)
```

and `alpha` is configurable.

---

# 16. Additive Noise

Implement:

```text
noise = sigma * epsilon
```

where:

```text
epsilon ~ N(0,1)
```

with configurable sigma.

---

# 17. Mixed Noise

Implement:

```text
noise =
    additive_noise
    +
    multiplicative_noise
```

Allow independent sampling of the two components.

---

# 18. Heavy-Tailed Noise

Implement optional:

```text
epsilon ~ Student-t
```

and/or:

```text
epsilon ~ Laplace
```

plus configurable outlier probability.

This should be used mainly for robustness augmentation and calibrated experiments.

---

# 19. Recommended Degradation Pipeline

Default:

```text
GT 256×256
       ↓
blur / optical approximation
       ↓
2× downsampling
       ↓
signal-dependent noise
       ↓
additive/multiplicative mixture
       ↓
optional heavy-tail component
       ↓
synthetic 128×128 LR
```

The degradation engine must be independent of the model.

API:

```python
degraded = degradation_pipeline(
    gt,
    rng,
    config
)
```

---

# 20. Blur Model

Initial model:

```text
Gaussian blur
sigma ≈ 1.0
```

Recommended augmentation:

```text
sigma ∈ [0.7, 1.3]
```

Configuration:

```yaml
degradation:
  blur:
    enabled: true
    sigma_min: 0.7
    sigma_max: 1.3
```

Do not claim this is the exact physical optical PSF.

It is an empirical approximation for training.

---

# 21. Downsampling

Required scale:

```text
256×256 → 128×128
```

Preferred initial method:

```text
Gaussian prefilter + 2× subsampling
```

Alternative kernels may be tested later.

The exact implementation must be deterministic and documented.

---

# 22. Real-vs-Synthetic Calibration

This is a first-class component.

Compare:

```text
REAL NoisyLR
vs
SYNTHETIC LR
```

using:

### Pixel statistics

- min
- max
- mean
- median
- std
- percentiles
- fraction `<0`
- fraction `>1`

### Per-image statistics

- mean
- std
- min
- max
- estimated noise level

### Frequency statistics

- radial power spectrum
- low-frequency energy
- mid-frequency energy
- high-frequency energy
- high-frequency noise floor

### Structural statistics

- edge density
- gradient magnitude
- local variance
- periodicity / FFT peaks

Generate:

```text
reports/degradation_calibration/
├── pixel_distribution.png
├── mean_distribution.png
├── noise_distribution.png
├── frequency_comparison.png
├── edge_density_comparison.png
└── calibration_report.json
```

The synthetic degradation configuration should be iteratively calibrated against these reports.

---

# 23. Dataset Split

Create the split once and persist it.

The GT dataset contains approximately:

```text
119 near-duplicate pairs
208 images involved
100 near-duplicate groups
```

Therefore a naive random split is prohibited.

Required procedure:

1. Build near-duplicate groups.
2. Keep every group in one split.
3. Stratify by structural/image statistics where possible.
4. Target approximately:
   - 85% train
   - 15% validation.
5. Ensure rare/high-edge/high-noise-related image types are represented in validation.

Save:

```text
splits/
├── train_groups.json
├── val_groups.json
└── split_metadata.json
```

---

# 24. Validation Modes

Implement four validation modes.

## A. Standard synthetic validation

```text
GT
 ↓
synthetic degradation
 ↓
model
 ↓
reconstruction
 ↓
compare against GT
```

## B. Hard degradation validation

Emphasize:

- high noise,
- low signal,
- high signal,
- extreme out-of-range inputs,
- heavy-tailed noise,
- mixed noise.

## C. Structural OOD validation

Hold out unusual structural/image clusters.

## D. Noise OOD validation

Train on one range/distribution and evaluate on a shifted noise distribution.

The final report must separate these modes.

---

# 25. Primary Model

Implement a lightweight **NAFNet-style restoration backbone with 2× super-resolution reconstruction**.

Reason:

- strong image-restoration baseline,
- efficient architecture,
- suitable for denoising/restoration,
- better starting point than an unnecessarily large Transformer,
- aligns with inference-time constraints.

The architecture must remain configurable.

---

# 26. Proposed Architecture

```text
NoisyLR 128×128
       │
       ├──────────────┐
       │              │
       ▼              ▼
Noise Estimator   Feature Encoder
       │              │
       │              ▼
       │        NAFNet-style blocks
       │              │
       └───────┬──────┘
               ▼
       Noise-conditioned
          restoration
               │
               ▼
        2× upsampling
               │
               ▼
       HR reconstruction
               │
               ▼
     conservative refinement
               │
               ▼
          prediction
               │
               ▼
          clamp [0,1]
               │
               ▼
          256×256
```

---

# 27. Noise Conditioning

Support three modes:

```yaml
model:
  noise_conditioning:
    mode: spatial
```

Options:

```text
none
scalar
spatial
```

### Scalar

Estimate one noise level per image.

### Spatial

Estimate a spatial noise map.

Preferred final candidate:

```text
spatial
```

but it must earn its place through validation.

---

# 28. Architecture Selection Policy

Do not start with a huge Transformer.

Required progression:

```text
Bicubic
   ↓
Small U-Net
   ↓
NAFNet
   ↓
Noise-aware NAFNet
   ↓
Optional SwinIR / Restormer
```

Only move to a larger architecture if it produces meaningful gains in:

- hard/OOD validation,
- structural metrics,
- or inspection utility,

without violating inference constraints.

---

# 29. Why GAN-First Is Prohibited

Do not use GAN/perceptual generation as the first solution.

GAN-style models optimize visual realism.

Our objective is evidence fidelity.

A realistic but fabricated structure can be more harmful than a slightly blurry but truthful structure.

GAN/perceptual models may be evaluated later as an experiment only.

---

# 30. Loss Function

Primary:

```text
L_total =
    λ1 * L_charbonnier
  + λ2 * L_ssim
  + λ3 * L_gradient
  + λ4 * L_frequency
```

Initial weights:

```yaml
loss:
  charbonnier: 1.0
  ssim: 0.15
  gradient: 0.05
  frequency: 0.02
```

These are starting values, not fixed truths.

---

# 31. Charbonnier Loss

Use as the primary pixel-level reconstruction loss.

Reason:

- robust to outliers,
- less sensitive than pure L2,
- suitable for restoration.

---

# 32. SSIM Loss

Use to encourage structural similarity.

Do not allow SSIM to replace pixel fidelity.

---

# 33. Gradient Loss

Compare:

```text
dx
dy
```

between GT and reconstruction.

Purpose:

- preserve boundaries,
- preserve thin structures,
- reduce oversmoothing.

---

# 34. Frequency Loss

The dataset analysis shows a strong GT frequency decay and a noisy high-frequency floor in NoisyLR.

Therefore do not uniformly maximize high-frequency energy.

Use frequency weighting:

```text
low frequency  → strong
mid frequency  → strong
high frequency → conservative
```

Compare radial spectra:

```text
FFT(GT)
vs
FFT(prediction)
```

The objective is not:

> maximize sharpness.

It is:

> reconstruct the appropriate frequency distribution.

---

# 35. Structural Metrics

Implement:

## Edge Preservation Score

Compare gradient magnitude / edge maps.

Possible formulation:

```text
EPS = 1 - normalized_gradient_error
```

Document exact implementation.

## Frequency Reconstruction Error

Compare radial power spectra.

## Periodicity Preservation

Compare:

- FFT peak locations,
- dominant periods,
- periodic energy.

These metrics exist to catch models that improve PSNR while damaging structural geometry.

---

# 36. Defect-Related Metrics

If labeled defect data is available, implement:

```text
precision
recall
F1
IoU / Dice
false-positive rate
false-negative rate
```

Use a frozen downstream detector/classifier.

Do not jointly train the detector and restoration model for the primary evaluation.

This ensures that restoration utility can be isolated.

---

# 37. Feature Preservation Score

Implement a controlled synthetic structural-feature experiment.

Procedure:

```text
clean GT
   ↓
identify small structural feature
   ↓
degrade
   ↓
restore
   ↓
compare
```

Measure:

- feature contrast,
- feature width,
- feature position,
- edge strength,
- local intensity profile.

Generate:

```text
feature_preservation_score
```

This is especially important for thin or high-frequency structures.

---

# 38. Hallucination / False-Structure Test

Compare:

```text
prediction
vs
GT
```

and detect unsupported structures.

Potential metrics:

- false edge rate,
- false high-frequency energy,
- false connected components,
- unexpected periodic peaks,
- ringing score.

This is a required research experiment.

---

# 39. Defect-Erasure Test

Where small known structures exist in GT:

```text
GT feature
 ↓
degrade
 ↓
restore
```

Measure whether the feature remains.

Models that improve average PSNR but erase small structures should be rejected.

---

# 40. Primary Metrics

Required:

```text
PSNR
SSIM
MAE
RMSE
```

Supplementary:

```text
LPIPS
edge preservation
frequency error
periodicity preservation
feature preservation
```

LPIPS must not be the primary selection criterion.

---

# 41. Model Selection

Internal ranking:

```text
40% PSNR
20% SSIM
15% MAE/RMSE
15% structural metrics
10% frequency consistency
```

But this is not an absolute law.

A model with slightly lower PSNR must be preferred if it demonstrates substantially better:

- OOD robustness,
- feature preservation,
- hallucination control,
- inference efficiency.

---

# 42. Baselines

Implement:

## Baseline 1 — Bicubic

```text
128×128 → bicubic → 256×256
```

## Baseline 2 — Denoise + Bicubic

Classical denoising followed by upsampling.

## Baseline 3 — Small U-Net

## Baseline 4 — NAFNet

## Baseline 5 — Noise-aware NAFNet

## Optional

- SwinIR
- Restormer
- perceptual/GAN model

---

# 43. Required Ablation Matrix

Minimum:

| Experiment | Noise conditioning | Structural loss | Calibrated degradation | Heavy tails |
|---|---|---|---|---|
| A | No | No | No | No |
| B | Yes | No | No | No |
| C | Yes | Yes | No | No |
| D | Yes | Yes | Yes | No |
| E | Yes | Yes | Yes | Yes |

Record:

```text
PSNR
SSIM
MAE
RMSE
edge score
frequency error
feature preservation
hallucination score
parameter count
training time
inference time
```

---

# 44. Training

Initial configuration:

```yaml
optimizer: AdamW
learning_rate: 0.0002
weight_decay: 0.0001
scheduler: cosine
warmup_epochs: 5
epochs: 200
mixed_precision: true
```

All values must be configurable.

---

# 45. Training Patches

Recommended initial training:

```text
GT patch = 128×128
synthetic LR = 64×64
```

Also support full:

```text
256×256 GT
→
128×128 LR
```

for validation.

The model architecture must support both patch and full-image inference.

---

# 46. Augmentation

Allowed:

- horizontal flip,
- vertical flip,
- 90° rotation,
- random crop.

Do not use:

- arbitrary elastic deformation,
- strong geometric distortion,
- transformations that destroy meaningful periodic structure.

Degradation randomization is more important than generic augmentation.

---

# 47. Degradation Randomization

Every training sample should be capable of receiving a different degradation realization.

Randomize:

```text
blur sigma
noise family
noise scale
additive component
multiplicative component
heavy-tail probability
```

This prevents the model from memorizing one degradation process.

---

# 48. Curriculum Training

## Stage 1

Stable:

```text
blur
+
2× downsampling
+
signal-dependent Gaussian noise
```

Goal:

> learn basic reconstruction.

## Stage 2

Introduce:

```text
multiplicative noise
additive noise
mixed noise
heavy tails
variable noise levels
```

Goal:

> learn degradation robustness.

## Stage 3 — Optional

Calibrated real-NoisyLR degradation distribution.

Goal:

> minimize synthetic-to-real degradation gap.

---

# 49. Real NoisyLR Usage

The 400 NoisyLR images may be used as an **unlabeled target-domain distribution**.

Allowed:

```text
real NoisyLR
 ↓
statistics
 ↓
degradation calibration
 ↓
synthetic training
```

Not allowed:

```text
real NoisyLR
 ↓
invent pseudo-GT
 ↓
claim supervised training
```

If any unsupervised adaptation is tested, it must be clearly separated from supervised validation.

---

# 50. OOD Evaluation

Build explicit OOD subsets.

## Noise OOD

Train on one noise range; validate on another.

## Brightness OOD

Hold out unusual brightness distributions.

## Structure OOD

Hold out unusual structural clusters.

## Frequency OOD

Hold out unusual dominant periodicity / frequency characteristics.

## Combined OOD

Combine multiple shifts.

Report:

```text
ID performance
OOD performance
relative degradation
```

A robust model should degrade gracefully.

---

# 51. Test-Time Augmentation

Implement optional:

```text
identity
horizontal flip
vertical flip
horizontal + vertical
90°
180°
270°
```

Invert transforms and average predictions.

Configuration:

```yaml
inference:
  tta:
    enabled: false
```

Only enable for final inference if validation demonstrates a useful improvement relative to runtime cost.

---

# 52. Ensemble Support

Optional:

```text
model A
model B
model C
   ↓
weighted average
```

Only use if validation demonstrates a meaningful improvement.

Inference-time limits must be respected.

---

# 53. Inference Requirements

Input:

```text
/path/to/NoisyLR/*.npy
```

For each image:

```text
load float32
      ↓
preserve raw range
      ↓
model
      ↓
256×256 prediction
      ↓
clip prediction to [0,1]
      ↓
save float32 .npy
```

---

# 54. Final Output Contract

Each output must be:

```text
shape: (256,256)
dtype: float32
range: [0,1]
```

Directory:

```text
outputs/restored/
```

with exactly one output per input.

Filename must match the input filename.

---

# 55. Output Integrity

Before final submission verify:

```text
400 inputs
400 outputs
all names match
all outputs are 256×256
all dtype float32
no NaN
no Inf
min >= 0
max <= 1
```

Generate:

```text
outputs/inference_manifest.json
```

Example:

```json
{
  "count": 400,
  "shape": [256, 256],
  "dtype": "float32",
  "min": 0.0,
  "max": 1.0,
  "nan_count": 0,
  "inf_count": 0
}
```

---

# 56. Inference Benchmarking

The challenge explicitly evaluates inference time.

The evaluation script must report:

```text
total inference time
average time/image
throughput
model parameter count
checkpoint size
```

Benchmark on:

```text
CPU
available CUDA GPU
```

where practical.

Do not optimize solely for speed at the expense of restoration fidelity.

---

# 57. H100 Compatibility

The official benchmark may run on an H100.

Therefore:

- avoid hardware-specific assumptions,
- support CUDA,
- support mixed precision,
- avoid unnecessary CPU↔GPU transfers,
- load model once,
- reuse tensors where possible,
- process batches where safe,
- make batch size configurable.

The evaluation script must not require source-code edits.

---

# 58. Required Evaluation Script

Implement:

```text
evaluate.py
```

or equivalent standalone script.

CLI:

```bash
python evaluate.py \
    --input /path/to/test_images \
    --output /path/to/restored_outputs \
    --checkpoint /path/to/model.pth
```

It must:

1. load the model,
2. load all `.npy` files,
3. preserve input values,
4. run inference,
5. save outputs,
6. validate outputs,
7. report runtime,
8. exit with non-zero status on failure.

No manual source modification is permitted.

---

# 59. Evaluation Script Constraints

The evaluation script must:

- be standalone,
- not depend on notebooks,
- not require training,
- not require internet,
- not require manual path editing,
- load the checkpoint automatically from CLI,
- work from a clean environment,
- support CUDA if available,
- produce deterministic output when configured.

This is one of the most important deliverables.

---

# 60. Training Reproducibility

Every experiment saves:

```text
experiments/<experiment_name>/
├── config.yaml
├── seed.txt
├── model_summary.txt
├── train_log.csv
├── metrics.json
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── validation/
│   ├── metrics.json
│   └── visual_grid.png
└── degradation/
    └── calibration.json
```

Store:

- random seed,
- Git commit hash if available,
- configuration,
- dataset split hash,
- model configuration,
- degradation configuration.

---

# 61. Checkpointing

Save:

```text
last.pth
best_psnr.pth
best_ssim.pth
best_structural.pth
```

The final selected checkpoint must be explicitly recorded.

---

# 62. Early Stopping

Optional:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 30
```

Never use training loss alone for model selection.

---

# 63. Visualization Reports

Generate validation grids:

```text
GT
↓
Synthetic LR
↓
Bicubic
↓
Model
↓
Absolute Error
```

For real NoisyLR:

```text
NoisyLR
↓
Restored
```

For hard cases, automatically display:

- highest noise,
- lowest noise,
- brightest,
- darkest,
- highest edge density,
- strongest periodicity,
- most out-of-range values.

---

# 64. Frequency Analysis

For each important experiment generate:

```text
GT radial spectrum
Synthetic LR spectrum
Restored spectrum
Real NoisyLR spectrum
```

Measure:

```text
low-frequency error
mid-frequency error
high-frequency error
noise-floor behavior
```

Do not reward artificial high-frequency amplification.

---

# 65. Visual Comparison Rules

Visual comparison must not use aggressive contrast manipulation that makes one result appear better than another.

Use identical:

- normalization,
- display range,
- crop,
- interpolation,
- magnification.

For diagnostic crops, show:

```text
GT
Input
Bicubic
Model
```

side by side.

---

# 66. Project Structure

Recommended:

```text
project/
│
├── README.md
├── PRD.md
├── requirements.txt
├── pyproject.toml
├── evaluate.py
├── train.py
├── run_pipeline.py
│
├── configs/
│   ├── baseline_bicubic.yaml
│   ├── baseline_unet.yaml
│   ├── nafnet.yaml
│   ├── noise_aware.yaml
│   └── final.yaml
│
├── data/
│   ├── dataset.py
│   ├── split.py
│   ├── statistics.py
│   └── validation.py
│
├── degradation/
│   ├── pipeline.py
│   ├── blur.py
│   ├── downsample.py
│   ├── gaussian_noise.py
│   ├── multiplicative_noise.py
│   ├── mixed_noise.py
│   ├── heavy_tail.py
│   └── calibrate.py
│
├── models/
│   ├── unet.py
│   ├── nafnet.py
│   ├── noise_estimator.py
│   ├── restoration.py
│   └── losses.py
│
├── training/
│   ├── train.py
│   ├── validate.py
│   ├── checkpoint.py
│   └── metrics.py
│
├── inference/
│   ├── infer.py
│   ├── tta.py
│   └── ensemble.py
│
├── analysis/
│   ├── dataset_report.py
│   ├── frequency.py
│   ├── structure.py
│   ├── feature_preservation.py
│   └── visualization.py
│
├── experiments/
├── reports/
├── outputs/
├── splits/
└── tests/
    ├── test_dataset.py
    ├── test_degradation.py
    ├── test_model.py
    ├── test_inference.py
    ├── test_output_integrity.py
    └── test_split.py
```

The agent may improve the structure, but responsibilities must remain separated.

---

# 67. CLI Requirements

Implement:

```bash
python -m analysis.dataset_report
```

```bash
python -m data.split
```

```bash
python -m degradation.calibrate
```

```bash
python train.py --config configs/noise_aware.yaml
```

```bash
python -m training.validate \
    --checkpoint experiments/.../best.pth
```

```bash
python evaluate.py \
    --input dataset/NoisyLR \
    --checkpoint experiments/.../best.pth \
    --output outputs/restored
```

```bash
python run_pipeline.py --config configs/final.yaml
```

---

# 68. Configuration-Driven Design

Never hard-code:

- dataset paths,
- output paths,
- seed,
- batch size,
- learning rate,
- noise ranges,
- blur ranges,
- loss weights,
- architecture depth,
- checkpoint path,
- TTA,
- ensemble,
- device.

All must be configurable.

---

# 69. Hardware

Support:

```text
CUDA
CPU fallback
```

When CUDA is available:

- use AMP,
- use pinned memory,
- use configurable workers,
- use non-blocking transfers,
- use efficient batch processing.

Do not load the entire dataset into GPU memory.

---

# 70. Data Loading

Use lazy loading for `.npy` files where practical.

Do not:

```text
load all 3200 images → GPU
```

Training should load only the required samples/patches.

---

# 71. Testing

## Dataset

Verify:

```text
GT shape = 256×256
LR shape = 128×128
GT dtype = float32
LR dtype = float32
GT range ≈ [0,1]
LR may exceed [0,1]
```

## Degradation

With fixed seed:

```text
same input + same seed = same output
```

Verify:

- shape,
- dtype,
- parameter behavior,
- noise statistics.

## Model

Verify:

```text
input  = [B,1,128,128]
output = [B,1,256,256]
```

## Inference

Verify:

- output count,
- shapes,
- dtype,
- range,
- NaN/Inf.

---

# 72. Experiment Tracking

Each experiment records:

```text
experiment name
Git commit
seed
split
model
parameter count
degradation configuration
loss configuration
optimizer
learning rate
epochs
best epoch
PSNR
SSIM
MAE
RMSE
edge score
frequency error
feature preservation
hallucination score
training time
inference time
checkpoint size
```

---

# 73. Required Research Experiments

## Experiment A — Blind vs noise-aware

```text
blind NAFNet
vs
noise-aware NAFNet
```

## Experiment B — Scalar vs spatial noise

```text
scalar conditioning
vs
spatial conditioning
```

## Experiment C — Degradation model

```text
Gaussian only
vs
signal-dependent
vs
mixed
vs
heavy-tail
vs
calibrated
```

## Experiment D — Loss

```text
Charbonnier
Charbonnier + SSIM
Charbonnier + gradient
Charbonnier + gradient + frequency
```

## Experiment E — OOD

Measure degradation when:

- noise increases,
- brightness shifts,
- structure shifts,
- frequency shifts.

## Experiment F — Inspection utility

If labels are available:

```text
degraded
vs
restored
```

using the same frozen detector.

---

# 74. Optional Self-Supervised Refinement

Only investigate after the supervised synthetic system is stable.

Potential approaches:

- blind-spot consistency,
- noise consistency,
- Noise2Noise-like training where statistically valid,
- carefully controlled unsupervised adaptation.

Do not blindly fine-tune against pseudo-GT.

Keep the original model and adapted model separately benchmarked.

---

# 75. Classical Baselines

At minimum compare against:

```text
bicubic
```

and optionally:

```text
Gaussian denoise + bicubic
BM3D + bicubic
```

if computationally practical.

The purpose is to prove that the neural model adds value beyond interpolation/denoising.

---

# 76. Final Model Selection

The final model must be selected using:

```text
ID validation
+
hard validation
+
OOD validation
+
structural fidelity
+
runtime
```

Do not choose solely by:

```text
best PSNR
```

Do not choose solely by:

```text
best visual appearance
```

Do not choose solely by:

```text
largest model
```

---

# 77. Failure Conditions

The final model must be rejected or investigated if it shows:

### Oversmoothing

Small structures disappear.

### Oversharpening

Edges become exaggerated.

### Ringing

False halos appear around strong edges.

### Hallucination

Unsupported structures appear.

### Periodicity distortion

Spacing/orientation changes.

### Noise-to-texture conversion

Noise becomes artificial detail.

### Input clipping sensitivity

Model performance changes dramatically when values outside `[0,1]` are removed.

### OOD collapse

Performance falls dramatically on shifted distributions.

---

# 78. Important Scientific Limitations

The system must explicitly acknowledge:

1. A single degraded LR image cannot uniquely recover all lost HR information.
2. Synthetic degradation may differ from real acquisition physics.
3. The provided dataset may not fully represent semiconductor imagery.
4. PSNR/SSIM do not prove inspection usefulness.
5. Restoration may improve appearance while harming defect detection.
6. Physical accuracy cannot be guaranteed from image reconstruction metrics alone.
7. A learned prior can introduce unsupported structures.

These limitations should appear in the final report and presentation.

---

# 79. Recommended Research Hypotheses

## H1 — Reconstruction

A learned restoration model improves reconstruction fidelity over bicubic upsampling.

## H2 — Degradation awareness

Empirically calibrated degradation training improves performance on real NoisyLR relative to generic Gaussian degradation.

## H3 — Noise awareness

Explicit noise conditioning improves robustness across varying noise levels.

## H4 — Structural loss

Gradient/frequency-aware losses improve fine-structure preservation without unacceptable hallucination.

## H5 — Inspection utility

If defect labels are available, restoration can improve downstream defect detection without increasing false positives.

---

# 80. Final Pipeline

The complete system should be:

```text
                 3200 GT Images
                       │
                       ▼
              Duplicate-aware split
                       │
                       ▼
             Clean-image distribution
                       │
                       ▼
              Empirical calibration
                       │
                       ▼
             Synthetic degradation
                       │
        ┌──────────────┼──────────────┐
        │              │              │
       blur       downsampling      noise
                                      │
                              ┌───────┼────────┐
                              │       │        │
                          additive  multi   heavy-tail
                              │       │        │
                              └───────┼────────┘
                                      ▼
                             Synthetic NoisyLR
                                      │
                                      ▼
                           Noise-aware NAFNet
                                      │
                                      ▼
                                2× reconstruction
                                      │
                                      ▼
                           Structure-aware output
                                      │
                                      ▼
                               [0,1] clipping
                                      │
                                      ▼
                                256×256 output
                                      │
                   ┌──────────────────┴─────────────────┐
                   ▼                                    ▼
           Reconstruction metrics                Structural metrics
                   │                                    │
                   └──────────────────┬─────────────────┘
                                      ▼
                               Model selection
                                      │
                                      ▼
                              400 Real NoisyLR
                                      │
                                      ▼
                                Final inference
                                      │
                                      ▼
                             400 restored .npy
```

---

# 81. Final Hackathon Deliverables

## PPT/PDF

The final presentation should communicate:

### Slide 1
Team details.

### Slide 2
Problem and industrial context.

### Slide 3
Why conventional enhancement is insufficient.

### Slide 4
Proposed degradation-aware restoration architecture.

### Slide 5
Evidence-preserving innovation.

### Slide 6
Validation results.

### Slide 7
Technology, runtime and feasibility.

### Slide 8
GitHub + demo.

### Slide 9
References.

Do not claim test PSNR/SSIM unless test GT is actually supplied.

---

# 82. GitHub Requirements

Repository must contain:

```text
README.md
evaluate.py
train.py
requirements.txt
configs/
data/
degradation/
models/
training/
inference/
analysis/
tests/
```

Also provide:

```text
trained checkpoint
restored outputs
validation reports
visual comparisons
```

Large weights may use Git LFS or an explicitly documented external artifact host if permitted by the submission rules.

---

# 83. README Requirements

A reviewer must be able to:

```text
clone repository
        ↓
install dependencies
        ↓
download/load checkpoint
        ↓
run evaluation script
        ↓
receive restored outputs
```

without editing source code.

README must include:

- environment requirements,
- installation,
- dataset structure,
- training command,
- validation command,
- inference command,
- checkpoint location,
- expected output format,
- troubleshooting.

---

# 84. Final Report Requirements

Generate:

```text
reports/final_report.md
```

Sections:

1. Problem definition.
2. Dataset.
3. Dataset statistics.
4. Noise characterization.
5. Synthetic degradation.
6. Model architecture.
7. Loss functions.
8. Validation methodology.
9. Baselines.
10. Ablations.
11. OOD results.
12. Structural preservation.
13. Hallucination/erasure analysis.
14. Runtime.
15. Final model.
16. Limitations.
17. Future work.

---

# 85. Definition of Done

The project is complete only when:

- [ ] Dataset loader works.
- [ ] Dataset integrity checks work.
- [ ] Duplicate-aware split works.
- [ ] Synthetic degradation engine works.
- [ ] Real-vs-synthetic calibration works.
- [ ] Bicubic baseline works.
- [ ] U-Net baseline works.
- [ ] NAFNet baseline works.
- [ ] Noise-aware NAFNet works.
- [ ] Charbonnier loss works.
- [ ] SSIM loss works.
- [ ] Gradient loss works.
- [ ] Frequency loss works.
- [ ] PSNR works.
- [ ] SSIM works.
- [ ] MAE/RMSE works.
- [ ] Edge metric works.
- [ ] Frequency metric works.
- [ ] Feature preservation test works.
- [ ] Hallucination test works.
- [ ] OOD validation works.
- [ ] Checkpointing works.
- [ ] Reproducibility works.
- [ ] Runtime benchmark works.
- [ ] Standalone evaluation script works.
- [ ] 400 final outputs generated.
- [ ] All outputs pass integrity checks.
- [ ] Final report generated.
- [ ] README allows reproduction without manual edits.

---

# 86. Required Implementation Order

The coding agent MUST implement incrementally.

## Phase 1 — Dataset

```text
dataset loader
↓
integrity tests
↓
statistics verification
```

## Phase 2 — Split

```text
duplicate detection
↓
grouping
↓
train/validation split
```

## Phase 3 — Degradation

```text
blur
↓
downsample
↓
signal-dependent noise
↓
mixed noise
↓
heavy-tail
↓
calibration
```

## Phase 4 — Baselines

```text
bicubic
↓
denoise+bicubic
↓
U-Net
```

## Phase 5 — Main model

```text
NAFNet
↓
noise-aware NAFNet
```

## Phase 6 — Loss

```text
pixel
↓
SSIM
↓
gradient
↓
frequency
```

## Phase 7 — Robustness

```text
hard validation
↓
OOD validation
↓
feature preservation
↓
hallucination testing
```

## Phase 8 — Optimization

```text
AMP
↓
batch optimization
↓
TTA benchmark
↓
optional ensemble
```

## Phase 9 — Final inference

```text
400 NoisyLR
↓
400 restored outputs
↓
integrity checks
```

## Phase 10 — Submission

```text
final checkpoint
+
evaluation.py
+
README
+
reports
+
restored outputs
```

---

# 87. Engineering Rules

The coding agent MUST follow these rules.

### Rule 1
Never assume filename matching means pairing.

### Rule 2
Never clip NoisyLR before model inference.

### Rule 3
Only clip final predictions to `[0,1]`.

### Rule 4
Never use a naive random split.

### Rule 5
Never claim supervised test metrics without GT.

### Rule 6
Never train exclusively on fixed Gaussian noise.

### Rule 7
Never optimize solely for visual sharpness.

### Rule 8
Never introduce GAN/perceptual hallucination as the default model.

### Rule 9
Never overwrite raw `.npy` data.

### Rule 10
Every experiment must be reproducible.

### Rule 11
Every final output must pass integrity checks.

### Rule 12
Every major architectural addition must be justified by an experiment.

### Rule 13
Do not claim physical noise mechanisms that the dataset alone cannot establish.

### Rule 14
Do not claim improved defect detection unless it is experimentally measured.

### Rule 15
Prefer a simpler model with stronger robust validation over a larger model with marginal average-metric gains.

---

# 88. Final Product Philosophy

The system is not an:

```text
AI Image Enhancer
```

It is:

# An Evidence-Preserving Blind Image Restoration System

The distinction is fundamental.

The model should not ask:

> “What would a sharper image look like?”

It should ask:

> “Given this degraded observation and what I learned from the clean-image distribution, what reconstruction is most strongly supported by the available evidence?”

The final system therefore prioritizes:

```text
Faithfulness
    >
Structural preservation
    >
OOD robustness
    >
Reconstruction quality
    >
Perceptual attractiveness
```

subject to:

```text
acceptable inference time
```

---

# 89. External Research References

The engineering team should consult the following research during implementation:

1. **NAFNet — Simple Baselines for Image Restoration**  
   https://arxiv.org/abs/2204.04676

2. **Designing a Practical Degradation Model for Deep Blind Image Super-Resolution — BSRGAN/BSRNet**  
   https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Designing_a_Practical_Degradation_Model_for_Deep_Blind_Image_Super-Resolution_ICCV_2021_paper.html

3. **Real-ESRGAN — Training Real-World Blind Super-Resolution with Pure Synthetic Data**  
   https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html

4. **SwinIR — Image Restoration Using Swin Transformer**  
   https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html

5. **The Perception-Distortion Tradeoff**  
   https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html

6. **KLA inspection/review portfolio**  
   https://ir.kla.com/news-events/press-releases/detail/43/kla-announces-new-defect-inspection-and-review-portfolio

7. **Recent semiconductor-specific super-resolution / inspection research**  
   https://arxiv.org/abs/2607.17401

These references provide background for architecture, degradation modeling, blind SR, perception-vs-distortion tradeoffs, and the distinction between image reconstruction and inspection utility.

---

# 90. FINAL INSTRUCTION TO THE CODING AGENT

Build this system as an engineering/research pipeline, not as a one-shot neural-network script.

Do not begin by implementing the largest possible model.

First prove:

```text
dataset correctness
        ↓
degradation correctness
        ↓
baseline correctness
        ↓
model correctness
        ↓
validation correctness
        ↓
OOD robustness
        ↓
structural preservation
        ↓
runtime
        ↓
final inference
```

Every component must be testable independently.

Every major design choice must be measurable.

The final model must not be selected because its outputs “look better.”

The final model must be selected because the evidence demonstrates that it provides a better balance of:

```text
reconstruction fidelity
+
structural fidelity
+
OOD robustness
+
low hallucination
+
low defect-erasure risk
+
acceptable inference time
```

# END OF PRD
