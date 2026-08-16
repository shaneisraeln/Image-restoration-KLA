# Semiconductor Image Restoration Dataset — Forensic Analysis Report

## PART 1 — DATASET STRUCTURE

### Directory Layout

```
E:\Hackathons\Semiconductor Image Resolution\
├── train/          (3200 .npy files: 000000.npy → 003199.npy)
├── NoisyLR/        (400 .npy files: 000000.npy → 000399.npy)
└── analysis_output/ (generated analysis plots)
```

### Key Findings

| Property | Value |
|----------|-------|
| Total files | 3600 |
| Train folder | 3200 files (clean GT images) |
| NoisyLR folder | 400 files (degraded test images) |
| File format | NumPy `.npy` (uncompressed float32 arrays) |
| Naming convention | Zero-padded 6-digit sequential index |
| Metadata files | **None** |
| Separate val/test folders | **No** |
| Source/domain labels | **None** |

### CRITICAL: Pairing Structure

**There are NO explicit paired training images.** The dataset provides:
- `train/`: 3200 **clean ground-truth only** images at 256×256
- `NoisyLR/`: 400 **degraded test** images at 128×128 (no corresponding GT available)

The filename overlap (indices 000000–000399 exist in both folders) does NOT represent pairing — these are unrelated images confirmed by cross-correlation analysis (max correlation between corresponding indices < 0.25 after downsampling).

**Implication: The model must be trained using SYNTHETIC degradation applied to the GT images.**

---

## PART 2 — IMAGE DIMENSIONS

### Complete Resolution Table

| Image Type | Resolution | Count | Channels | dtype | Aspect Ratio |
|-----------|-----------|-------|----------|-------|-------------|
| GT (train) | 256×256 | 3200 | 1 (grayscale) | float32 | 1:1 |
| Degraded (NoisyLR) | 128×128 | 400 | 1 (grayscale) | float32 | 1:1 |

### Dimensional Analysis (ALL files scanned)
- **ALL** 3200 train files are exactly 256×256, float32, 2D arrays
- **ALL** 400 NoisyLR files are exactly 128×128, float32, 2D arrays
- Scale relationship: **exactly 2×** (128→256)
- No unexpected resolutions found
- No aspect ratio anomalies
- All images are single-channel grayscale (no color channels)

---

## PART 3 — PIXEL VALUE / DYNAMIC RANGE ANALYSIS

### A. Ground-Truth Images (train/)

| Statistic | Value |
|-----------|-------|
| Global min | 0.00000000 |
| Global max | 1.00000000 |
| Global mean | 0.4335 |
| Global median | 0.4113 |
| Global std | 0.2726 |
| 1st percentile | 0.0100 |
| 5th percentile | 0.0422 |
| 25th percentile | 0.1965 |
| 75th percentile | 0.6489 |
| 95th percentile | 0.9058 |
| 99th percentile | 0.9751 |
| Pixels < 0 | **0.0000%** |
| Pixels > 1 | **0.0000%** |
| Pixels in [0,1] | **100.0000%** |

**CRITICAL: ALL 3200 GT images have exact min=0.0 and exact max=1.0.** This proves per-image min-max normalization was applied.

### B. Degraded Images (NoisyLR/)

| Statistic | Value |
|-----------|-------|
| Global min | -0.2249 |
| Global max | 2.1580 |
| Global mean | 0.4427 |
| Global median | 0.4134 |
| Global std | 0.2843 |
| 1st percentile | 0.0029 |
| 5th percentile | 0.0378 |
| 25th percentile | 0.2084 |
| 75th percentile | 0.6511 |
| 95th percentile | 0.9403 |
| 99th percentile | 1.1192 |
| Pixels < 0 | **0.6601%** |
| Pixels > 1 | **3.0801%** |
| Pixels in [0,1] | **96.2597%** |

### Per-Image Statistics

| Metric | GT (mean ± std) | NoisyLR (mean ± std) |
|--------|-----------------|---------------------|
| Image mean | 0.4335 ± 0.1862 | 0.4427 ± 0.1659 |
| Image std | 0.1876 ± 0.0667 | 0.2203 ± 0.0692 |
| Image min | 0.0000 (all) | -0.0222 ± 0.0419 |
| Image max | 1.0000 (all) | 1.4334 ± 0.1873 |

### Out-of-Range Details (NoisyLR)
- **282/400 images (70.5%)** have at least one pixel < 0
- **395/400 images (98.8%)** have at least one pixel > 1
- Values below 0: mean=-0.011, min=-0.225 (mostly slight negative, rarely extreme)
- Values above 1: mean=1.102, max=2.158, 99th percentile=1.42
- 4 images have >20% of pixels exceeding 1.0

---

## PART 4 — DATA TYPE AND IMAGE ENCODING

| Property | GT | NoisyLR |
|----------|----|---------| 
| Data type | float32 | float32 |
| Storage format | .npy (NumPy binary, uncompressed) |  .npy |
| Unique values per image | ~65,500 (near-maximum for 256×256) | ~16,377 (near-maximum for 128×128) |
| Minimum value gap | ~1e-8 (float32 machine epsilon) | ~3e-8 |
| Quantization | **None** — full float32 precision | **None** |
| Prior normalization | Per-image min-max to [0,1] | **Not normalized** — contains values outside [0,1] |
| Compression | None | None |

**The data is stored at full float32 precision with no quantization artifacts. GT images were clearly per-image normalized to [0,1] before saving.**

---

## PART 5 — PAIRING VERIFICATION

### Key Finding: NO Explicit Pairs Exist

Exhaustive verification confirms:
1. NoisyLR images are **NOT derived from** any train images (max cross-correlation after downsampling = 0.71, which is far too low for a noisy version of the same content)
2. The 400 NoisyLR images appear to be from a **separate image source** not represented in the training set
3. The training set contains only clean GT images — **no degraded counterparts**

### What This Means
The problem setup is:
- **Train**: Learn the appearance of clean semiconductor images
- **Test (NoisyLR)**: Restore these degraded images (no GT available for scoring locally)
- **Training approach**: Must synthesize degradation from GT to create training pairs

---

## PART 6 — DEGRADATION PROCESS ANALYSIS

### A. NOISE MODEL

#### Signal-Dependent Noise (CONFIRMED)

The noise variance follows a **mixed multiplicative-Poisson model**:

```
Var(noise | signal=X) ≈ 0.0094 * X² + 0.0092 * X - 0.0003
```

Best single-parameter characterization: **sigma ≈ 0.11 * sqrt(X) + 0.008**

| Signal Level | Measured Noise Std | 
|-------------|-------------------|
| 0.0–0.05 | 0.013 |
| 0.1–0.15 | 0.042 |
| 0.2–0.25 | 0.055 |
| 0.3–0.35 | 0.067 |
| 0.4–0.45 | 0.082 |
| 0.5–0.55 | 0.091 |
| 0.6–0.65 | 0.097 |
| 0.7–0.75 | 0.100 |
| 0.8–0.85 | 0.103 |
| 0.9–0.95 | 0.104 |

The R² for the mixed model (quadratic+linear in signal) is **0.992** — excellent fit.

#### Noise Distribution Shape

| Signal Level | Skewness | Excess Kurtosis | Interpretation |
|-------------|----------|-----------------|---------------|
| Low (< 0.2) | 1.31 | 7.37 | **Heavy-tailed, strongly non-Gaussian** |
| Mid (0.3–0.5) | 0.16 | 0.05 | **Approximately Gaussian** |
| High (> 0.6) | 0.21 | 0.18 | **Nearly Gaussian** |

**At low signal levels, the noise has extremely heavy tails (kurtosis > 7).** This makes the restoration problem particularly challenging in dark regions.

#### Per-Image Noise Level Variation

| Statistic | Value |
|-----------|-------|
| Mean noise std | 0.0422 |
| Noise std CV | **0.615** (highly variable!) |
| Min noise std | 0.0021 |
| Max noise std | 0.1237 |
| Correlation with brightness | **0.77** |

**CRITICAL: Noise level varies 60× across images (0.002 to 0.124).**

#### Noise Type Categories (400 NoisyLR images)

| Type | Count | Percentage |
|------|-------|-----------|
| Dominant multiplicative (signal-dependent) | 136 | 34% |
| Dominant additive (signal-independent) | 88 | 22% |
| Mixed multiplicative + additive | 64 | 16% |
| Low noise | 8 | 2% |
| Other/unclassified | 104 | 26% |

### B. DOWNSAMPLING METHOD

**Best estimate: Gaussian pre-filter (σ ≈ 1.0 at HR) + pixel subsampling**

Evidence:
- Frequency spectrum matching gives optimal σ = 1.0–1.05 (minimum error at σ=1.05)
- This is standard for 2× anti-aliased downsampling
- The spectral shape of NoisyLR matches Gaussian-filtered-then-subsampled GT

### C. DEGRADATION ORDER

**Most likely: GT → Gaussian blur → Subsample → Add noise (at LR resolution)**

Evidence from noise autocorrelation analysis:
- Measured noise lag-1 autocorrelation: **0.05** (per-image median)
- Simulated Model A (downsample then noise): AC = **0.03** ← matches
- Simulated Model B (noise then downsample): AC = **0.26** ← does not match

The low spatial correlation of the noise strongly supports noise being added AFTER downsampling.

### D. COMPLETE DEGRADATION MODEL

```
degraded = downsample(gaussian_blur(GT, sigma≈1.0)) + noise(signal)
where:
  noise(X) ~ Normal(0, sigma²)
  sigma ≈ scale_factor * (0.11 * sqrt(X) + 0.008)
  scale_factor varies per-image (Uniform(0.3, 2.0) approximately)
```

Note: At low signal levels, the noise distribution has heavier tails than Gaussian (kurtosis ~7).

---

## PART 7 — FREQUENCY-DOMAIN ANALYSIS

### Power Spectrum Characteristics

| Metric | GT (256×256) | NoisyLR (128×128) |
|--------|-------------|-------------------|
| Low freq energy (r<10) | 1.34e7 | 2.09e6 |
| Mid freq energy (r=20-50) | 1,945 | 1,540 |
| High freq energy (r>80%Nyquist) | 179 | 257 |

### Key Observations
1. **Noise dominates high frequencies in NoisyLR**: At 75% of Nyquist, noise accounts for **90%** of power
2. **GT has natural power-law decay**: Power decreases smoothly with frequency (typical of natural/structured images)
3. **NoisyLR shows flat noise floor**: Starting from ~50% Nyquist, the spectrum flattens (white noise dominance)
4. **GT power ratio (high/mid)**: 0.53 — moderate high-frequency content, natural for real images

### Implications for Restoration
- High-frequency detail recovery is extremely challenging (signal-to-noise < 0.1 at high frequencies)
- Low and mid frequencies are recoverable (signal dominates)
- The model must effectively hallucinate high-frequency detail from lower-frequency structure

---

## PART 8 — IMAGE CONTENT / SEMICONDUCTOR STRUCTURE ANALYSIS

### Content Categorization (320 sampled GT images)

| Category | Count | Percentage |
|----------|-------|-----------|
| Texture/irregular patterns | 225 | 70.3% |
| Complex multi-scale patterns | 54 | 16.9% |
| Line/grid structures (orthogonal) | 25 | 7.8% |
| Near-uniform regions | 16 | 5.0% |
| Images with strong FFT periodicity | 319 | **99.7%** |

### Structural Properties
- **Almost all images show periodic/repetitive structure** (characteristic of semiconductor lithography)
- Dominant fine-scale periods: 32–64 pixels (at 256×256)
- Feature sizes (connected components): median=3.6 pixels, 95th percentile=128 pixels
- Edge density: mean=0.16, std=0.17, range [0.0, 0.97]
- Wide diversity in mean intensity: [0.016, 0.959]

### Difficulty Factors
1. Fine features (2–4 pixel width) will be lost in 2× downsampling
2. Near-uniform regions make noise very visible
3. Periodic structures create strong spectral peaks that could alias
4. High-contrast edges can produce ringing artifacts in reconstruction

---

## PART 9 — DATASET DIVERSITY

### K-Means Clustering (k=4) of GT Images

| Cluster | N | Mean Intensity | Std | Gradient | Edge Density | Description |
|---------|---|---------------|-----|----------|-------------|-------------|
| 0 | 686 | 0.459 | 0.192 | 0.108 | 0.405 | High-edge, medium intensity |
| 1 | 1108 | 0.260 | 0.149 | 0.042 | 0.096 | Dark, smooth |
| 2 | 1380 | 0.560 | 0.215 | 0.038 | 0.076 | Bright, smooth |
| 3 | 26 | 0.450 | 0.260 | 0.458 | 0.952 | Very high edge (extreme) |

**Silhouette scores**: k=2: 0.347, k=3: 0.288, k=4: 0.302, k=5: 0.318

The best separation is k=2 (smooth vs. edge-heavy), but the dataset is relatively **continuous** (moderate silhouette scores). Cluster 3 (26 images with extreme edge density >0.95) is a distinct outlier group.

### NoisyLR vs GT Distribution Comparison
- Mean intensity distributions are well-matched (GT: 0.43±0.19, NoisyLR: 0.44±0.17)
- NoisyLR has slightly higher average std (0.22 vs 0.19) due to noise contribution
- NoisyLR edge density is higher (0.46 vs 0.16) — noise creates apparent edges

---

## PART 10 — DUPLICATES AND DATA LEAKAGE

### Exact Duplicates
- **0 exact duplicates found** (verified via MD5 hash of all 3200 files)

### Near-Duplicates (MSE < 0.0001)
- **119 near-duplicate pairs** found
- Organized into **100 groups**: 92 pairs + 8 triplets
- **208 images** involved in near-duplicate groups
- **2992 effectively unique images**

### Nature of Near-Duplicates
- Always adjacent or near-adjacent in index (distance 1–2)
- Differences are sub-pixel (max_abs_diff < 0.009)
- NOT flips/rotations — pixel differences are uniformly tiny
- Appear to be slightly different processing/acquisitions of the same scene

### Data Leakage Risk
- **No leakage between train and NoisyLR** (confirmed by correlation analysis)
- **Within-train near-duplicates** pose a risk if split naively: duplicate images in train and validation would inflate validation metrics
- 208/3200 (6.5%) of images are near-duplicates of another image in the set

---

## PART 11 — TRAIN/VALIDATION SPLIT RECOMMENDATION

### Recommended Strategy: **Cluster-Aware Split with Duplicate Grouping**

1. **Group near-duplicates** into 100 groups (plus 2992 singletons)
2. **Assign entire groups** to either train or validation (never split a group)
3. **Stratify by cluster** (4 clusters) to ensure representation
4. **Recommended ratio**: 85% train / 15% validation (≈2720 train, 480 val)
5. **Ensure Cluster 3** (26 extreme edge images) has representation in both splits

### Why Not Random Split
A random split would place near-duplicate pairs across train/val, giving misleadingly high validation PSNR. The cluster-aware split better approximates OOD performance.

### Additional Consideration
Since the test set may contain "out-of-distribution" samples, the validation set should intentionally include some images that are statistically different from the majority (e.g., extreme mean intensity, extreme edge density).

---

## PART 12 — DATASET QUALITY CHECK

### Anomaly Report

| Check | Result |
|-------|--------|
| NaN values | **0** (both train and NoisyLR) |
| Infinity values | **0** (both) |
| Zero-variance images | **0** |
| Very low variance (<0.0005) | **5 images** (train indices: 361, 2224, 2225, 2226, 2227) |
| Very dark (mean < 0.02) | **1 image** (train index 2224, mean=0.0156) |
| Very bright (mean > 0.95) | **8 images** (train indices: 643, 977, 978, 979, 1193, +3 more) |
| Wrong dimensions | **0** |
| Corrupted/unreadable | **0** |
| Exact duplicates | **0** |
| Near-duplicates (MSE<0.0001) | **119 pairs** |

### Potentially Problematic Images
- **Train 2224**: Extremely dark (mean=0.0156, variance=0.0002) — nearly black
- **Train 2224–2227**: All very low variance — possibly from same problematic source
- **Images with >20% pixels above 1** (NoisyLR): Indices need special handling during inference

---

## PART 13 — STATISTICAL SUMMARY

| Property | Value |
|----------|-------|
| Dataset size | 3600 total files |
| Number of GT images | 3200 (train only, no paired degraded) |
| Number of test images | 400 (NoisyLR, no GT available) |
| GT resolutions | 256×256 only |
| LR resolutions | 128×128 only |
| Scale factor | 2× |
| Channels | 1 (grayscale) |
| Data types | float32 |
| GT range | [0.0, 1.0] exact (per-image normalized) |
| LR range | [-0.225, 2.158] |
| Average GT mean | 0.4335 |
| Average LR mean | 0.4427 |
| Average GT std | 0.1876 |
| Average LR std | 0.2203 |
| Estimated noise type | **Mixed signal-dependent: Poisson-Gaussian** |
| Noise model | sigma ≈ 0.11*sqrt(signal) + 0.008, variable per-image |
| Noise distribution | Nearly Gaussian at mid/high signal, heavy-tailed at low signal |
| Per-image noise variability | CV = 0.61 (highly variable) |
| Estimated downsampling | Gaussian blur (σ≈1.0) + subsample at 2× |
| Degradation order | GT → blur → downsample → noise (noise after down) |
| Number of detected domains | 4 clusters (smooth-dark, smooth-bright, edge-heavy, extreme-edge) |
| Number of anomalous files | 5 very-low-variance + 1 very-dark + 8 very-bright = 14 |
| Exact duplicates | 0 |
| Near-duplicates | 119 pairs (208 images) |
| Potential leakage | Within-train near-duplicates (6.5% of images) |
| Recommended validation strategy | Cluster-aware, duplicate-grouped, 85/15 split |

---

## PART 14 — VISUAL ANALYSIS

All visualizations saved to `analysis_output/`:

1. **01_sample_images.png** — GT and NoisyLR example images
2. **02_histograms.png** — Pixel value distributions (GT vs NoisyLR)
3. **03_noise_model.png** — Noise variance vs signal level with fitted model
4. **04_noise_distribution.png** — Per-image noise level distribution and correlation with brightness
5. **05_frequency_spectrum.png** — Radial power spectra comparison
6. **06_mean_distributions.png** — Per-image mean intensity distributions
7. **07_cluster_examples.png** — Representative images from each cluster
8. **08_noise_level_examples.png** — Low-noise vs high-noise NoisyLR examples
9. **09_worst_cases.png** — Most degraded/problematic images
10. **10_summary.png** — Comprehensive 6-panel summary
11. **11_noise_types.png** — Per-image noise type scatter plot
12. **12_noise_type_examples.png** — Multiplicative vs additive noise examples

---

## PART 15 — MOST IMPORTANT FINDINGS

### TOP 10 DATASET FINDINGS

1. **No paired training data exists.** The 3200 train images are GT-only; degradation must be synthesized. This is the single most important finding.

2. **Noise is signal-dependent** with a well-characterized model: Var ≈ 0.0094·X² + 0.0092·X (R²=0.992). This is a Poisson-Gaussian mixture.

3. **Noise levels vary dramatically per-image** (CV=0.61, range 60×). The model must handle noise levels from near-zero to σ≈0.12.

4. **Different images have different noise types** — 34% predominantly multiplicative, 22% predominantly additive, 16% mixed, 2% near-clean.

5. **Noise has heavy tails at low signal** (kurtosis=7.4 vs Gaussian kurtosis=0). Simple Gaussian noise assumption will underfit dark regions.

6. **Degradation order is blur→downsample→noise** (noise added at LR resolution), confirmed by autocorrelation analysis.

7. **Downsampling uses Gaussian pre-filter (σ≈1.0) + subsampling**, not area averaging or bicubic.

8. **GT images are per-image min-max normalized** to exactly [0,1]. All images span the full dynamic range.

9. **119 near-duplicate pairs** exist within training (6.5% of data), requiring careful validation splitting.

10. **98.8% of NoisyLR images exceed pixel value 1.0** — output must NOT be clipped during processing; only clip at the final output stage.

### WHAT THESE FINDINGS MEAN FOR MODEL DEVELOPMENT

1. **Synthetic degradation pipeline is mandatory**: Must implement: GT → Gaussian_blur(σ~1.0) → subsample(2×) → signal_dependent_noise(variable_strength). The noise model should be: `noise = scale * (0.11*sqrt(max(signal,0)) + 0.008) * randn()` with scale randomly sampled per-image.

2. **Input should NOT be clipped**: Since NoisyLR values extend to [-0.22, 2.16], the model's input layer must accept arbitrary float values. Clipping would destroy information. Only clip the OUTPUT to [0,1].

3. **Noise-level conditioning may be beneficial**: Given 60× variation in noise strength, a blind denoiser may underperform. Consider noise-level-aware architectures or degradation-strength estimation.

4. **Heavy-tailed noise augmentation**: Training with pure Gaussian noise will not match the true distribution at low signal levels. Consider mixing in Laplace or Student-t noise for dark regions, or using a noise model with occasional outliers.

5. **Variable degradation augmentation is critical**: During training, randomly vary the noise level (scale factor 0.3–2.0×), blur sigma (0.7–1.3), and mix ratio between multiplicative/additive noise.

6. **Patch training is appropriate**: 256×256 GT images can be cropped to 128×128 or 64×64 patches during training. The features at 2–64 pixel scale are most relevant.

7. **Frequency-aware loss may help**: High frequencies are dominated by noise in the input. An L1 or perceptual loss that emphasizes mid-frequency reconstruction (where signal is recoverable) could outperform pure pixel MSE.

8. **OOD validation must use cluster-based splitting**: Random splits would leak near-duplicates. Group images by similarity clusters and keep groups intact.

9. **Dataset is moderately sized (3200 images)**: Sufficient for fine-tuning a pretrained model but may be insufficient for training large architectures from scratch. Pretrained image restoration backbones (e.g., SwinIR, RRDB, NAFNet) are likely beneficial.

10. **The model must generalize to unseen noise types**: Since the test set contains OOD samples and different noise types coexist in NoisyLR, the model should be trained with diverse degradation augmentations beyond the estimated parameters.

---

## PART 16 — MACHINE-READABLE REPORT

```json
{
  "num_gt_images": 3200,
  "num_test_images": 400,
  "num_training_pairs": 0,
  "gt_resolutions": ["256x256"],
  "lr_resolutions": ["128x128"],
  "scale_factors": [2],
  "channels": 1,
  "gt_dtype": "float32",
  "lr_dtype": "float32",
  "gt_min": 0.0,
  "gt_max": 1.0,
  "gt_mean": 0.4335,
  "gt_std": 0.2726,
  "lr_min": -0.2249,
  "lr_max": 2.158,
  "lr_mean": 0.4427,
  "lr_std": 0.2843,
  "lr_pixels_below_0_percent": 0.66,
  "lr_pixels_above_1_percent": 3.08,
  "gt_per_image_normalized": true,
  "estimated_noise_type": "mixed_signal_dependent_poisson_gaussian",
  "noise_model": "sigma = scale * (0.11 * sqrt(signal) + 0.008)",
  "noise_model_variance_fit": "Var = 0.0094*X^2 + 0.0092*X - 0.0003 (R^2=0.992)",
  "noise_level_range": [0.002, 0.124],
  "noise_level_cv": 0.615,
  "noise_distribution_low_signal": "heavy_tailed (kurtosis=7.4, skewness=1.3)",
  "noise_distribution_high_signal": "approximately_gaussian (kurtosis=0.06)",
  "noise_type_breakdown": {
    "dominant_multiplicative": 136,
    "dominant_additive": 88,
    "mixed": 64,
    "low_noise": 8,
    "other": 104
  },
  "estimated_downsampling_method": "gaussian_blur_sigma_1.0_plus_subsample",
  "estimated_degradation_order": "GT -> blur -> downsample -> noise",
  "detected_domains": 4,
  "domain_sizes": [686, 1108, 1380, 26],
  "near_duplicate_pairs": 119,
  "near_duplicate_images": 208,
  "effectively_unique_images": 2992,
  "exact_duplicate_count": 0,
  "anomaly_count": 14,
  "nan_or_inf_count": 0,
  "leakage_risk": "moderate_within_train_near_duplicates",
  "recommended_validation_strategy": "cluster_aware_duplicate_grouped_85_15_split",
  "periodic_structure_prevalence": "99.7%",
  "dominant_feature_periods_pixels": "32-64 at 256x256",
  "median_feature_size_pixels": 3.6
}
```
