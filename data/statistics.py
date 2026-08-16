"""Dataset statistics computation and reporting."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def compute_dataset_statistics(data_dir: str, name: str = "dataset") -> Dict:
    """Compute comprehensive statistics for a set of .npy images."""
    path = Path(data_dir)
    files = sorted(path.glob("*.npy"))

    if not files:
        return {"error": f"No .npy files found in {data_dir}"}

    # Per-image stats
    means = []
    stds = []
    mins = []
    maxs = []
    shapes = set()
    dtypes = set()

    for f in files:
        img = np.load(f)
        shapes.add(img.shape)
        dtypes.add(str(img.dtype))
        means.append(float(img.mean()))
        stds.append(float(img.std()))
        mins.append(float(img.min()))
        maxs.append(float(img.max()))

    return {
        "name": name,
        "count": len(files),
        "shapes": [list(s) for s in shapes],
        "dtypes": list(dtypes),
        "pixel_range": {
            "global_min": min(mins),
            "global_max": max(maxs),
        },
        "per_image": {
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
            "mean_of_stds": float(np.mean(stds)),
            "std_of_stds": float(np.std(stds)),
            "mean_min": float(np.mean(mins)),
            "mean_max": float(np.mean(maxs)),
            "min_of_mins": min(mins),
            "max_of_maxs": max(maxs),
        },
        "percentiles": {
            "means_p5": float(np.percentile(means, 5)),
            "means_p25": float(np.percentile(means, 25)),
            "means_p50": float(np.percentile(means, 50)),
            "means_p75": float(np.percentile(means, 75)),
            "means_p95": float(np.percentile(means, 95)),
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="train")
    parser.add_argument("--lr_dir", default="NoisyLR")
    args = parser.parse_args()

    gt_stats = compute_dataset_statistics(args.gt_dir, "GT")
    lr_stats = compute_dataset_statistics(args.lr_dir, "NoisyLR")

    print("=== GT Statistics ===")
    print(json.dumps(gt_stats, indent=2))
    print("\n=== NoisyLR Statistics ===")
    print(json.dumps(lr_stats, indent=2))
