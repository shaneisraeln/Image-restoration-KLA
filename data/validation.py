"""Output validation utilities."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def validate_outputs(
    output_dir: str,
    expected_count: int = 400,
    expected_shape: tuple = (256, 256),
) -> Dict:
    """Validate all output .npy files meet the submission contract.

    Checks:
    - Correct count
    - Correct shape (256x256)
    - float32 dtype
    - Range [0, 1]
    - No NaN or Inf values
    """
    output_path = Path(output_dir)
    files = sorted(output_path.glob("*.npy"))

    result = {
        "count": len(files),
        "expected_count": expected_count,
        "all_correct_shape": True,
        "all_correct_dtype": True,
        "all_in_range": True,
        "no_nan": True,
        "no_inf": True,
        "global_min": float("inf"),
        "global_max": float("-inf"),
        "issues": [],
        "pass": True,
    }

    for f in files:
        img = np.load(f)

        if img.shape != expected_shape:
            result["all_correct_shape"] = False
            result["issues"].append(f"{f.name}: shape {img.shape}")

        if img.dtype != np.float32:
            result["all_correct_dtype"] = False
            result["issues"].append(f"{f.name}: dtype {img.dtype}")

        if np.isnan(img).any():
            result["no_nan"] = False
            result["issues"].append(f"{f.name}: contains NaN")

        if np.isinf(img).any():
            result["no_inf"] = False
            result["issues"].append(f"{f.name}: contains Inf")

        img_min = float(img.min())
        img_max = float(img.max())

        if img_min < 0 or img_max > 1:
            result["all_in_range"] = False
            result["issues"].append(f"{f.name}: range [{img_min:.6f}, {img_max:.6f}]")

        result["global_min"] = min(result["global_min"], img_min)
        result["global_max"] = max(result["global_max"], img_max)

    if len(files) != expected_count:
        result["issues"].append(f"Expected {expected_count} files, found {len(files)}")

    result["pass"] = (
        len(files) == expected_count
        and result["all_correct_shape"]
        and result["all_correct_dtype"]
        and result["all_in_range"]
        and result["no_nan"]
        and result["no_inf"]
    )

    return result


def validate_input_files(input_dir: str) -> Dict:
    """Validate input NoisyLR files."""
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.npy"))

    result = {
        "count": len(files),
        "shapes": set(),
        "dtypes": set(),
        "has_negatives": False,
        "has_above_one": False,
        "min": float("inf"),
        "max": float("-inf"),
    }

    for f in files:
        img = np.load(f)
        result["shapes"].add(img.shape)
        result["dtypes"].add(str(img.dtype))
        if img.min() < 0:
            result["has_negatives"] = True
        if img.max() > 1:
            result["has_above_one"] = True
        result["min"] = min(result["min"], float(img.min()))
        result["max"] = max(result["max"], float(img.max()))

    result["shapes"] = [list(s) for s in result["shapes"]]
    result["dtypes"] = list(result["dtypes"])
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Output directory to validate")
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()

    result = validate_outputs(args.dir, args.count)
    print(json.dumps(result, indent=2))
    if result["pass"]:
        print("\n✓ All checks PASSED")
    else:
        print("\n✗ FAILED - see issues above")
