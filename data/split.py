"""Duplicate-aware train/validation split.

The GT dataset contains near-duplicate pairs. A naive random split
would leak information. This module:
1. Detects near-duplicate groups using structural similarity.
2. Keeps all images in a group together in one split.
3. Stratifies by image statistics.
4. Targets 85% train / 15% validation.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from itertools import combinations
from collections import defaultdict


def compute_image_hash(img: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Compute a perceptual hash for duplicate detection."""
    from scipy.ndimage import zoom
    # Downsample to hash_size x hash_size
    scale = hash_size / img.shape[0]
    small = zoom(img, scale, order=1)
    # Compute difference hash
    diff = small[:, 1:] > small[:, :-1]
    return diff.flatten()


def find_near_duplicates(
    gt_dir: str, threshold: float = 0.90
) -> List[Set[int]]:
    """Find groups of near-duplicate images using perceptual hashing + correlation."""
    gt_path = Path(gt_dir)
    files = sorted(gt_path.glob("*.npy"))
    n = len(files)

    print(f"Computing hashes for {n} images...")
    hashes = []
    means = []
    stds = []

    for f in files:
        img = np.load(f).astype(np.float32)
        h = compute_image_hash(img)
        hashes.append(h)
        means.append(img.mean())
        stds.append(img.std())

    hashes = np.array(hashes, dtype=np.float32)
    means = np.array(means)
    stds = np.array(stds)

    # Find similar pairs using hash similarity
    print("Finding near-duplicate pairs...")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Quick filter: means must be close
            if abs(means[i] - means[j]) > 0.1:
                continue
            if abs(stds[i] - stds[j]) > 0.1:
                continue
            # Hash similarity (Hamming)
            sim = np.mean(hashes[i] == hashes[j])
            if sim >= threshold:
                pairs.append((i, j))

    # Build connected components (groups)
    groups = _build_groups(pairs, n)
    # Filter to groups with >1 member
    dup_groups = [g for g in groups if len(g) > 1]

    print(f"Found {len(dup_groups)} near-duplicate groups "
          f"({sum(len(g) for g in dup_groups)} images involved)")
    return groups


def _build_groups(pairs: List[Tuple[int, int]], n: int) -> List[Set[int]]:
    """Build connected components from pairs using union-find."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in pairs:
        union(i, j)

    groups_dict = defaultdict(set)
    for i in range(n):
        groups_dict[find(i)].add(i)

    return list(groups_dict.values())


def create_split(
    gt_dir: str,
    output_dir: str = "splits",
    val_fraction: float = 0.15,
    seed: int = 42,
    threshold: float = 0.90,
) -> Dict:
    """Create duplicate-aware train/val split."""
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(gt_path.glob("*.npy"))
    n = len(files)

    # Find near-duplicate groups
    groups = find_near_duplicates(gt_dir, threshold=threshold)

    # Compute per-group statistics for stratification
    group_stats = []
    for group in groups:
        group_means = []
        group_stds = []
        group_edges = []
        for idx in group:
            img = np.load(files[idx]).astype(np.float32)
            group_means.append(img.mean())
            group_stds.append(img.std())
            # Edge density approximation
            grad_x = np.abs(np.diff(img, axis=1)).mean()
            grad_y = np.abs(np.diff(img, axis=0)).mean()
            group_edges.append((grad_x + grad_y) / 2)

        group_stats.append({
            "indices": sorted(group),
            "mean": np.mean(group_means),
            "std": np.mean(group_stds),
            "edge_density": np.mean(group_edges),
            "size": len(group),
        })

    # Sort groups by mean brightness for stratified splitting
    rng = np.random.default_rng(seed)
    group_order = list(range(len(groups)))
    rng.shuffle(group_order)

    # Assign groups to val until we reach target fraction
    target_val = int(n * val_fraction)
    val_indices = []
    train_indices = []
    val_groups = []
    train_groups = []

    # Reserve some extreme cases for validation
    sorted_by_edge = sorted(range(len(group_stats)),
                            key=lambda i: group_stats[i]["edge_density"],
                            reverse=True)

    # Take top 5% by edge density into validation
    n_edge_val = max(1, int(len(groups) * 0.05))
    forced_val_groups = set(sorted_by_edge[:n_edge_val])

    for gi in group_order:
        group_idx = sorted(groups[gi])
        if gi in forced_val_groups or len(val_indices) < target_val:
            val_indices.extend(group_idx)
            val_groups.append(group_idx)
        else:
            train_indices.extend(group_idx)
            train_groups.append(group_idx)

    # Rebalance if val is too large
    while len(val_indices) > target_val * 1.3 and len(val_groups) > 5:
        moved = val_groups.pop()
        train_groups.append(moved)
        for idx in moved:
            val_indices.remove(idx)
            train_indices.append(idx)

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)

    # Save split
    split_data = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "total": n,
        "val_fraction_actual": len(val_indices) / n,
        "seed": seed,
        "duplicate_threshold": threshold,
        "num_groups": len(groups),
        "num_duplicate_groups": sum(1 for g in groups if len(g) > 1),
    }

    with open(output_path / "split_metadata.json", "w") as f:
        json.dump(split_data, f, indent=2)

    with open(output_path / "train_groups.json", "w") as f:
        json.dump(train_groups, f, indent=2)

    with open(output_path / "val_groups.json", "w") as f:
        json.dump(val_groups, f, indent=2)

    print(f"Split created: {len(train_indices)} train, {len(val_indices)} val "
          f"({len(val_indices)/n*100:.1f}% val)")

    return split_data


def load_split(split_dir: str = "splits") -> Tuple[List[int], List[int]]:
    """Load existing split indices."""
    split_path = Path(split_dir)
    with open(split_path / "split_metadata.json", "r") as f:
        data = json.load(f)
    return data["train_indices"], data["val_indices"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create train/val split")
    parser.add_argument("--gt_dir", default="train", help="GT image directory")
    parser.add_argument("--output_dir", default="splits", help="Output directory")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_split(args.gt_dir, args.output_dir, args.val_fraction, args.seed)
