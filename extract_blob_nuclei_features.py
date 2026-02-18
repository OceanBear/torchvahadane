#!/usr/bin/env python3
"""
Extract features from sample images of dark blobs (artifacts) vs dark purple nuclei.

Use two directories of small crop images:
  - Blob samples: black/carbon/artifact regions (achromatic dark).
  - Nuclei samples: dark purple nuclei (chromatic dark).

Outputs per-class statistics (mean, std, percentiles) for:
  - Grayscale (0-255)
  - LAB L, a, b and chroma = sqrt(a^2 + b^2)
  - HSV H, S, V (S and V normalized 0-1 for readability)
  - RGB std per pixel (achromatic: low std; purple: higher std)

Use the resulting JSON to choose thresholds for separating blobs from nuclei
(e.g. chroma or saturation threshold) in the black-artifact filter.

Default sample dirs (WSL paths from J:\\HandE\\...):
  - Artifacts (dark blobs): /mnt/j/HandE/artifacts
  - Nuclei (dark purple):   /mnt/j/HandE/nuclei_dark
Results are saved to a JSON file (default: blob_nuclei_features.json).

Example:

  python extract_blob_nuclei_features.py

  python extract_blob_nuclei_features.py --output results/features.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Default sample directories (WSL format; Windows J:\HandE\...)
DEFAULT_ARTIFACTS_DIR = Path("/mnt/j/HandE/artifacts")      # J:\HandE\artifacts
DEFAULT_NUCLEI_DIR = Path("/mnt/j/HandE/nuclei_dark")      # J:\HandE\nuclei_dark
DEFAULT_OUTPUT_JSON = Path("blob_nuclei_features.json")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def load_images_from_dir(dir_path: Path) -> list[np.ndarray]:
    """Load all images from a directory as RGB uint8 arrays."""
    images = []
    for p in sorted(dir_path.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            try:
                from PIL import Image
                img = np.array(Image.open(p))
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            except Exception:
                continue
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    return images


def extract_pixel_features(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extract per-pixel features from an RGB image.
    Returns dict of 1D arrays (flattened), one per feature.
    """
    h, w = rgb.shape[:2]
    n = h * w

    # Grayscale (0-255)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_flat = gray.ravel().astype(np.float64)

    # LAB: L in [0,255], a,b in [-128,127] typically
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].ravel().astype(np.float64)
    a = lab[:, :, 1].ravel().astype(np.float64) - 128.0  # center at 0
    b = lab[:, :, 2].ravel().astype(np.float64) - 128.0
    chroma = np.sqrt(a * a + b * b)

    # HSV: OpenCV H 0-180, S 0-255, V 0-255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].ravel().astype(np.float64)
    S = hsv[:, :, 1].ravel().astype(np.float64) / 255.0  # 0-1
    V = hsv[:, :, 2].ravel().astype(np.float64) / 255.0  # 0-1

    # Per-pixel RGB std (achromatic => low; purple => higher)
    R = rgb[:, :, 0].ravel().astype(np.float64)
    G = rgb[:, :, 1].ravel().astype(np.float64)
    B = rgb[:, :, 2].ravel().astype(np.float64)
    rgb_std = np.std(np.stack([R, G, B], axis=1), axis=1)

    return {
        "grayscale": gray_flat,
        "L": L,
        "a": a,
        "b": b,
        "chroma": chroma,
        "H": H,
        "S": S,
        "V": V,
        "rgb_std": rgb_std,
    }


def collect_pixels_from_images(images: list[np.ndarray], max_pixels: int | None) -> dict[str, np.ndarray]:
    """Collect feature vectors from all images, optionally subsampling to max_pixels."""
    all_features: dict[str, list[np.ndarray]] = {k: [] for k in ["grayscale", "L", "a", "b", "chroma", "H", "S", "V", "rgb_std"]}
    total = 0
    for img in images:
        feats = extract_pixel_features(img)
        n = feats["grayscale"].size
        if max_pixels is not None and (total + n) > max_pixels:
            # Subsample this image
            take = max_pixels - total
            idx = np.random.choice(n, size=take, replace=False)
            for k in all_features:
                all_features[k].append(feats[k][idx])
            total = max_pixels
            break
        for k in all_features:
            all_features[k].append(feats[k])
        total += n
    out = {}
    for k in all_features:
        out[k] = np.concatenate(all_features[k], axis=0)
    return out


def compute_stats(arr: np.ndarray) -> dict:
    """Compute mean, std, min, max, and percentiles for a 1D array."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "percentiles": {}}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": {p: float(np.percentile(arr, p)) for p in PERCENTILES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from blob vs nuclei sample images for threshold selection."
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory containing sample images of dark blobs/artifacts (default: {DEFAULT_ARTIFACTS_DIR}).",
    )
    parser.add_argument(
        "--nuclei_dir",
        type=Path,
        default=DEFAULT_NUCLEI_DIR,
        help=f"Directory containing sample images of dark purple nuclei (default: {DEFAULT_NUCLEI_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_JSON}).",
    )
    parser.add_argument(
        "--max_pixels_per_class",
        type=int,
        default=100_000,
        help="Max pixels to sample per class to keep runtime and memory reasonable (default: 100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    if not args.artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {args.artifacts_dir}")
    if not args.nuclei_dir.is_dir():
        raise FileNotFoundError(f"Nuclei directory not found: {args.nuclei_dir}")

    blob_images = load_images_from_dir(args.artifacts_dir)
    nuclei_images = load_images_from_dir(args.nuclei_dir)

    if not blob_images:
        raise RuntimeError(f"No images found in {args.artifacts_dir}")
    if not nuclei_images:
        raise RuntimeError(f"No images found in {args.nuclei_dir}")

    print(f"Loaded {len(blob_images)} blob images, {len(nuclei_images)} nuclei images.")

    blob_pixels = collect_pixels_from_images(blob_images, args.max_pixels_per_class)
    nuclei_pixels = collect_pixels_from_images(nuclei_images, args.max_pixels_per_class)

    n_blob = blob_pixels["grayscale"].size
    n_nuclei = nuclei_pixels["grayscale"].size
    print(f"Collected {n_blob} blob pixels, {n_nuclei} nuclei pixels.")

    blob_stats = {k: compute_stats(blob_pixels[k]) for k in blob_pixels}
    nuclei_stats = {k: compute_stats(nuclei_pixels[k]) for k in nuclei_pixels}

    report = {
        "artifacts_dir": str(args.artifacts_dir),
        "nuclei_dir": str(args.nuclei_dir),
        "n_blob_images": len(blob_images),
        "n_nuclei_images": len(nuclei_images),
        "n_blob_pixels": n_blob,
        "n_nuclei_pixels": n_nuclei,
        "blob": blob_stats,
        "nuclei": nuclei_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved stats to {args.output}")

    # Print summary table for key discriminative features
    print("\n--- Summary (use to choose thresholds) ---")
    print("Feature        Blob (mean ± std)          Nuclei (mean ± std)")
    print("-" * 70)
    for key, label in [
        ("grayscale", "Grayscale"),
        ("L", "LAB L"),
        ("chroma", "LAB chroma"),
        ("S", "HSV S (0-1)"),
        ("V", "HSV V (0-1)"),
        ("rgb_std", "RGB std"),
    ]:
        b = blob_stats[key]
        n = nuclei_stats[key]
        bm, bs = b["mean"], b["std"]
        nm, ns = n["mean"], n["std"]
        if bm is None or nm is None:
            continue
        print(f"{label:14} {bm:7.2f} ± {bs or 0:5.2f}    {nm:7.2f} ± {ns or 0:5.2f}")
    print("\nSuggested: use chroma or S (saturation) to separate blobs (low) from nuclei (higher).")


if __name__ == "__main__":
    main()
