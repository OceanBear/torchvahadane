#!/usr/bin/env python3
"""
Extract features from sample images of red blood cells (RBCs) vs regular tissue.

Use two directories of small crop images:
  - RBC samples: red blood cell regions.
  - Regular tissue samples: normal tissue regions (excluding RBCs).

Outputs per-class statistics (mean, std, percentiles) for:
  - Grayscale (0-255)
  - LAB L, a, b and chroma = sqrt(a^2 + b^2)
  - HSV H, S, V (S and V normalized 0-1 for readability)
  - RGB std per pixel (achromatic: low std; colored: higher std)

Use the resulting JSON to choose thresholds for separating RBCs from regular tissue
(e.g. hue, chroma, or saturation thresholds) in an RBC removal filter.

Default sample dirs (WSL paths from J:\\HandE\\...):
  - RBC:      /mnt/j/HandE/rbc
  - Regular:  /mnt/j/HandE/regular
Results are saved to a JSON file (default: rbc_regular_features.json).

Example:

  python extract_rbc_regular_features.py

  python extract_rbc_regular_features.py --output results/rbc_features.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvahadane import TorchVahadaneNormalizer
from torchvahadane.utils import convert_RGB_to_OD_cpu

# Default sample directories (WSL format; Windows J:\HandE\...)
DEFAULT_RBC_DIR = Path("/mnt/j/HandE/rbc")           # J:\HandE\rbc
DEFAULT_REGULAR_DIR = Path("/mnt/j/HandE/regular")    # J:\HandE\regular
DEFAULT_OUTPUT_JSON = Path("rbc_regular_features.json")

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


def extract_pixel_features(rgb: np.ndarray, white_threshold: float = 0.9) -> dict[str, np.ndarray]:
    """
    Extract per-pixel features from an RGB image.
    Excludes blank/white areas and pure black pixels using luminosity filter
    (same logic as normalize_tiles_new_2.py tissue mask).
    
    :param rgb: RGB uint8 image (H, W, 3).
    :param white_threshold: LAB L threshold (0-1). Pixels with L >= this are considered blank (default 0.9).
    :return: Dict of 1D arrays (flattened), one per feature, containing only tissue pixels.
    """
    h, w = rgb.shape[:2]
    
    # LAB: L in [0,255], a,b in [-128,127] typically
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L_full = lab[:, :, 0].astype(np.float32) / 255.0  # range [0, 1]
    
    # Tissue mask: L < threshold and L > 0 (exclude blank and pure black, same as normalize_tiles_new_2.py)
    tissue_mask = (L_full < white_threshold) & (L_full > 0)
    
    if not np.any(tissue_mask):
        # No tissue pixels, return empty arrays
        n = 0
        return {
            "grayscale": np.array([], dtype=np.float64),
            "L": np.array([], dtype=np.float64),
            "a": np.array([], dtype=np.float64),
            "b": np.array([], dtype=np.float64),
            "chroma": np.array([], dtype=np.float64),
            "H": np.array([], dtype=np.float64),
            "S": np.array([], dtype=np.float64),
            "V": np.array([], dtype=np.float64),
            "rgb_std": np.array([], dtype=np.float64),
            "R_G_ratio": np.array([], dtype=np.float64),
            "R_B_ratio": np.array([], dtype=np.float64),
            "R_dominance": np.array([], dtype=np.float64),
            "lab_hue_angle": np.array([], dtype=np.float64),
        }
    
    # Extract features only for tissue pixels
    tissue_mask_flat = tissue_mask.ravel()
    n = np.sum(tissue_mask_flat)

    # Grayscale (0-255)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_flat = gray.ravel()[tissue_mask_flat].astype(np.float64)

    # LAB: L in [0,255], a,b in [-128,127] typically
    L = lab[:, :, 0].ravel()[tissue_mask_flat].astype(np.float64)
    a = (lab[:, :, 1].ravel()[tissue_mask_flat].astype(np.float64) - 128.0)  # center at 0
    b = (lab[:, :, 2].ravel()[tissue_mask_flat].astype(np.float64) - 128.0)
    chroma = np.sqrt(a * a + b * b)

    # HSV: OpenCV H 0-180, S 0-255, V 0-255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0].ravel()[tissue_mask_flat].astype(np.float64)
    S = hsv[:, :, 1].ravel()[tissue_mask_flat].astype(np.float64) / 255.0  # 0-1
    V = hsv[:, :, 2].ravel()[tissue_mask_flat].astype(np.float64) / 255.0  # 0-1

    # Per-pixel RGB std (achromatic => low; colored => higher)
    R = rgb[:, :, 0].ravel()[tissue_mask_flat].astype(np.float64)
    G = rgb[:, :, 1].ravel()[tissue_mask_flat].astype(np.float64)
    B = rgb[:, :, 2].ravel()[tissue_mask_flat].astype(np.float64)
    rgb_std = np.std(np.stack([R, G, B], axis=1), axis=1)

    # Priority 1: Color ratios (R/G, R/B, R dominance)
    # Avoid division by zero
    R_G_ratio = np.where(G > 0, R / G, np.inf)
    R_B_ratio = np.where(B > 0, R / B, np.inf)
    R_dominance = R / (R + G + B + 1e-10)  # R/(R+G+B) percentage

    # Priority 2: LAB hue angle (atan2(b, a) in LAB space)
    # This gives the color direction in the a-b plane
    lab_hue_angle = np.arctan2(b, a)  # Returns angle in radians [-π, π]
    lab_hue_angle_deg = np.degrees(lab_hue_angle)  # Convert to degrees [-180, 180]

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
        # Priority 1: Color ratios
        "R_G_ratio": R_G_ratio,
        "R_B_ratio": R_B_ratio,
        "R_dominance": R_dominance,
        # Priority 2: LAB hue angle
        "lab_hue_angle": lab_hue_angle_deg,
    }


def collect_pixels_from_images(
    images: list[np.ndarray],
    max_pixels: int | None,
    normalizer: TorchVahadaneNormalizer | None = None,
    white_threshold: float = 0.9,
) -> dict[str, np.ndarray]:
    """
    Collect feature vectors from all images, optionally subsampling to max_pixels.
    
    :param images: List of RGB images.
    :param max_pixels: Max pixels to sample per class.
    :param normalizer: Optional TorchVahadaneNormalizer for stain features (Priority 4).
    :return: Dict of feature arrays.
    """
    feature_keys = [
        "grayscale", "L", "a", "b", "chroma", "H", "S", "V", "rgb_std",
        "R_G_ratio", "R_B_ratio", "R_dominance", "lab_hue_angle",
    ]
    # Add stain features if normalizer is provided
    if normalizer is not None:
        feature_keys.extend(["H_concentration", "E_concentration", "H_E_ratio"])
    
    all_features: dict[str, list[np.ndarray]] = {k: [] for k in feature_keys}
    total = 0
    for img in images:
        feats = extract_pixel_features(img, white_threshold=white_threshold)
        
        # Add stain features if normalizer is provided
        if normalizer is not None:
            stain_feats = extract_stain_features(img, normalizer, white_threshold=white_threshold)
            feats.update(stain_feats)
        
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


def collect_shape_features(images: list[np.ndarray]) -> dict[str, list[float]]:
    """Collect shape features from all images (Priority 3)."""
    shape_features = {
        "circularity": [],
        "aspect_ratio": [],
        "area": [],
        "perimeter": [],
    }
    for img in images:
        shapes = extract_shape_features(img)
        for k in shape_features:
            shape_features[k].append(shapes[k])
    return shape_features


def extract_shape_features(rgb: np.ndarray) -> dict[str, float]:
    """
    Extract shape features from an image (Priority 3).
    Assumes each image is a single region. For multi-region images, segmentation is needed first.
    
    :param rgb: RGB uint8 image (H, W, 3).
    :return: Dict with shape features (circularity, aspect_ratio, area, perimeter).
    """
    # Convert to grayscale and threshold to get binary mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Use Otsu's method for automatic thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            "circularity": None,
            "aspect_ratio": None,
            "area": None,
            "perimeter": None,
        }
    
    # Use largest contour (assuming main region)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Circularity: 4π × area / perimeter² (1.0 = perfect circle)
    circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    
    # Aspect ratio: width / height of bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0
    
    return {
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "area": float(area),
        "perimeter": float(perimeter),
    }


def extract_stain_features(rgb: np.ndarray, normalizer: TorchVahadaneNormalizer, white_threshold: float = 0.9) -> dict[str, np.ndarray]:
    """
    Extract stain separation features (Priority 4): Hematoxylin and Eosin concentrations.
    Excludes blank/white areas using luminosity filter.
    
    :param rgb: RGB uint8 image (H, W, 3).
    :param normalizer: TorchVahadaneNormalizer instance (must be fitted).
    :param white_threshold: LAB L threshold (0-1). Pixels with L >= this are considered blank (default 0.9).
    :return: Dict with per-pixel stain features (H_concentration, E_concentration, H_E_ratio), only for tissue pixels.
    """
    # Create tissue mask (same as extract_pixel_features)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L_full = lab[:, :, 0].astype(np.float32) / 255.0  # range [0, 1]
    tissue_mask = (L_full < white_threshold) & (L_full > 0)
    
    if not np.any(tissue_mask):
        # No tissue pixels
        return {
            "H_concentration": np.array([], dtype=np.float64),
            "E_concentration": np.array([], dtype=np.float64),
            "H_E_ratio": np.array([], dtype=np.float64),
        }
    
    # Convert to optical density
    OD = convert_RGB_to_OD_cpu(rgb)
    OD_flat = OD.reshape((-1, 3))  # (N, 3)
    
    # Filter to tissue pixels only
    tissue_mask_flat = tissue_mask.ravel()
    OD_tissue = OD_flat[tissue_mask_flat]  # (n_tissue, 3)
    
    # Get stain matrix from the normalizer (use target matrix if fitted, otherwise extract from image)
    if hasattr(normalizer, 'stain_matrix_target') and normalizer.stain_matrix_target is not None:
        # Use the fitted target stain matrix (convert from tensor if needed)
        stain_matrix = normalizer.stain_matrix_target
        if torch.is_tensor(stain_matrix):
            stain_matrix = stain_matrix.cpu().numpy()
    else:
        # Extract stain matrix from this image
        stain_matrix = normalizer.stain_extractor.get_stain_matrix(rgb)
    
    if stain_matrix is None:
        # If still None, return empty arrays
        n_tissue = OD_tissue.shape[0]
        return {
            "H_concentration": np.zeros(n_tissue, dtype=np.float64),
            "E_concentration": np.zeros(n_tissue, dtype=np.float64),
            "H_E_ratio": np.zeros(n_tissue, dtype=np.float64),
        }
    
    # Ensure stain_matrix is numpy array (2x3: rows=H,E; cols=RGB)
    if torch.is_tensor(stain_matrix):
        stain_matrix = stain_matrix.cpu().numpy()
    stain_matrix = np.asarray(stain_matrix)
    
    # Solve for stain concentrations: OD = stain_matrix^T × concentrations
    # stain_matrix is 2x3, so stain_matrix^T is 3x2
    HE_T = stain_matrix.T  # (3, 2)
    # Solve: OD_tissue = HE_T @ C, so C = (HE_T)^-1 @ OD_tissue
    # But HE_T is 3x2 (overdetermined), so use least squares
    concentrations, _, _, _ = np.linalg.lstsq(HE_T, OD_tissue.T, rcond=None)
    # concentrations is (2, n_tissue): [H concentrations; E concentrations]
    
    H_conc = concentrations[0, :].astype(np.float64)
    E_conc = concentrations[1, :].astype(np.float64)
    
    # H/E ratio (avoid division by zero)
    H_E_ratio = np.where(E_conc > 1e-10, H_conc / E_conc, np.inf)
    
    return {
        "H_concentration": H_conc,
        "E_concentration": E_conc,
        "H_E_ratio": H_E_ratio,
    }


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
        description="Extract features from RBC vs regular tissue sample images for threshold selection."
    )
    parser.add_argument(
        "--rbc_dir",
        type=Path,
        default=DEFAULT_RBC_DIR,
        help=f"Directory containing sample images of red blood cells (default: {DEFAULT_RBC_DIR}).",
    )
    parser.add_argument(
        "--regular_dir",
        type=Path,
        default=DEFAULT_REGULAR_DIR,
        help=f"Directory containing sample images of regular tissue (default: {DEFAULT_REGULAR_DIR}).",
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
    parser.add_argument(
        "--enable_stain_features",
        action="store_true",
        help="Enable stain separation features (Priority 4). Requires fitting normalizer on sample images.",
    )
    parser.add_argument(
        "--white_threshold",
        type=float,
        default=0.9,
        help="LAB L threshold (0-1) for excluding blank/white areas. Pixels with L >= this are excluded (default: 0.9, matches normalize_tiles_new_2.py).",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    if not args.rbc_dir.is_dir():
        raise FileNotFoundError(f"RBC directory not found: {args.rbc_dir}")
    if not args.regular_dir.is_dir():
        raise FileNotFoundError(f"Regular tissue directory not found: {args.regular_dir}")

    rbc_images = load_images_from_dir(args.rbc_dir)
    regular_images = load_images_from_dir(args.regular_dir)

    if not rbc_images:
        raise RuntimeError(f"No images found in {args.rbc_dir}")
    if not regular_images:
        raise RuntimeError(f"No images found in {args.regular_dir}")

    print(f"Loaded {len(rbc_images)} RBC images, {len(regular_images)} regular tissue images.")

    # Priority 4: Fit normalizer for stain separation (if enabled)
    normalizer = None
    if args.enable_stain_features:
        print("Fitting stain normalizer for stain separation features...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        normalizer = TorchVahadaneNormalizer(device=device, correct_exposure=False)
        # Fit on a representative sample (use first regular tissue image as reference)
        if regular_images:
            normalizer.fit(regular_images[0])
            print(f"Normalizer fitted on device: {device}")
        else:
            print("Warning: No regular images to fit normalizer, disabling stain features.")
            normalizer = None

    # Collect pixel features (including stain features if enabled)
    # Exclude blank/white areas using luminosity filter (same as normalize_tiles_new_2.py)
    print(f"Using luminosity filter: excluding pixels with LAB L >= {args.white_threshold} (blank areas)")
    rbc_pixels = collect_pixels_from_images(rbc_images, args.max_pixels_per_class, normalizer, args.white_threshold)
    regular_pixels = collect_pixels_from_images(regular_images, args.max_pixels_per_class, normalizer, args.white_threshold)

    n_rbc = rbc_pixels["grayscale"].size
    n_regular = regular_pixels["grayscale"].size
    print(f"Collected {n_rbc} RBC pixels, {n_regular} regular tissue pixels.")

    # Priority 3: Collect shape features (per-image)
    print("Extracting shape features (Priority 3)...")
    rbc_shapes = collect_shape_features(rbc_images)
    regular_shapes = collect_shape_features(regular_images)

    rbc_stats = {k: compute_stats(rbc_pixels[k]) for k in rbc_pixels}
    regular_stats = {k: compute_stats(regular_pixels[k]) for k in regular_pixels}
    
    # Compute shape stats (aggregate per-image features)
    rbc_shape_stats = {k: compute_stats(np.array([v for v in rbc_shapes[k] if v is not None])) for k in rbc_shapes}
    regular_shape_stats = {k: compute_stats(np.array([v for v in regular_shapes[k] if v is not None])) for k in regular_shapes}

    report = {
        "rbc_dir": str(args.rbc_dir),
        "regular_dir": str(args.regular_dir),
        "n_rbc_images": len(rbc_images),
        "n_regular_images": len(regular_images),
        "n_rbc_pixels": n_rbc,
        "n_regular_pixels": n_regular,
        "stain_features_enabled": args.enable_stain_features,
        "rbc": rbc_stats,
        "regular": regular_stats,
        "rbc_shape": rbc_shape_stats,
        "regular_shape": regular_shape_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved stats to {args.output}")

    # Print summary table for key discriminative features
    print("\n--- Summary: Pixel Features (use to choose thresholds) ---")
    print("Feature              RBC (mean ± std)          Regular (mean ± std)")
    print("-" * 75)
    for key, label in [
        ("grayscale", "Grayscale"),
        ("L", "LAB L"),
        ("chroma", "LAB chroma"),
        ("H", "HSV H (0-180)"),
        ("S", "HSV S (0-1)"),
        ("V", "HSV V (0-1)"),
        ("rgb_std", "RGB std"),
        ("R_G_ratio", "R/G ratio"),
        ("R_B_ratio", "R/B ratio"),
        ("R_dominance", "R dominance"),
        ("lab_hue_angle", "LAB hue angle"),
    ]:
        if key not in rbc_stats:
            continue
        r = rbc_stats[key]
        reg = regular_stats[key]
        rm, rs = r["mean"], r["std"]
        regm, regs = reg["mean"], reg["std"]
        if rm is None or regm is None:
            continue
        print(f"{label:20} {rm:7.2f} ± {rs or 0:5.2f}    {regm:7.2f} ± {regs or 0:5.2f}")
    
    if args.enable_stain_features:
        print("\n--- Stain Features (Priority 4) ---")
        for key, label in [
            ("H_concentration", "H concentration"),
            ("E_concentration", "E concentration"),
            ("H_E_ratio", "H/E ratio"),
        ]:
            if key not in rbc_stats:
                continue
            r = rbc_stats[key]
            reg = regular_stats[key]
            rm, rs = r["mean"], r["std"]
            regm, regs = reg["mean"], reg["std"]
            if rm is None or regm is None:
                continue
            print(f"{label:20} {rm:7.2f} ± {rs or 0:5.2f}    {regm:7.2f} ± {regs or 0:5.2f}")
    
    print("\n--- Shape Features (Priority 3) ---")
    print("Feature              RBC (mean ± std)          Regular (mean ± std)")
    print("-" * 75)
    for key, label in [
        ("circularity", "Circularity"),
        ("aspect_ratio", "Aspect ratio"),
        ("area", "Area"),
        ("perimeter", "Perimeter"),
    ]:
        r = rbc_shape_stats[key]
        reg = regular_shape_stats[key]
        rm, rs = r["mean"], r["std"]
        regm, regs = reg["mean"], reg["std"]
        if rm is None or regm is None:
            continue
        print(f"{label:20} {rm:7.2f} ± {rs or 0:5.2f}    {regm:7.2f} ± {regs or 0:5.2f}")
    
    print("\nSuggested: use R/G ratio, R/B ratio, LAB hue angle, or E concentration to separate RBCs from regular tissue.")


if __name__ == "__main__":
    main()
