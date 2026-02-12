#!/usr/bin/env python3
"""
Stain-normalize H&E tiles using a reference image.
Follows the NucSegAI stain_norm_new pattern:
  - Tissue-only brightness standardization (scale only tissue pixels; blank regions unchanged).
  - Vahadane normalization on standardized reference and tiles.

Reference: ref_image/
Raw tiles: /mnt/d/Downloads/Programs/original_tiles (WSL path for D:\Downloads\Programs\original_tiles)
Output: saved to an output directory (default: normalized_tiles in project, or under the same WSL tree).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvahadane import TorchVahadaneNormalizer

# Use GPU-only stain extraction (no spams) if spams is not installed
try:
    import spams  # noqa: F401
    STAINTOOLS_ESTIMATE = True
except ImportError:
    STAINTOOLS_ESTIMATE = False
    print("spams not found: using GPU stain extraction (staintools_estimate=False)")

# Paths (WSL format for Windows D:\...)
REF_IMAGE_DIR = Path(__file__).resolve().parent / "ref_image"
RAW_TILES_DIR = Path("/mnt/d/Downloads/Programs/original_tiles")
OUTPUT_DIR = Path("/mnt/d/Downloads/Programs/normalized_tiles")  # or use project: Path(__file__).parent / "normalized_tiles"

# Configuration (match stain_norm_new.py)
LUMINANCE_PERCENTILE = 95.0  # Percentile for tissue luminance (90.0, 95.0, 99.0). Lower = more aggressive.

# Image extensions to consider
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def tissue_only_brightness_standardize(
    img: np.ndarray,
    white_threshold: float = 0.9,
    target_p95: float = 0.9,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    luminance_percentile: float = 95.0,
) -> np.ndarray:
    """
    Standardize brightness using only non-blank (tissue) pixels.
    Blank pixels (luminance >= white_threshold) are left unchanged so tiles with
    large white regions are not over-darkened.

    :param img: RGB uint8 image (H, W, 3).
    :param white_threshold: Luminance above this is treated as blank (default 0.9).
    :param target_p95: Target luminance percentile value after scaling (default 0.9).
    :param min_scale: Minimum scale factor (default 0.5).
    :param max_scale: Maximum scale factor (default 2.0).
    :param luminance_percentile: Percentile for tissue luminance (default 95.0).
    :return: RGB uint8 image with standardized brightness.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    img_float = img.astype(np.float32) / 255.0
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # Rec. 709

    tissue_mask = lum < white_threshold
    if not np.any(tissue_mask):
        return img

    tissue_luminance_p = np.percentile(lum[tissue_mask], luminance_percentile)
    if tissue_luminance_p <= 0:
        return img

    scale = target_p95 / tissue_luminance_p
    scale = float(np.clip(scale, min_scale, max_scale))

    out = img_float.copy()
    out[tissue_mask] *= scale
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(img.dtype)


def find_reference_image(ref_dir: Path) -> Path:
    """Find first reference image in ref_image directory."""
    for ext in IMAGE_EXTENSIONS:
        candidates = list(ref_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No image found in {ref_dir}")


def main():
    ref_path = find_reference_image(REF_IMAGE_DIR)
    print(f"Reference image: {ref_path}")

    if not RAW_TILES_DIR.exists():
        raise FileNotFoundError(
            f"Raw tiles directory not found: {RAW_TILES_DIR}\n"
            "Ensure the Windows path D:\\Downloads\\Programs\\original_tiles is accessible in WSL."
        )

    tile_paths = [
        p
        for p in RAW_TILES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not tile_paths:
        raise FileNotFoundError(f"No image files found in {RAW_TILES_DIR}")

    print(f"Found {len(tile_paths)} tiles. Fitting normalizer on reference...")

    # Load reference as RGB (supports .tiff, .png, etc.)
    ref = cv2.imread(str(ref_path))
    if ref is None:
        from PIL import Image
        pil_img = np.array(Image.open(ref_path))
        if pil_img.ndim == 2:
            ref = cv2.cvtColor(pil_img, cv2.COLOR_GRAY2RGB)
        elif pil_img.shape[2] == 4:
            ref = cv2.cvtColor(pil_img, cv2.COLOR_RGBA2RGB)
        else:
            ref = pil_img
    else:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    # Tissue-only brightness standardization (stain_norm_new pattern)
    ref = tissue_only_brightness_standardize(ref, luminance_percentile=LUMINANCE_PERCENTILE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    normalizer = TorchVahadaneNormalizer(device=device, staintools_estimate=STAINTOOLS_ESTIMATE, correct_exposure=False)
    normalizer.fit(ref)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    for i, path in enumerate(tile_paths):
        out_path = OUTPUT_DIR / path.name
        if out_path.exists():
            print(f"  {path.name} has been processed, skipping")
            continue
        tile = cv2.imread(str(path))
        if tile is None:
            print(f"  Skip (unreadable): {path.name}")
            continue
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_std = tissue_only_brightness_standardize(tile_rgb, luminance_percentile=LUMINANCE_PERCENTILE)
        normed = normalizer.transform(tile_std)
        out_arr = normed.cpu().numpy() if hasattr(normed, "cpu") else normed
        out_arr = out_arr.astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR))
        print(f"  {i + 1}/{len(tile_paths)}: {path.name} -> {out_path.name}")

    print(f"Done. Normalized {len(tile_paths)} tiles -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
