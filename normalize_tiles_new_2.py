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
from typing import Dict

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

# Directory containing precomputed per-WSI stain features from extract_wsi_features.py
WSI_FEATURES_DIR = Path(__file__).resolve().parent / "wsi_features"

# Configuration: tissue brightness uses LAB L (same as stain_extractor_cpu/gpu)
LUMINANCE_PERCENTILE = 95.0  # Percentile for tissue LAB L (90.0, 95.0, 99.0). Lower = more aggressive.

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
    Uses LAB L for tissue mask and percentile, aligned with stain_extractor_cpu/gpu
    so the same pixels are considered tissue as in stain normalization.

    Blank pixels (L >= white_threshold) and black (L == 0) are left unchanged.

    :param img: RGB uint8 image (H, W, 3).
    :param white_threshold: LAB L above this is treated as blank (default 0.9).
    :param target_p95: Target LAB L percentile value after scaling (default 0.9).
    :param min_scale: Minimum scale factor (default 0.5).
    :param max_scale: Maximum scale factor (default 2.0).
    :param luminance_percentile: Percentile for tissue LAB L (default 95.0).
    :return: RGB uint8 image with standardized brightness.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    # Same luminance definition as stain_extractor_cpu / stain_extractor_gpu (cv2 path)
    I_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0].astype(np.float32) / 255.0  # range [0, 1]

    # Tissue mask: L < threshold and L > 0 (exclude black background like extractors)
    tissue_mask = (L < white_threshold) & (L > 0)
    if not np.any(tissue_mask):
        return img

    tissue_L_p = np.percentile(L[tissue_mask], luminance_percentile)
    if tissue_L_p <= 0:
        return img

    scale = target_p95 / tissue_L_p
    scale = float(np.clip(scale, min_scale, max_scale))

    img_float = img.astype(np.float32) / 255.0
    out = img_float.copy()
    out[tissue_mask] *= scale
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(img.dtype)


def get_known_wsi_ids(wsi_features_dir: Path) -> set[str]:
    """
    Collect WSI IDs from wsi_features by listing *_stain_matrix.npy files.
    Each such file gives one WSI ID (filename with _stain_matrix.npy stripped).
    """
    ids: set[str] = set()
    if not wsi_features_dir.exists():
        return ids
    for p in wsi_features_dir.glob("*_stain_matrix.npy"):
        wsi_id = p.name.replace("_stain_matrix.npy", "")
        ids.add(wsi_id)
    return ids


def infer_wsi_id_from_tile_name(tile_path: Path, known_wsi_ids: set[str]) -> str:
    """
    Infer the originating WSI ID from a tile filename by prefix matching.

    Tile names start with their original WSI ID. We match the longest known
    WSI ID that is a prefix of the tile stem so the correct slide is used.

    Examples (with known_wsi_ids containing JN_TS_001, JN_TS_013, ...):
        JN_TS_013_bg_tile_10309_3904.tiff -> JN_TS_013
        JN_TS_013_margin_tile_14853_8902.tiff -> JN_TS_013
        JN_TS_013_tumour_inv_tile_18392_16717.tiff -> JN_TS_013

    If no known WSI ID matches, returns empty string (caller will fall back
    to per-tile stain estimation).
    """
    stem = tile_path.stem
    # Try longest IDs first so e.g. JN_TS_013 matches before JN_TS or JN
    for wsi_id in sorted(known_wsi_ids, key=len, reverse=True):
        if stem == wsi_id or stem.startswith(wsi_id + "_"):
            return wsi_id
    return ""


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
    normalizer = TorchVahadaneNormalizer(
        device=device,
        staintools_estimate=STAINTOOLS_ESTIMATE,
        correct_exposure=False,
    )

    # Fit on the reference image to define the TARGET stain style
    # (stain_matrix_target and maxC_target).
    normalizer.fit(ref)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Known WSI IDs from wsi_features (prefix match for tile names).
    known_wsi_ids = get_known_wsi_ids(WSI_FEATURES_DIR)
    if known_wsi_ids:
        print(f"Known WSI IDs from {WSI_FEATURES_DIR}: {sorted(known_wsi_ids)}")

    # Cache of per-WSI stain matrices so we only load each once.
    wsi_stain_cache: Dict[str, np.ndarray] = {}

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

        # Use the precomputed WSI-level stain matrix (from extract_wsi_features.py)
        # to describe how the source slide is stained, while still mapping to the
        # TARGET style defined by ref_image. Match tile name by WSI prefix.
        wsi_id = infer_wsi_id_from_tile_name(path, known_wsi_ids)
        stain_matrix = None

        # Try to load and cache the stain matrix for this WSI (when prefix matched).
        if wsi_id:
            if wsi_id in wsi_stain_cache:
                stain_matrix = wsi_stain_cache[wsi_id]
            else:
                stain_path = WSI_FEATURES_DIR / f"{wsi_id}_stain_matrix.npy"
                if stain_path.exists():
                    try:
                        stain_matrix = np.load(stain_path)
                        wsi_stain_cache[wsi_id] = stain_matrix
                        print(f"  Using WSI stain matrix for slide {wsi_id}: {stain_path.name}")
                    except Exception as exc:
                        print(f"  Warning: failed to load {stain_path}: {exc}")
                else:
                    print(f"  Warning: no WSI stain matrix found for slide {wsi_id} at {stain_path}")

        # If we have a per-WSI stain matrix, fix it for this tile; otherwise let
        # TorchVahadane estimate per-tile stains as usual.
        if stain_matrix is not None:
            normalizer.set_stain_matrix(stain_matrix)
        else:
            # Clear any previously fixed matrix so we don't accidentally reuse
            # a different slide's matrix.
            normalizer.stain_m_fixed = None

        tile_std = tissue_only_brightness_standardize(
            tile_rgb,
            luminance_percentile=LUMINANCE_PERCENTILE,
        )
        normed = normalizer.transform(tile_std)
        out_arr = normed.cpu().numpy() if hasattr(normed, "cpu") else normed
        out_arr = out_arr.astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR))
        print(f"  {i + 1}/{len(tile_paths)}: {path.name} -> {out_path.name}")

    print(f"Done. Normalized {len(tile_paths)} tiles -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
