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

import argparse
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

# Default paths (overridable via CLI)
SCRIPT_DIR = Path(__file__).resolve().parent
REF_IMAGE_DIR = SCRIPT_DIR / "ref_image"
RAW_TILES_DIR = SCRIPT_DIR / "original_tiles"  # default; override with --input
OUTPUT_DIR = SCRIPT_DIR / "normalized_tiles"   # default; override with --output
WSI_FEATURES_DIR = SCRIPT_DIR / "wsi_features"

# Configuration: tissue brightness uses LAB L (same as stain_extractor_cpu/gpu)
LUMINANCE_PERCENTILE = 95.0  # Percentile for tissue LAB L (90.0, 95.0, 99.0). Lower = more aggressive.

# Configuration: black artifact detection (carbon dots, pollution, etc.)
# Stage 1: grayscale filter – focus on dark pixels only.
# Stage 2: chroma filter – among dark pixels, low chroma = artifact (achromatic), high chroma = dark nuclei (keep).
GRAYSCALE_DARK_THRESHOLD = 35   # Grayscale (0-255): pixels darker than this enter stage 2
CHROMA_ARTIFACT_THRESHOLD = 20  # LAB chroma: below this = artifact (artifacts ~4, nuclei ~37 from blob_nuclei_features.json)
BLACK_ARTIFACT_MAX_AREA = None  # Max blob area in pixels; None = no limit (remove all dark+achromatic regions)
BLACK_ARTIFACT_ENABLED = True   # Set to False to disable artifact filtering

# Configuration: RBC (red blood cell) detection and removal
# Uses R_G_ratio as main factor, R_B_ratio and R_dominance as subsidiary factors.
# Focuses on dark RBCs (overlapping/bad angle) that might be mistaken for nuclei.
# Based on rbc_regular_features.json: RBC R_G_ratio 1.69-3.21, Regular 1.17-2.02 (no overlap).
# Loosened thresholds to reduce false positives (normal tissue removal).
R_G_RATIO_THRESHOLD = 2.2       # Main factor: R/G ratio above this = RBC candidate (raised from 2.0 for fewer false positives)
R_B_RATIO_THRESHOLD = 1.50      # Subsidiary: R/B ratio above this supports RBC (raised from 1.45)
R_DOMINANCE_THRESHOLD = 0.48    # Subsidiary: R/(R+G+B) above this supports RBC (raised from 0.45)
RBC_DARK_THRESHOLD = 100        # Only remove RBCs darker than this (grayscale 0-255). Targets dark RBCs that might be mistaken for nuclei.
RBC_CHROMA_SAFEGUARD = 55.0     # Chroma safeguard: if chroma > this AND R/G ratio < 2.4, exclude (likely purple nucleus, not RBC)
RBC_ENABLED = True              # Set to False to disable RBC filtering

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


def detect_black_artifact_mask(
    img: np.ndarray,
    grayscale_threshold: int = 35,
    chroma_threshold: float = 20.0,
    max_area: int | None = None,
) -> np.ndarray:
    """
    Detect dark artifact regions (carbon, pollution, etc.) to exclude from normalization,
    so they do not turn purple and get misidentified as nuclei.

    Two-stage filter:
      1. Grayscale: keep only dark pixels (grayscale < grayscale_threshold).
      2. Chroma: among dark pixels, low chroma = achromatic = artifact; high chroma = dark purple nuclei = keep.

    Optionally, connected-component max_area can limit removal to small blobs only.

    :param img: RGB uint8 image (H, W, 3).
    :param grayscale_threshold: Grayscale (0-255). Darker pixels enter stage 2 (default 35).
    :param chroma_threshold: LAB chroma. Below this = artifact (default 20).
    :param max_area: Max blob area in pixels; None = no limit (default None).
    :return: Boolean mask where True indicates artifact pixels to exclude.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return np.zeros(img.shape[:2], dtype=bool)

    # Stage 1: grayscale – focus on dark pixels only
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_mask = gray < grayscale_threshold

    if not np.any(dark_mask):
        return np.zeros(img.shape[:2], dtype=bool)

    # Stage 2: chroma – among dark pixels, low chroma = artifact (achromatic)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # OpenCV LAB: L 0-255, a/b 0-255 with 128 as neutral
    a = lab[:, :, 1].astype(np.float64) - 128.0
    b = lab[:, :, 2].astype(np.float64) - 128.0
    chroma = np.sqrt(a * a + b * b)
    chroma_low = chroma < chroma_threshold

    artifact_candidate = dark_mask & chroma_low

    if not np.any(artifact_candidate):
        return np.zeros(img.shape[:2], dtype=bool)

    # Optional: limit to blobs with area <= max_area (None = no limit)
    if max_area is None or max_area <= 0:
        return artifact_candidate

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        artifact_candidate.astype(np.uint8), connectivity=8
    )
    artifact_mask = np.zeros(img.shape[:2], dtype=bool)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area <= max_area:
            artifact_mask[labels == label_id] = True
    return artifact_mask


def remove_black_artifacts(
    img: np.ndarray,
    artifact_mask: np.ndarray,
    replacement_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Replace artifact pixels with background-like color to prevent them from being normalized.

    :param img: RGB uint8 image (H, W, 3).
    :param artifact_mask: Boolean mask where True indicates artifact pixels.
    :param replacement_color: RGB color to use for artifacts (default: white).
    :return: Image with artifacts replaced.
    """
    if not np.any(artifact_mask):
        return img

    out = img.copy()
    out[artifact_mask] = replacement_color
    return out


def detect_rbc_mask(
    img: np.ndarray,
    r_g_ratio_threshold: float = 2.2,
    r_b_ratio_threshold: float = 1.50,
    r_dominance_threshold: float = 0.48,
    dark_threshold: int | None = 100,
    chroma_safeguard: float = 55.0,
) -> np.ndarray:
    """
    Detect red blood cells (RBCs) to exclude from normalization.
    Focuses on dark RBCs (overlapping/bad angle) that might be mistaken for nuclei.
    
    Uses R_G_ratio as main factor, R_B_ratio and R_dominance as subsidiary factors.
    Filters by darkness to target only problematic dark RBCs.
    Includes chroma safeguard to avoid removing purple nuclei (high chroma but lower R/G ratio).
    
    Based on rbc_regular_features.json analysis: RBCs have higher R/G, R/B, and R dominance.
    
    :param img: RGB uint8 image (H, W, 3).
    :param r_g_ratio_threshold: Main factor: R/G ratio above this = RBC candidate (default 2.2, loosened).
    :param r_b_ratio_threshold: Subsidiary: R/B ratio above this supports RBC (default 1.50, loosened).
    :param r_dominance_threshold: Subsidiary: R/(R+G+B) above this supports RBC (default 0.48, loosened).
    :param dark_threshold: Grayscale threshold (0-255). Only remove RBCs darker than this (default 100).
    :param chroma_safeguard: Chroma threshold. If chroma > this AND R/G < 2.4, exclude (likely purple nucleus).
    :return: Boolean mask where True indicates RBC pixels to exclude.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return np.zeros(img.shape[:2], dtype=bool)
    
    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)
    
    # Main factor: R/G ratio
    r_g_ratio = np.where(G > 0, R / G, np.inf)
    rbc_main = r_g_ratio > r_g_ratio_threshold
    
    if not np.any(rbc_main):
        return np.zeros(img.shape[:2], dtype=bool)
    
    # Subsidiary factors: R/B ratio and R dominance
    r_b_ratio = np.where(B > 0, R / B, np.inf)
    r_dominance = R / (R + G + B + 1e-10)
    
    # Combine: main factor AND (at least one subsidiary factor)
    rbc_subsidiary = (r_b_ratio > r_b_ratio_threshold) | (r_dominance > r_dominance_threshold)
    rbc_mask = rbc_main & rbc_subsidiary
    
    # Darkness filter: only remove dark RBCs (overlapping/bad angle that might be mistaken for nuclei)
    if dark_threshold is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dark_mask = gray < dark_threshold
        rbc_mask = rbc_mask & dark_mask
    
    # Chroma safeguard: avoid removing purple nuclei
    # Purple nuclei have high chroma but lower R/G ratio. If chroma is very high but R/G is borderline,
    # it's likely a purple nucleus, not a red RBC.
    if chroma_safeguard is not None and np.any(rbc_mask):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        a = lab[:, :, 1].astype(np.float64) - 128.0
        b = lab[:, :, 2].astype(np.float64) - 128.0
        chroma = np.sqrt(a * a + b * b)
        
        # Exclude pixels with very high chroma AND moderate R/G ratio (likely purple nucleus)
        # RBCs have high chroma (~53) but also very high R/G (>2.2). Purple nuclei have high chroma but R/G < 2.0 typically.
        # Safeguard: if chroma > threshold AND R/G < 2.4, exclude from removal
        high_chroma = chroma > chroma_safeguard
        moderate_r_g = r_g_ratio < 2.4  # Borderline R/G ratio
        purple_nucleus_likely = high_chroma & moderate_r_g
        
        # Remove purple nucleus candidates from RBC mask
        rbc_mask = rbc_mask & ~purple_nucleus_likely
    
    return rbc_mask


def remove_rbcs(
    img: np.ndarray,
    rbc_mask: np.ndarray,
    replacement_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Replace RBC pixels with background-like color to prevent them from being normalized.

    :param img: RGB uint8 image (H, W, 3).
    :param rbc_mask: Boolean mask where True indicates RBC pixels.
    :param replacement_color: RGB color to use for RBCs (default: white).
    :return: Image with RBCs replaced.
    """
    if not np.any(rbc_mask):
        return img

    out = img.copy()
    out[rbc_mask] = replacement_color
    return out


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


def parse_args():
    parser = argparse.ArgumentParser(description="Stain-normalize H&E tiles using a reference image.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_TILES_DIR,
        help="Directory containing raw tile images (default: script dir / original_tiles)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for normalized output images (default: script dir / normalized_tiles)",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=REF_IMAGE_DIR,
        help="Directory containing the reference image (default: script dir / ref_image)",
    )
    parser.add_argument(
        "--wsi-features",
        type=Path,
        default=WSI_FEATURES_DIR,
        help="Directory with per-WSI stain matrices from extract_wsi_features.py (default: script dir / wsi_features)",
    )
    parser.add_argument(
        "--grayscale-dark-threshold",
        type=int,
        default=GRAYSCALE_DARK_THRESHOLD,
        help=f"Stage 1: grayscale (0-255). Pixels darker than this are considered dark (default: {GRAYSCALE_DARK_THRESHOLD}).",
    )
    parser.add_argument(
        "--chroma-artifact-threshold",
        type=float,
        default=CHROMA_ARTIFACT_THRESHOLD,
        help=f"Stage 2: LAB chroma. Among dark pixels, below this = artifact (default: {CHROMA_ARTIFACT_THRESHOLD}).",
    )
    parser.add_argument(
        "--black-artifact-max-area",
        type=int,
        default=BLACK_ARTIFACT_MAX_AREA,
        metavar="N",
        help="Optional max blob area (pixels). 0 or omit for no limit (default: no limit).",
    )
    parser.add_argument(
        "--disable-black-artifact-filter",
        action="store_true",
        help="Disable black artifact filtering (artifacts will be normalized normally).",
    )
    parser.add_argument(
        "--r-g-ratio-threshold",
        type=float,
        default=R_G_RATIO_THRESHOLD,
        help=f"RBC filter main factor: R/G ratio above this = RBC candidate (default: {R_G_RATIO_THRESHOLD}).",
    )
    parser.add_argument(
        "--r-b-ratio-threshold",
        type=float,
        default=R_B_RATIO_THRESHOLD,
        help=f"RBC filter subsidiary: R/B ratio above this supports RBC (default: {R_B_RATIO_THRESHOLD}).",
    )
    parser.add_argument(
        "--r-dominance-threshold",
        type=float,
        default=R_DOMINANCE_THRESHOLD,
        help=f"RBC filter subsidiary: R/(R+G+B) above this supports RBC (default: {R_DOMINANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--rbc-dark-threshold",
        type=int,
        default=RBC_DARK_THRESHOLD,
        metavar="N",
        help=f"RBC filter: only remove RBCs darker than this (grayscale 0-255, default: {RBC_DARK_THRESHOLD}). Use 0 to disable darkness filter.",
    )
    parser.add_argument(
        "--rbc-chroma-safeguard",
        type=float,
        default=RBC_CHROMA_SAFEGUARD,
        help=f"RBC filter chroma safeguard: if chroma > this AND R/G < 2.4, exclude (likely purple nucleus, default: {RBC_CHROMA_SAFEGUARD}). Use 0 to disable.",
    )
    parser.add_argument(
        "--disable-rbc-filter",
        action="store_true",
        help="Disable RBC filtering (RBCs will be normalized normally).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_tiles_dir = args.input
    output_dir = args.output
    ref_image_dir = args.ref_dir
    wsi_features_dir = args.wsi_features

    ref_path = find_reference_image(ref_image_dir)
    print(f"Reference image: {ref_path}")

    if not raw_tiles_dir.exists():
        raise FileNotFoundError(
            f"Raw tiles directory not found: {raw_tiles_dir}"
        )

    tile_paths = [
        p
        for p in raw_tiles_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not tile_paths:
        raise FileNotFoundError(f"No image files found in {raw_tiles_dir}")

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

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Print filter status
    if not args.disable_black_artifact_filter:
        area_str = str(args.black_artifact_max_area) if args.black_artifact_max_area else "no limit"
        print(
            f"Black artifact filter: enabled (grayscale<{args.grayscale_dark_threshold}, chroma<{args.chroma_artifact_threshold}, max_area={area_str})"
        )
    else:
        print("Black artifact filter: disabled")
    
    if not args.disable_rbc_filter:
        dark_str = f", dark<{args.rbc_dark_threshold}" if args.rbc_dark_threshold and args.rbc_dark_threshold > 0 else ""
        chroma_str = f", chroma_safe>{args.rbc_chroma_safeguard}" if args.rbc_chroma_safeguard and args.rbc_chroma_safeguard > 0 else ""
        print(
            f"RBC filter: enabled (R/G>{args.r_g_ratio_threshold}, R/B>{args.r_b_ratio_threshold} OR R_dom>{args.r_dominance_threshold}{dark_str}{chroma_str})"
        )
    else:
        print("RBC filter: disabled")

    # Known WSI IDs from wsi_features (prefix match for tile names).
    known_wsi_ids = get_known_wsi_ids(wsi_features_dir)
    if known_wsi_ids:
        print(f"Known WSI IDs from {wsi_features_dir}: {sorted(known_wsi_ids)}")

    # Cache of per-WSI stain matrices so we only load each once.
    wsi_stain_cache: Dict[str, np.ndarray] = {}

    for i, path in enumerate(tile_paths):
        out_path = output_dir / path.name
        if out_path.exists():
            print(f"  {path.name} has been processed, skipping")
            continue
        tile = cv2.imread(str(path))
        if tile is None:
            print(f"  Skip (unreadable): {path.name}")
            continue
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        # Detect and remove black artifacts (carbon dots, pollution, etc.) before normalization
        # to prevent them from turning purple and being misidentified as nuclei.
        if not args.disable_black_artifact_filter:
            artifact_mask = detect_black_artifact_mask(
                tile_rgb,
                grayscale_threshold=args.grayscale_dark_threshold,
                chroma_threshold=args.chroma_artifact_threshold,
                max_area=args.black_artifact_max_area,
            )
            if np.any(artifact_mask):
                n_artifacts = np.sum(artifact_mask)
                tile_rgb = remove_black_artifacts(tile_rgb, artifact_mask)
                print(f"    Removed {n_artifacts} black artifact pixels")
        
        # Detect and remove RBCs (red blood cells) before normalization.
        # Focuses on dark RBCs (overlapping/bad angle) that might be mistaken for nuclei.
        # Uses R_G_ratio as main factor, R_B_ratio and R_dominance as subsidiary factors.
        # Includes chroma safeguard to avoid removing purple nuclei.
        if not args.disable_rbc_filter:
            dark_thresh = args.rbc_dark_threshold if args.rbc_dark_threshold and args.rbc_dark_threshold > 0 else None
            chroma_safe = args.rbc_chroma_safeguard if args.rbc_chroma_safeguard and args.rbc_chroma_safeguard > 0 else None
            rbc_mask = detect_rbc_mask(
                tile_rgb,
                r_g_ratio_threshold=args.r_g_ratio_threshold,
                r_b_ratio_threshold=args.r_b_ratio_threshold,
                r_dominance_threshold=args.r_dominance_threshold,
                dark_threshold=dark_thresh,
                chroma_safeguard=chroma_safe,
            )
            if np.any(rbc_mask):
                n_rbcs = np.sum(rbc_mask)
                tile_rgb = remove_rbcs(tile_rgb, rbc_mask)
                print(f"    Removed {n_rbcs} RBC pixels")

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
                stain_path = wsi_features_dir / f"{wsi_id}_stain_matrix.npy"
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

    print(f"Done. Normalized {len(tile_paths)} tiles -> {output_dir}")


if __name__ == "__main__":
    main()
