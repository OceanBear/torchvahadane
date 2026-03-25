#!/usr/bin/env python3
"""
Stain-normalize H&E tiles using a reference image.
Follows the NucSegAI stain_norm_new pattern:
  - Tissue-only brightness standardization (scale only tissue pixels; blank regions unchanged).
  - Vahadane normalization on standardized reference and tiles.
  - Optional black-artifact handling (same detection as normalize_tiles_new_2): those
    pixels are excluded from maxC and copied from the original RGB in the output.
  - Optional RBC handling (same detection as normalize_tiles_new_2): RBC pixels are
    excluded from maxC and copied from the original RGB (ignored for stain norm, not
    painted white).

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

# Black artifact detection (same logic as normalize_tiles_new_2.py).
# Detected pixels are excluded from Vahadane maxC scaling and left unchanged in the output.
GRAYSCALE_DARK_THRESHOLD = 35
CHROMA_ARTIFACT_THRESHOLD = 20
V_ARTIFACT_THRESHOLD = 0.20
RGB_STD_ARTIFACT_THRESHOLD = 12.0
BLACK_ARTIFACT_MAX_AREA = None
BLACK_ARTIFACT_ENABLED = True

# RBC detection (same logic as normalize_tiles_new_2.py); excluded from maxC, original RGB kept.
R_G_RATIO_THRESHOLD = 2.3
R_B_RATIO_THRESHOLD = 1.60
R_DOMINANCE_THRESHOLD = 0.50
RBC_DARK_THRESHOLD = 100
RBC_CHROMA_SAFEGUARD = 55.0
RBC_ENABLED = True

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
    v_threshold: float = 0.20,
    rgb_std_threshold: float = 12.0,
    max_area: int | None = None,
) -> np.ndarray:
    """
    Two-stage dark artifact mask (carbon, pollution, etc.), matching normalize_tiles_new_2.

    Stage 1: grayscale < threshold (dark pixels only).
    Stage 2: among those, require low LAB chroma AND low HSV V AND low per-pixel RGB std.

    Returns a boolean mask (H, W) where True = artifact (exclude from maxC; preserve in output).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return np.zeros(img.shape[:2], dtype=bool)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_mask = gray < grayscale_threshold
    if not np.any(dark_mask):
        return np.zeros(img.shape[:2], dtype=bool)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    a = lab[:, :, 1].astype(np.float64) - 128.0
    b = lab[:, :, 2].astype(np.float64) - 128.0
    chroma = np.sqrt(a * a + b * b)
    chroma_low = chroma < chroma_threshold

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float64) / 255.0
    v_low = v < v_threshold

    r = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    b_ch = img[:, :, 2].astype(np.float64)
    rgb_std = np.std(np.stack([r, g, b_ch], axis=2), axis=2)
    rgb_std_low = rgb_std < rgb_std_threshold

    artifact_candidate = dark_mask & chroma_low & v_low & rgb_std_low
    if not np.any(artifact_candidate):
        return np.zeros(img.shape[:2], dtype=bool)

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


def detect_rbc_mask(
    img: np.ndarray,
    r_g_ratio_threshold: float = 2.3,
    r_b_ratio_threshold: float = 1.60,
    r_dominance_threshold: float = 0.50,
    dark_threshold: int | None = 100,
    chroma_safeguard: float = 55.0,
) -> np.ndarray:
    """
    RBC mask matching normalize_tiles_new_2 (RGB ratios + darkness + nucleus safeguards).

    Returns boolean (H, W) where True = RBC pixel (exclude from maxC; preserve in output).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return np.zeros(img.shape[:2], dtype=bool)

    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)

    r_g_ratio = np.where(G > 0, R / G, np.inf)
    rbc_main = r_g_ratio > r_g_ratio_threshold

    if not np.any(rbc_main):
        return np.zeros(img.shape[:2], dtype=bool)

    r_b_ratio = np.where(B > 0, R / B, np.inf)
    r_dominance = R / (R + G + B + 1e-10)

    rbc_subsidiary = (r_b_ratio > r_b_ratio_threshold) | (r_dominance > r_dominance_threshold)
    rbc_mask = rbc_main & rbc_subsidiary

    if dark_threshold is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dark_mask = gray < dark_threshold
        rbc_mask = rbc_mask & dark_mask

    if np.any(rbc_mask):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        a_lab = lab[:, :, 1].astype(np.float64) - 128.0
        b_lab = lab[:, :, 2].astype(np.float64) - 128.0
        chroma = np.sqrt(a_lab * a_lab + b_lab * b_lab)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 0].astype(np.float64)

        blue_b = b_lab < -10.0
        purple_hue = (H >= 130.0) & (H <= 160.0)
        nucleus_chroma = chroma > 24.0
        purple_nucleus = blue_b & purple_hue & nucleus_chroma

        rbc_mask = rbc_mask & ~purple_nucleus

        if chroma_safeguard is not None and chroma_safeguard > 0:
            high_chroma = chroma > chroma_safeguard
            moderate_r_g = r_g_ratio < 2.4
            likely_nucleus = high_chroma & moderate_r_g
            rbc_mask = rbc_mask & ~likely_nucleus

    return rbc_mask


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
        help=f"Black artifact stage 1: grayscale (0-255), darker = candidate (default: {GRAYSCALE_DARK_THRESHOLD}).",
    )
    parser.add_argument(
        "--chroma-artifact-threshold",
        type=float,
        default=CHROMA_ARTIFACT_THRESHOLD,
        help=f"Black artifact stage 2: LAB chroma below this = artifact (default: {CHROMA_ARTIFACT_THRESHOLD}).",
    )
    parser.add_argument(
        "--v-artifact-threshold",
        type=float,
        default=V_ARTIFACT_THRESHOLD,
        help=f"Black artifact stage 2: HSV V (0-1) below this = artifact (default: {V_ARTIFACT_THRESHOLD}).",
    )
    parser.add_argument(
        "--rgb-std-artifact-threshold",
        type=float,
        default=RGB_STD_ARTIFACT_THRESHOLD,
        help=f"Black artifact stage 2: per-pixel RGB std below this = artifact (default: {RGB_STD_ARTIFACT_THRESHOLD}).",
    )
    parser.add_argument(
        "--black-artifact-max-area",
        type=int,
        default=0,
        metavar="N",
        help="Optional max connected-component area (pixels) for black artifacts; 0 = no limit (default: 0).",
    )
    parser.add_argument(
        "--disable-black-artifact-filter",
        action="store_true",
        help="Disable black artifact handling (all pixels affect maxC for that step).",
    )
    parser.add_argument(
        "--r-g-ratio-threshold",
        type=float,
        default=R_G_RATIO_THRESHOLD,
        help=f"RBC: R/G above this starts candidate (default: {R_G_RATIO_THRESHOLD}).",
    )
    parser.add_argument(
        "--r-b-ratio-threshold",
        type=float,
        default=R_B_RATIO_THRESHOLD,
        help=f"RBC: subsidiary R/B above this supports RBC (default: {R_B_RATIO_THRESHOLD}).",
    )
    parser.add_argument(
        "--r-dominance-threshold",
        type=float,
        default=R_DOMINANCE_THRESHOLD,
        help=f"RBC: subsidiary R/(R+G+B) above this supports RBC (default: {R_DOMINANCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--rbc-dark-threshold",
        type=int,
        default=RBC_DARK_THRESHOLD,
        metavar="N",
        help=(
            f"RBC: only flag pixels darker than this grayscale (0-255); "
            f"0 = no darkness gate (default: {RBC_DARK_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--rbc-chroma-safeguard",
        type=float,
        default=RBC_CHROMA_SAFEGUARD,
        help=(
            f"RBC: if chroma > this and R/G < 2.4, keep as nucleus (default: {RBC_CHROMA_SAFEGUARD}). "
            "Use 0 to disable this safeguard."
        ),
    )
    parser.add_argument(
        "--disable-rbc-filter",
        action="store_true",
        help="Disable RBC exclusion from maxC (RBC pixels normalized like other tissue).",
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

    use_black_artifact = BLACK_ARTIFACT_ENABLED and not args.disable_black_artifact_filter
    if use_black_artifact:
        max_area = args.black_artifact_max_area if args.black_artifact_max_area > 0 else None
        area_str = str(max_area) if max_area else "no limit"
        print(
            "Black artifact handling: enabled "
            f"(exclude from maxC, preserve original RGB; "
            f"gray<{args.grayscale_dark_threshold}, chroma<{args.chroma_artifact_threshold}, "
            f"V<{args.v_artifact_threshold}, rgb_std<{args.rgb_std_artifact_threshold}, "
            f"max_area={area_str})"
        )
    else:
        print("Black artifact handling: disabled")

    use_rbc = RBC_ENABLED and not args.disable_rbc_filter
    if use_rbc:
        dark_note = (
            f"dark<{args.rbc_dark_threshold}"
            if args.rbc_dark_threshold and args.rbc_dark_threshold > 0
            else "no dark gate"
        )
        chroma_note = (
            f", chroma_safe>{args.rbc_chroma_safeguard}"
            if args.rbc_chroma_safeguard and args.rbc_chroma_safeguard > 0
            else ", chroma_safe=off"
        )
        print(
            "RBC handling: enabled "
            f"(exclude from maxC, preserve original RGB; R/G>{args.r_g_ratio_threshold}, "
            f"R/B>{args.r_b_ratio_threshold} OR R_dom>{args.r_dominance_threshold}; "
            f"{dark_note}{chroma_note})"
        )
    else:
        print("RBC handling: disabled")

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

        preserve_mask = np.zeros(tile_rgb.shape[:2], dtype=bool)
        if use_black_artifact:
            max_area = args.black_artifact_max_area if args.black_artifact_max_area > 0 else None
            black_mask = detect_black_artifact_mask(
                tile_rgb,
                grayscale_threshold=args.grayscale_dark_threshold,
                chroma_threshold=args.chroma_artifact_threshold,
                v_threshold=args.v_artifact_threshold,
                rgb_std_threshold=args.rgb_std_artifact_threshold,
                max_area=max_area,
            )
            preserve_mask |= black_mask
            if np.any(black_mask):
                print(
                    f"    Black artifact pixels (excluded from maxC, preserved): {np.sum(black_mask)}"
                )

        if use_rbc:
            dark_thresh = (
                args.rbc_dark_threshold if args.rbc_dark_threshold and args.rbc_dark_threshold > 0 else None
            )
            chroma_safe = (
                args.rbc_chroma_safeguard
                if args.rbc_chroma_safeguard and args.rbc_chroma_safeguard > 0
                else None
            )
            rbc_mask = detect_rbc_mask(
                tile_rgb,
                r_g_ratio_threshold=args.r_g_ratio_threshold,
                r_b_ratio_threshold=args.r_b_ratio_threshold,
                r_dominance_threshold=args.r_dominance_threshold,
                dark_threshold=dark_thresh,
                chroma_safeguard=chroma_safe,
            )
            preserve_mask |= rbc_mask
            if np.any(rbc_mask):
                print(f"    RBC pixels (excluded from maxC, preserved): {np.sum(rbc_mask)}")

        tile_std = tissue_only_brightness_standardize(
            tile_rgb,
            luminance_percentile=LUMINANCE_PERCENTILE,
        )
        if np.any(preserve_mask):
            normed = normalizer.transform(
                tile_std,
                artifact_mask=preserve_mask,
                artifact_preserve_rgb=tile_rgb,
            )
        else:
            normed = normalizer.transform(tile_std)
        out_arr = normed.cpu().numpy() if hasattr(normed, "cpu") else normed
        out_arr = out_arr.astype(np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR))
        print(f"  {i + 1}/{len(tile_paths)}: {path.name} -> {out_path.name}")

    print(f"Done. Normalized {len(tile_paths)} tiles -> {output_dir}")


if __name__ == "__main__":
    main()
