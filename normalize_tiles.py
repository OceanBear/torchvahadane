#!/usr/bin/env python3
"""
Stain-normalize H&E tiles using a reference image.
Reference: ref_image/
Raw tiles: /mnt/d/Downloads/Programs/original_tiles (WSL path for D:\Downloads\Programs\original_tiles)
Output: saved to an output directory (default: normalized_tiles in project, or under the same WSL tree).
"""

from pathlib import Path

import cv2
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

# Image extensions to consider
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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
        import numpy as np
        pil_img = np.array(Image.open(ref_path))
        if pil_img.ndim == 2:
            ref = cv2.cvtColor(pil_img, cv2.COLOR_GRAY2RGB)
        elif pil_img.shape[2] == 4:
            ref = cv2.cvtColor(pil_img, cv2.COLOR_RGBA2RGB)
        else:
            ref = pil_img
    else:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    normalizer = TorchVahadaneNormalizer(device=device, staintools_estimate=STAINTOOLS_ESTIMATE, correct_exposure=True)
    normalizer.fit(ref)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    for i, path in enumerate(tile_paths):
        tile = cv2.imread(str(path))
        if tile is None:
            print(f"  Skip (unreadable): {path.name}")
            continue
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        normed = normalizer.transform(tile_rgb)
        out_arr = normed.cpu().numpy() if hasattr(normed, "cpu") else normed
        out_path = OUTPUT_DIR / path.name
        cv2.imwrite(str(out_path), cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR))
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {i + 1}/{len(tile_paths)}: {path.name} -> {out_path.name}")

    print(f"Done. Normalized {len(tile_paths)} tiles -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
