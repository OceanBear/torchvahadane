"""
Extract robust stain features from Whole Slide Images (WSIs) using TorchVahadane.

For each WSI in an input directory, this script:
- estimates the median stain intensity matrix and max concentrations (maxCRef) via `estimate_median_matrix_and_maxC`
- saves the stain matrix and maxCRef as NumPy `.npy` files under `wsi_features/`

Example (WSL path converted from `F:\\projects\\Pathology\\HandE\\SOW1885_n=201_AT2 40X`),
running entirely on CPU:

    python extract_wsi_features.py \
        --wsi_dir "/mnt/f/projects/Pathology/HandE/SOW1885_n=201_AT2 40X" \
        --output_dir wsi_features
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import openslide  # type: ignore

from torchvahadane import TorchVahadaneNormalizer
from torchvahadane.wsi_util import estimate_median_matrix_and_maxC


WSI_EXTENSIONS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".bif",
)


def find_wsi_files(wsi_dir: Path) -> list[Path]:
    """Return a list of WSI files under `wsi_dir` with known extensions."""
    files: list[Path] = []
    for p in sorted(wsi_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in WSI_EXTENSIONS:
            files.append(p)
    return files


def _patch_luminosity_threshold(normalizer, luminosity_threshold: float) -> None:
    """
    Patch the normalizer's stain extractor to use a custom luminosity threshold
    for tissue detection. Does not modify stain_extractor_cpu/gpu source files.
    """
    extractor = normalizer.stain_extractor
    orig_get_tissue_mask = extractor.get_tissue_mask
    orig_get_stain_matrix = extractor.get_stain_matrix

    def patched_get_tissue_mask(I, *args, **kwargs):
        kwargs["luminosity_threshold"] = luminosity_threshold
        return orig_get_tissue_mask(I, *args, **kwargs)

    def patched_get_stain_matrix(I, *args, **kwargs):
        kwargs["luminosity_threshold"] = luminosity_threshold
        return orig_get_stain_matrix(I, *args, **kwargs)

    extractor.get_tissue_mask = patched_get_tissue_mask
    extractor.get_stain_matrix = patched_get_stain_matrix


def extract_stain_feature_for_wsi(
    wsi_path: Path,
    output_dir: Path,
    device: str = "cuda",
    osh_level: int = 0,
    tile_size: int = 4096,
    num_workers: int = 4,
    subsample_fraction: float = 1.0,
    subsample_seed: int = 42,
    luminosity_threshold: float = 0.9,
    max_tiles: int | None = 80,
) -> Path:
    """
    Estimate the median stain matrix for a single WSI and save it as `.npy`.

    Parameters
    ----------
    wsi_path : Path
        Path to the WSI file.
    output_dir : Path
        Directory where the feature file will be written.
    device : str
        Device passed to `TorchVahadaneNormalizer` (e.g. "cuda" or "cpu").
    osh_level : int
        Openslide level to use for tile extraction.
    tile_size : int
        Tile size for estimation.
    num_workers : int
        Number of processes/threads used in `estimate_median_matrix`.
    luminosity_threshold : float
        Luminosity threshold for tissue mask (L < threshold = tissue). Default 0.9.
        Higher values include brighter regions as tissue (0.8 is the stain extractor default).
    max_tiles : int, optional
        Maximum number of tiles to use. If subsampled count exceeds this, keep only max_tiles.
        Default 80 for tile_size=4096. None = no limit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing WSI: {wsi_path}")
    osh = openslide.open_slide(str(wsi_path))
    level_dims = osh.level_dimensions
    print(f"  Number of levels: {osh.level_count}")
    print(f"  Level 0 size (W x H): {level_dims[0]}")
    print(f"  Using openslide level for estimation: {osh_level}")
    print(f"  Tile size: {tile_size}, num_workers: {num_workers}")
    if subsample_fraction < 1.0:
        print(f"  Subsampling: keep {subsample_fraction*100:.0f}% of tissue tiles (seed={subsample_seed})")
    if max_tiles is not None:
        print(f"  Max tiles cap: {max_tiles}")
    print(f"  Luminosity threshold (tissue): {luminosity_threshold}")
    print(f"  TorchVahadane device: {device}")

    if device.startswith("cuda"):
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    t0 = time.perf_counter()
    normalizer = TorchVahadaneNormalizer(device=device)
    _patch_luminosity_threshold(normalizer, luminosity_threshold)
    stain_matrix, maxCRef, tile_info = estimate_median_matrix_and_maxC(
        osh,
        normalizer,
        osh_level=osh_level,
        tile_size=tile_size,
        num_workers=num_workers,
        subsample_fraction=subsample_fraction,
        subsample_seed=subsample_seed,
        max_tiles=max_tiles,
    )
    t1 = time.perf_counter()

    print(f"  Tiles after tissue filter (≥50%): {tile_info['n_after_tissue']}")
    print(f"  Tiles after subsample: {tile_info['n_after_subsample']}")
    print(f"  Tiles used: {tile_info['n_final']}")

    stem = wsi_path.stem
    stain_path = output_dir / (stem + "_stain_matrix.npy")
    maxC_path = output_dir / (stem + "_maxCRef.npy")
    np.save(stain_path, stain_matrix)
    np.save(maxC_path, maxCRef)

    print(f"  Estimation time: {t1 - t0:.2f} s")
    print(f"  Saved stain matrix to: {stain_path} (shape={stain_matrix.shape})")
    print(f"  Saved maxCRef to: {maxC_path} (shape={maxCRef.shape})")
    return stain_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract robust stain features (median stain matrix) for WSIs using TorchVahadane."
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=False,
        default="/mnt/f/projects/Pathology/HandE/SOW1885_n=201_AT2 40X",    # "/mnt/f/projects/Pathology/HandE/SOW1885_n=201_AT2 40X"
        help=(
            "Directory containing WSI files. "
            "For the sample path 'F:\\\\projects\\\\Pathology\\\\HandE\\\\SOW1885_n=201_AT2 40X' on Windows, "
            "the corresponding WSL path is '/mnt/f/projects/Pathology/HandE/SOW1885_n=201_AT2 40X'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="wsi_features",
        help="Directory to store the extracted WSI feature files (default: wsi_features).",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help='Device for TorchVahadaneNormalizer (e.g. "cpu" or "cuda"). Default: "cpu".',
    )
    parser.add_argument(
        "--osh_level",
        type=int,
        required=False, # resolution level, 0 for full resolution
        default=0,
        help="Openslide level to use for stain estimation (default: 0).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        required=False,
        default=4096,   # was 4096
        help="Tile size used in the grid search for stain estimation (default: 4096).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers used for parallel processing (default: 4).",
    )
    parser.add_argument(
        "--subsample_fraction",
        type=float,
        required=False,
        default=0.3,    # was 0.5
        help="Fraction of tissue-passing tiles to keep, e.g. 0.5 for 50pct, 0.3 for 30pct (default: 1.0 = no subsampling).",
    )
    parser.add_argument(
        "--subsample_seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for subsampling (default: 42).",
    )
    parser.add_argument(
        "--luminosity_threshold",
        type=float,
        required=False,
        default=0.9,
        help="Luminosity threshold for tissue mask: L < threshold = tissue (default: 0.9). Higher values include brighter regions. The stain extractor default is 0.8.",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        required=False,
        default=80,
        help="Maximum number of tiles to use. If subsampled count exceeds this, keep only max_tiles (default: 80). Use 0 for no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    wsi_dir = Path(args.wsi_dir)
    output_dir = Path(args.output_dir)

    if not wsi_dir.exists() or not wsi_dir.is_dir():
        raise FileNotFoundError(f"WSI directory does not exist or is not a directory: {wsi_dir}")

    wsi_files = find_wsi_files(wsi_dir)
    if not wsi_files:
        raise RuntimeError(
            f"No WSI files with extensions {WSI_EXTENSIONS} found in directory: {wsi_dir}"
        )

    print(f"Found {len(wsi_files)} WSI file(s) in {wsi_dir}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Requested TorchVahadane device: {args.device}")
    if args.device.startswith("cuda"):
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    for wsi_path in wsi_files:
        # Skip if this WSI already has saved stain matrix and maxCRef
        stem = wsi_path.stem
        stain_path = output_dir / (stem + "_stain_matrix.npy")
        maxC_path = output_dir / (stem + "_maxCRef.npy")
        if stain_path.exists() and maxC_path.exists():
            print(
                f"\nSkipping WSI (already processed): {wsi_path}\n"
                f"  Existing: {stain_path.name}, {maxC_path.name}"
            )
            continue

        try:
            extract_stain_feature_for_wsi(
                wsi_path=wsi_path,
                output_dir=output_dir,
                device=args.device,
                osh_level=args.osh_level,
                tile_size=args.tile_size,
                num_workers=args.num_workers,
                subsample_fraction=args.subsample_fraction,
                subsample_seed=args.subsample_seed,
                luminosity_threshold=args.luminosity_threshold,
                max_tiles=args.max_tiles if args.max_tiles > 0 else None,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to process {wsi_path}: {exc}")


if __name__ == "__main__":
    main()

