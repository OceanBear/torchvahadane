"""
Extract robust stain features from Whole Slide Images (WSIs) using TorchVahadane.

For each WSI in an input directory, this script:
- estimates the median stain intensity matrix via `estimate_median_matrix`
- saves the resulting stain matrix as a NumPy `.npy` file under `wsi_features/`

Example (WSL path converted from `D:\\BCCRC-work\\NucSegAI\\sample_wsi`),
running entirely on CPU:

    python extract_wsi_features.py \
        --wsi_dir /mnt/d/BCCRC-work/NucSegAI/sample_wsi \
        --output_dir wsi_features
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import openslide  # type: ignore

from torchvahadane import TorchVahadaneNormalizer
from torchvahadane.wsi_util import estimate_median_matrix


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


def extract_stain_feature_for_wsi(
    wsi_path: Path,
    output_dir: Path,
    device: str = "cuda",
    osh_level: int = 0,
    tile_size: int = 4096,
    num_workers: int = 4,
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing WSI: {wsi_path}")
    osh = openslide.open_slide(str(wsi_path))
    level_dims = osh.level_dimensions
    print(f"  Number of levels: {osh.level_count}")
    print(f"  Level 0 size (W x H): {level_dims[0]}")
    print(f"  Using openslide level for estimation: {osh_level}")
    print(f"  Tile size: {tile_size}, num_workers: {num_workers}")
    print(f"  TorchVahadane device: {device}")

    if device.startswith("cuda"):
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    t0 = time.perf_counter()
    normalizer = TorchVahadaneNormalizer(device=device)
    stain_matrix = estimate_median_matrix(
        osh,
        normalizer,
        osh_level=osh_level,
        tile_size=tile_size,
        num_workers=num_workers,
    )
    t1 = time.perf_counter()

    feature_filename = wsi_path.stem + "_stain_matrix.npy"
    out_path = output_dir / feature_filename
    np.save(out_path, stain_matrix)

    print(f"  Estimation time: {t1 - t0:.2f} s")
    print(f"  Saved stain matrix to: {out_path} (shape={stain_matrix.shape})")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract robust stain features (median stain matrix) for WSIs using TorchVahadane."
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=False,
        default="/mnt/d/BCCRC-work/NucSegAI/sample_wsi",
        help=(
            "Directory containing WSI files. "
            "For the sample path 'D:\\\\BCCRC-work\\\\NucSegAI\\\\sample_wsi' on Windows, "
            "the corresponding WSL path is '/mnt/d/BCCRC-work/NucSegAI/sample_wsi'."
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
        required=False,
        default=0,
        help="Openslide level to use for stain estimation (default: 0).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        required=False,
        default=4096,
        help="Tile size used in the grid search for stain estimation (default: 4096).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Number of workers used for parallel processing (default: 4).",
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
        # Skip if this WSI already has a saved stain matrix
        feature_filename = wsi_path.stem + "_stain_matrix.npy"
        existing_path = output_dir / feature_filename
        if existing_path.exists():
            print(
                f"\nSkipping WSI (already processed): {wsi_path}\n"
                f"  Existing feature file: {existing_path}"
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
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to process {wsi_path}: {exc}")


if __name__ == "__main__":
    main()

