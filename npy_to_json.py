"""
Convert TorchVahadane WSI feature .npy files to human-readable JSON.

This script expects per-WSI paired files:
- <stem>_stain_matrix.npy   (shape: 2x3)
- <stem>_maxCRef.npy        (shape: 2,)

It writes ONE combined JSON per WSI:
- <stem>.json

Example:

    python npy_to_json.py --input_dir wsi_features --output_dir wsi_features_json
"""

import argparse
import json
from pathlib import Path
import numpy as np


STAIN_SUFFIX = "_stain_matrix.npy"
MAXC_SUFFIX = "_maxCRef.npy"


def paired_npy_to_json(
    stain_path: Path, maxc_path: Path, output_path: Path
) -> None:
    """Load paired feature .npy files and save a combined JSON."""
    stain_matrix = np.load(stain_path)
    maxCRef = np.load(maxc_path)

    data = {
        "wsi_stem": stain_path.name[: -len(STAIN_SUFFIX)],
        "files": {
            "stain_matrix": stain_path.name,
            "maxCRef": maxc_path.name,
        },
        "stain_matrix": {
            "shape": list(stain_matrix.shape),
            "values": stain_matrix.tolist(),
        },
        "maxCRef": {
            "shape": list(maxCRef.shape),
            "values": maxCRef.tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert paired WSI feature .npy files to JSON.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="wsi_features",
        help="Directory containing .npy files (default: wsi_features).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wsi_features_json",
        help="Directory for output .json files (default: wsi_features_json).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    stain_files = sorted(input_dir.glob(f"*{STAIN_SUFFIX}"))
    if not stain_files:
        raise RuntimeError(f"No '{STAIN_SUFFIX}' files found in {input_dir}")

    converted = 0
    missing: list[str] = []
    for stain_path in stain_files:
        stem = stain_path.name[: -len(STAIN_SUFFIX)]
        maxc_path = input_dir / f"{stem}{MAXC_SUFFIX}"
        if not maxc_path.exists():
            missing.append(stem)
            continue

        json_path = output_dir / f"{stem}.json"
        paired_npy_to_json(stain_path, maxc_path, json_path)
        converted += 1
        print(f"Converted {stain_path.name} + {maxc_path.name} -> {json_path}")

    if missing:
        print(
            f"\nWarning: {len(missing)} WSI(s) missing '{MAXC_SUFFIX}'. Skipped."
        )

    if converted == 0:
        raise RuntimeError("No paired WSI feature files were converted.")


if __name__ == "__main__":
    main()
