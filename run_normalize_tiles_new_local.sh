#!/usr/bin/env bash
# Run stain normalization with NucSegAI sample images (local/WSL paths).
# Input:  /mnt/j/HandE/.../original_tiles
# Output: /mnt/j/HandE/.../SCN_torch_v3
#
# RBC filter: removes dark RBCs (dark<100) with chroma safeguard to avoid removing purple nuclei.
# Use --disable-rbc-filter or --disable-black-artifact-filter to turn filters off.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/mnt/d/Downloads/Compressed/compath-master/latticea_test_data/imgs_tiff"
OUTPUT_DIR="/mnt/d/Downloads/Compressed/compath-master/latticea_test_data/imgs_tiff_scn"

cd "$SCRIPT_DIR"
python normalize_tiles.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --rbc-dark-threshold 100 \
  --rbc-chroma-safeguard 55 \
  --disable-rbc-filter