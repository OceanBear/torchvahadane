#!/usr/bin/env bash
# Run stain normalization with NucSegAI sample images (local/WSL paths).
# Input:  /mnt/j/HandE/.../original_tiles
# Output: /mnt/j/HandE/.../SCN_torch_v3
#
# RBC filter: removes dark RBCs (dark<100) with chroma safeguard to avoid removing purple nuclei.
# Use --disable-rbc-filter or --disable-black-artifact-filter to turn filters off.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_test/original_tiles"
OUTPUT_DIR="/mnt/j/HandE/results/SOW1885_n=201_AT2 40X/JN_TS_test/SCN_torch_v3"

cd "$SCRIPT_DIR"
python normalize_tiles_new_2.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --rbc-dark-threshold 100 \
  --rbc-chroma-safeguard 55
