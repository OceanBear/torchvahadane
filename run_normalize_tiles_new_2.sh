#!/usr/bin/env bash
# Run stain normalization with NucSegAI sample images.
# Input:  /scratch/st-kenfield-1/repos/NucSegAI/sample_images
# Output: /scratch/st-kenfield-1/repos/NucSegAI/std_output
#
# RBC filter: removes dark RBCs (dark<100) with chroma safeguard to avoid removing purple nuclei.
# Use --disable-rbc-filter or --disable-black-artifact-filter to turn filters off.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/scratch/st-kenfield-1/repos/NucSegAI/sample_images2"
OUTPUT_DIR="/scratch/st-kenfield-1/repos/NucSegAI/std_output4"

cd "$SCRIPT_DIR"
python normalize_tiles_new_2.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --rbc-dark-threshold 100 \
  --rbc-chroma-safeguard 55