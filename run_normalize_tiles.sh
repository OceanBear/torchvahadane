#!/usr/bin/env bash
# Run stain normalization (normalize_tiles.py).
# WSI stain matrices + tissue-only brightness + Vahadane.
#
# Filters (both ON by default in normalize_tiles.py): pixels matching the mask are
# excluded from Vahadane maxC scaling and keep original RGB in the output.
#   --disable-black-artifact-filter   turn off dark achromatic artifact handling
#   --disable-rbc-filter              turn off RBC handling
#
# Example RBC tuning (same knobs as normalize_tiles_new_2.py):
#   --rbc-dark-threshold 100 --rbc-chroma-safeguard 55
#
# Input:  /scratch/st-kenfield-1/repos/NucSegAI/sample_images2
# Output: /scratch/st-kenfield-1/repos/NucSegAI/std_output4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/scratch/st-kenfield-1/repos/NucSegAI/sample_images2"
OUTPUT_DIR="/scratch/st-kenfield-1/repos/NucSegAI/std_output4"

cd "$SCRIPT_DIR"
python normalize_tiles.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --rbc-dark-threshold 100 \
  --rbc-chroma-safeguard 55
