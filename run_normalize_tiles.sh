#!/usr/bin/env bash
# Run stain normalization (normalize_tiles.py).
# WSI stain matrices + tissue-only brightness + Vahadane.
# Black-artifact handling is ON by default (exclude from maxC, preserve original RGB).
# Add --disable-black-artifact-filter to the python line for legacy all-pixels maxC.
# RBC removal is only in normalize_tiles_new_2.py.
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
  --output "$OUTPUT_DIR"
