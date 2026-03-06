#!/usr/bin/env bash
# Run stain normalization (normalize_tiles.py).
# Uses WSI feature stain normalization pipeline; no RBC or black-artifact filters.
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
