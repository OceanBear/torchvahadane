#!/usr/bin/env bash
# Run stain normalization with NucSegAI sample images.
# Input:  /scratch/st-kenfield-1/repos/NucSegAI/sample_images
# Output: /scratch/st-kenfield-1/repos/NucSegAI/std_output2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/scratch/st-kenfield-1/repos/NucSegAI/sample_images"
OUTPUT_DIR="/scratch/st-kenfield-1/repos/NucSegAI/std_output2"

cd "$SCRIPT_DIR"
python normalize_tiles_new_2.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR"
