#!/usr/bin/env bash
# Run stain normalization with NucSegAI sample images (local/WSL paths).
# Input:  /mnt/j/HandE/.../original_tiles
# Output: /mnt/j/HandE/.../SCN_torch_v3
#
# RBC filter: removes dark RBCs (dark<100) with chroma safeguard to avoid removing purple nuclei.
# Use --disable-rbc-filter or --disable-black-artifact-filter to turn maxC/output handling off.
#
# Per-tile stain estimation (when no WSI stain matrix): normalize_tiles.py can also exclude
# artifact/RBC pixels from dictionary learning; that is separate from maxC. Pairing:
#   --disable-rbc-filter  ->  add --disable-stain-est-rbc-exclusion for consistent "RBC off"
#   --disable-black-artifact-filter  ->  add --disable-stain-est-artifact-exclusion if desired

set -euo pipefail

SCRIPT_DIR="/scratch/st-kenfield-1/repos/torchvahadane"
INPUT_DIR="/scratch/st-kenfield-1/repos/torchvahadane/LATTICe-A/imgs_mpp025_tiff"
OUTPUT_DIR="/scratch/st-kenfield-1/repos/torchvahadane/LATTICe-A/imgs_mpp025_scn"

cd "$SCRIPT_DIR"
python normalize_tiles.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --rbc-dark-threshold 100 \
  --rbc-chroma-safeguard 55 \
  --disable-rbc-filter