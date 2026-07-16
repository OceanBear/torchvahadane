#!/usr/bin/env bash
# Run stain normalization with per-tile stain estimation only (no wsi_features).
# Input:  /mnt/d/BCCRC-work/MoNuSeg-Selected-LUDA/tiff
# Output: /mnt/d/BCCRC-work/MoNuSeg-Selected-LUDA/tiff_scn
#
# RBC filter: removes dark RBCs (dark<100) with chroma safeguard to avoid removing purple nuclei.
# Use --disable-rbc-filter or --disable-black-artifact-filter to turn maxC/output handling off.
#
# Per-tile stain estimation (when no WSI stain matrix): normalize_tiles.py can also exclude
# artifact/RBC pixels from dictionary learning; that is separate from maxC. Pairing:
#   --disable-rbc-filter  ->  add --disable-stain-est-rbc-exclusion for consistent "RBC off"
#   --disable-black-artifact-filter  ->  add --disable-stain-est-artifact-exclusion if desired
# default thresholds:
#  --grayscale-dark-threshold 35
#  --chroma-artifact-threshold 20
#  --v-artifact-threshold 0.20
#  --rgb-std-artifact-threshold 12
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/mnt/d/BCCRC-work/new_training/JN_TS_batch3"
OUTPUT_DIR="/mnt/d/BCCRC-work/new_training/scn"

cd "$SCRIPT_DIR"
python normalize_tiles.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --no-wsi-features \
  --rbc-dark-threshold 100 \
  --rbc-chroma-safeguard 55 \
  --disable-rbc-filter