#!/usr/bin/env bash
set -euo pipefail

Edit these paths for your workspace.
CSV_FILE="../UCF-EL-Defect/AnnotationsCombined.csv"
IMG_DIR="../UCF-EL-Defect/<source-image-directory>"
INPAINT_ANYTHING_DIR="../Inpaint-Anything"
LAMA_CONFIG="../Inpaint-Anything/lama/configs/prediction/default.yaml"
LAMA_CKPT="../Inpaint-Anything/pretrained_models/big-lama"
OUTPUT_DIR="./generated_gdi"

# Paper settings. Both can be changed.
AREA_THRESHOLD=10
DILATE_KERNEL_SIZE=15

python generate_gdi.py \
  --csv-file "$CSV_FILE" \
  --img-dir "$IMG_DIR" \
  --inpaint-anything-dir "$INPAINT_ANYTHING_DIR" \
  --lama-config "$LAMA_CONFIG" \
  --lama-ckpt "$LAMA_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --area-threshold "$AREA_THRESHOLD" \
  --dilate-kernel-size "$DILATE_KERNEL_SIZE" \
  --device cuda \
  --visualizations-dir ./gdi_visualizations
