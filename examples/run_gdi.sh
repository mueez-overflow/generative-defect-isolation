#!/usr/bin/env bash
set -euo pipefail

python generate_gdi.py \
  --csv-file ../UCF-EL-Defect/AnnotationsCombined.csv \
  --img-dir ../UCF-EL-Defect/<source-image-directory> \
  --inpaint-anything-dir ../Inpaint-Anything \
  --lama-config ../Inpaint-Anything/lama/configs/prediction/default.yaml \
  --lama-ckpt ../Inpaint-Anything/pretrained_models/big-lama \
  --output-dir ./generated_gdi \
  --visualizations-dir ./gdi_visualizations \
  --area-threshold 10 \
  --dilate-kernel-size 15 \
  --device cuda
