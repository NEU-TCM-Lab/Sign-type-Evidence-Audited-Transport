#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
TAG=dinov2grid_seed42
for LAM in 3 10 30 100; do
  "$PY" "$SRC/train_readout.py" --tag "$TAG" --readout ot --spatial-lambda "$LAM" --run-name "sp_ot_l${LAM}_${TAG}" > "$LOG/sp_ot_l${LAM}.log" 2>&1
done
for LAM in 10 30; do
  "$PY" "$SRC/train_readout.py" --tag "$TAG" --readout softmax --spatial-lambda "$LAM" --run-name "sp_sm_l${LAM}_${TAG}" > "$LOG/sp_sm_l${LAM}.log" 2>&1
done
