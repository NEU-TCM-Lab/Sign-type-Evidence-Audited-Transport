#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
TAG=dinov2grid_seed42
for LAM in 0.1 0.3 1.0 3.0; do
  "$PY" "$SRC/train_readout.py" --tag "$TAG" --readout ot --coocc-lambda "$LAM" --run-name "co_ot_l${LAM}_${TAG}" > "$LOG/co_ot_l${LAM}.log" 2>&1
done
for LAM in 0.3 1.0; do
  "$PY" "$SRC/train_readout.py" --tag "$TAG" --readout softmax --coocc-lambda "$LAM" --run-name "co_sm_l${LAM}_${TAG}" > "$LOG/co_sm_l${LAM}.log" 2>&1
done
