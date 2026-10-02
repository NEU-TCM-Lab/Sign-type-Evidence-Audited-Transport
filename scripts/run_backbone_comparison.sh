#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
declare -A TAGS=( [clip]=clipL336_seed42 [siglip2]=siglip2so400m384_seed42 [dinov2]=dinov2L_seed42 )
for BK in clip siglip2 dinov2; do
  TAG="${TAGS[$BK]}"
  if [[ ! -f "$FEATURES/train_pooled_${TAG}.pt" || ! -f "$FEATURES/val_pooled_${TAG}.pt" || ! -f "$FEATURES/test_pooled_${TAG}.pt" ]]; then
    "$PY" "$SRC/cache_backbone_features.py" --backbone "$BK" --tag "$TAG" --splits train val test --batch-size 16 > "$LOG/backbone_${TAG}.log" 2>&1
  fi
  "$PY" "$SRC/run_experiments.py" --tag "$TAG" --comparison-name "$TAG" --feature-modes global global_mask global_mask_bbox --seed 42 >> "$LOG/backbone_${TAG}.log" 2>&1
done
