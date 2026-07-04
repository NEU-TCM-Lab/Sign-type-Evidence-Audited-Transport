#!/usr/bin/env bash
set -euo pipefail
# Backbone comparison vs Qwen3-VL maskpool: for each of CLIP / SigLIP2 / DINOv2, cache
# frozen pooled features (full train/val/test, same manifest) then train+eval the SAME
# MaskPoolBBoxHead over {global, global_mask, global_mask_bbox}. Runs serially (GPU is free;
# feature extraction is light). Init from project root so `from common import` resolves.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda/envs/vlm-vllm/bin/python
PROJ=/root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls
cd "$PROJ"
mkdir -p artifacts/logs

# backbone : tag
declare -A TAGS=( [clip]=clipL336_seed42 [siglip2]=siglip2so400m384_seed42 [dinov2]=dinov2L_seed42 )

for BK in clip siglip2 dinov2; do
  TAG="${TAGS[$BK]}"
  LOG="$PROJ/artifacts/logs/backbone_${TAG}.log"
  echo "===== $(date '+%F %T') backbone=$BK tag=$TAG =====" | tee -a "$LOG"

  # 1) cache features (skip if test cache already present)
  if [[ -f "$PROJ/artifacts/features/test_pooled_${TAG}.pt" ]]; then
    echo "[skip cache] features exist for $TAG" | tee -a "$LOG"
  else
    $PY scripts/cache_backbone_features.py --backbone "$BK" --tag "$TAG" \
        --splits train val test --batch-size 16 >> "$LOG" 2>&1
  fi

  # 2) train+eval head over the three feature modes
  $PY scripts/run_experiments.py --tag "$TAG" --comparison-name "$TAG" \
      --feature-modes global global_mask global_mask_bbox --seed 42 >> "$LOG" 2>&1

  echo "[done] $BK -> artifacts/comparisons/$TAG/headline_metrics.json" | tee -a "$LOG"
done

echo "===== ALL BACKBONES DONE $(date '+%F %T') ====="
