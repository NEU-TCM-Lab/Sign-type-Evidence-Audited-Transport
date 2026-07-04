#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda/envs/vlm-vllm/bin/python
cd /root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls; mkdir -p artifacts/logs
LOG=artifacts/logs/coocc.log; TAG=dinov2grid_seed42
say(){ echo "===== $(date '+%F %T') $* =====" | tee -a "$LOG"; }
# label-level co-occurrence OT regularizer on ot-readout (main) + softmax control
for LAM in 0.1 0.3 1.0 3.0; do
  say "ot + coocc-OT lambda=$LAM on $TAG"
  $PY scripts/train_readout.py --tag "$TAG" --readout ot --coocc-lambda $LAM \
      --run-name "co_ot_l${LAM}_${TAG}" >> "$LOG" 2>&1 || echo "[FAIL] ot l$LAM" | tee -a "$LOG"
done
for LAM in 0.3 1.0; do
  say "softmax + coocc-OT lambda=$LAM on $TAG (control)"
  $PY scripts/train_readout.py --tag "$TAG" --readout softmax --coocc-lambda $LAM \
      --run-name "co_sm_l${LAM}_${TAG}" >> "$LOG" 2>&1 || echo "[FAIL] sm l$LAM" | tee -a "$LOG"
done
say "COOCC DONE"
