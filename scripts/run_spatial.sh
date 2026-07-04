#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda/envs/vlm-vllm/bin/python
cd /root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls; mkdir -p artifacts/logs
LOG=artifacts/logs/spatial.log; TAG=dinov2grid_seed42
say(){ echo "===== $(date '+%F %T') $* =====" | tee -a "$LOG"; }
# save OT with spatial coherence (main); + softmax control
for LAM in 3 10 30 100; do
  say "ot + spatial lambda=$LAM on $TAG"
  $PY scripts/train_readout.py --tag "$TAG" --readout ot --spatial-lambda $LAM \
      --run-name "sp_ot_l${LAM}_${TAG}" >> "$LOG" 2>&1 || echo "[FAIL] ot l$LAM" | tee -a "$LOG"
done
for LAM in 10 30; do
  say "softmax + spatial lambda=$LAM on $TAG (control)"
  $PY scripts/train_readout.py --tag "$TAG" --readout softmax --spatial-lambda $LAM \
      --run-name "sp_sm_l${LAM}_${TAG}" >> "$LOG" 2>&1 || echo "[FAIL] sm l$LAM" | tee -a "$LOG"
done
say "SPATIAL DONE"
