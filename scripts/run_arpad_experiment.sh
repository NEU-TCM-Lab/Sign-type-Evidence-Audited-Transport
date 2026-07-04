#!/usr/bin/env bash
# ISOLATED EXPERIMENT: best full pipeline (coocc 74.37 + color fusion 74.67) on ASPECT-PRESERVING
# PAD geometry (dinov2grid_arpad / color_arpad), to test whether the square-resize squish mattered.
set -uo pipefail
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda/envs/vlm-vllm/bin/python
cd /root/autodl-tmp/TongueDx2_Dinov2_V14/scripts
OUT=/root/autodl-tmp/TongueDx2_Dinov2_V14/outputs
LOG=/root/autodl-tmp/TongueDx2_Dinov2_V14/artifacts/logs
mkdir -p "$OUT" "$LOG"

echo "===== [1/3] co-occurrence model (arpad) $(date +%T) ====="
$PY train_readout.py --tag dinov2grid_arpad_seed42 --readout softmax --num-heads 8 --seed 42 \
  --epochs 120 --patience 15 --selection-metric macro_f1 --threshold-step 0.01 \
  --loss bce --pos-weight --coocc-lambda 0.32 --coocc-eps 0.08 \
  --dump-teacher "$OUT/arpad_coocc_seed42.npz" \
  --run-name arpad_coocc --experiment-name arpad_coocc > "$LOG/arpad_coocc.log" 2>&1
grep -aE "^\[TEST|macro_f1" "$LOG/arpad_coocc.log" | tail -2

echo "===== [2/3] color OT detector (arpad) $(date +%T) ====="
$PY train_readout.py --tag color_arpad_seed42 --readout otgen --seed 42 \
  --epochs 120 --patience 20 --selection-metric rare2 --threshold-step 0.01 \
  --loss bce --pos-weight --ot-k 8 --otgen-combine concat --peak-topk 16 --eps-init 0.1 --lam-init 0.05 \
  --dump-teacher "$OUT/arpad_color_seed42.npz" \
  --run-name arpad_color --experiment-name arpad_color > "$LOG/arpad_color.log" 2>&1
grep -aE "^\[TEST|rare2" "$LOG/arpad_color.log" | tail -2

echo "===== [3/3] per-class gated late fusion (TonguePale only) $(date +%T) ====="
$PY fuse_npz.py --base "$OUT/arpad_coocc_seed42.npz" --add "$OUT/arpad_color_seed42.npz" \
  --fuse-classes TonguePale | tee "$OUT/arpad_fusion_result.txt"
echo "===== arpad experiment DONE $(date +%T) ====="
