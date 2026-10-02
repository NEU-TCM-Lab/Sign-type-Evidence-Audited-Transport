#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# Build actual aspect-preserving padded grids; a filename tag alone does not change geometry.
for BK in dinov2 color; do
  if [[ "$BK" == dinov2 ]]; then
    "$PY" "$SRC/cache_patchgrid_features.py" --backbone dinov2 --tag dinov2grid_arpad_seed42 --geometry arpad
  else
    "$PY" "$SRC/cache_color_features.py" --tag color_arpad_seed42 --geometry arpad
  fi
done
"$PY" "$SRC/train_readout.py" --tag dinov2grid_arpad_seed42 --readout softmax --num-heads 8 --seed 42 --epochs 120 --patience 15 --selection-metric macro_f1 --threshold-step 0.01 --loss bce --pos-weight --coocc-lambda 0.32 --coocc-eps 0.08 --dump-teacher "$OUT/arpad_coocc_seed42.npz" --run-name arpad_coocc --experiment-name arpad_coocc > "$LOG/arpad_coocc.log" 2>&1
"$PY" "$SRC/train_readout.py" --tag color_arpad_seed42 --readout otgen --seed 42 --epochs 120 --patience 20 --selection-metric rare2 --threshold-step 0.01 --loss bce --pos-weight --ot-k 8 --otgen-combine concat --peak-topk 16 --eps-init 0.1 --lam-init 0.05 --dump-teacher "$OUT/arpad_color_seed42.npz" --run-name arpad_color --experiment-name arpad_color > "$LOG/arpad_color.log" 2>&1
"$PY" "$SRC/fuse_npz.py" --base "$OUT/arpad_coocc_seed42.npz" --add "$OUT/arpad_color_seed42.npz" --fuse-classes TonguePale --out-json "$OUT/arpad_fusion_result.json"
