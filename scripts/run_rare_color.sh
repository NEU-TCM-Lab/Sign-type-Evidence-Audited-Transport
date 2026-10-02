#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ ! -f "$FEATURES/train_patchgrid_color_seed42.pt" || ! -f "$FEATURES/val_patchgrid_color_seed42.pt" || ! -f "$FEATURES/test_patchgrid_color_seed42.pt" ]]; then
  "$PY" "$SRC/cache_color_features.py" --splits train val test
fi
COMMON=(--readout otgen --seed 42 --epochs 120 --patience 20 --selection-metric rare2 --threshold-step 0.01 --loss bce --pos-weight --ot-k 8 --otgen-combine concat --peak-topk 16 --eps-init 0.1 --lam-init 0.05)
"$PY" "$SRC/train_readout.py" "${COMMON[@]}" --tag color_seed42 --run-name rarecolor_coloronly --experiment-name rarecolor_coloronly --per-class-csv "$OUT/per_class_results_rarecolor.csv" > "$LOG/rarecolor_coloronly.log" 2>&1
"$PY" "$SRC/train_readout.py" "${COMMON[@]}" --tag dinov2grid_seed42 --extra-tag color_seed42 --run-name rarecolor_dinocolor --experiment-name rarecolor_dinocolor --per-class-csv "$OUT/per_class_results_rarecolor.csv" > "$LOG/rarecolor_dinocolor.log" 2>&1
"$PY" "$SRC/collect_matrix.py" --runs-glob 'rarecolor_*' --out "$OUT/rarecolor_summary.csv"
