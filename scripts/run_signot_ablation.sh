#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
TAG=dinov2grid_seed42
COMMON=(--tag "$TAG" --seed 42 --epochs 120 --patience 15 --selection-metric macro_f1 --threshold-step 0.01 --loss bce --pos-weight --ot-k 4 --beta-init 0.1 --per-class-csv "$OUT/per_class_results_signot.csv")
run() {
  local NAME="$1"; shift
  "$PY" "$SRC/train_readout.py" "${COMMON[@]}" --run-name "signot_${NAME}" --experiment-name "$NAME" "$@" > "$LOG/signot_${NAME}.log" 2>&1
}
run global_only --readout softmax
run global_branch --readout signot --branch global
run ot_only --readout signot --branch ot --evidence ot --use-mask-mass
run global_attn --readout signot --branch both --evidence attention
run global_ot_joint --readout signot --branch both --evidence ot
run global_ot_frozen --readout signot --branch both --evidence ot --init-global signot_global_branch --freeze-global
run global_ot_pdiv_joint --readout signot --branch both --evidence ot --proto-div-lambda 0.01
run global_ot_pdiv_frozen --readout signot --branch both --evidence ot --proto-div-lambda 0.01 --init-global signot_global_branch --freeze-global
run global_ot_mask_joint --readout signot --branch both --evidence ot --use-mask-mass
run global_ot_mask_frozen --readout signot --branch both --evidence ot --use-mask-mass --init-global signot_global_branch --freeze-global
"$PY" "$SRC/collect_matrix.py" --runs-glob 'signot_*' --out "$OUT/signot_matrix_summary.csv"
