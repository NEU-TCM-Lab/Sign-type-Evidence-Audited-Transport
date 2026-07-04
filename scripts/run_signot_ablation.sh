#!/usr/bin/env bash
# Sign-Prototype OT ablation on DINOv2 patch-grid. Each readout head trains in minutes.
# Ablations (cols 1-7) + frozen-vs-joint training-mode axis for the OT configs.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda/envs/vlm-vllm/bin/python
PROJ=/root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls
cd "$PROJ/scripts"
TAG=dinov2grid_seed42
CSV="$PROJ/outputs/per_class_results_signot.csv"
LOGDIR="$PROJ/artifacts/logs"; mkdir -p "$PROJ/outputs" "$LOGDIR"; rm -f "$CSV"
rm -rf "$PROJ"/artifacts/runs/signot_*    # train_readout mkdir(exist_ok=False)
COMMON="--tag $TAG --seed 42 --epochs 120 --patience 15 --selection-metric macro_f1 \
  --threshold-step 0.01 --loss bce --pos-weight --per-class-csv $CSV --ot-k 4 --beta-init 0.1"

run () { local NAME="$1"; shift
  echo ">>> $NAME"
  $PY train_readout.py $COMMON --run-name "signot_${NAME}" --experiment-name "$NAME" "$@" \
      > "$LOGDIR/signot_${NAME}.log" 2>&1 || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/signot_${NAME}.log"; return; }
  grep -aE "^\[TEST|test_calibrated|MacroF1" "$LOGDIR/signot_${NAME}.log" | tail -2
}

# (1) Global only — the real softmax+BCE baseline (73.45)
run global_only        --readout softmax
# (2) signot --branch global — global-branch checkpoint (for --init-global) + sanity (~73.45)
run global_branch      --readout signot --branch global
# (3) OT only
run ot_only            --readout signot --branch ot --evidence ot --use-mask-mass
# (4) Global + Attention pooling (vs OT)
run global_attn        --readout signot --branch both --evidence attention
# (5) Global + OT  (joint + frozen)
run global_ot_joint    --readout signot --branch both --evidence ot
run global_ot_frozen   --readout signot --branch both --evidence ot --init-global signot_global_branch --freeze-global
# (6) Global + OT + prototype diversity (joint + frozen)
run global_ot_pdiv_joint   --readout signot --branch both --evidence ot --proto-div-lambda 0.01
run global_ot_pdiv_frozen  --readout signot --branch both --evidence ot --proto-div-lambda 0.01 --init-global signot_global_branch --freeze-global
# (7) Global + OT + tongue mask mass (joint + frozen)
run global_ot_mask_joint   --readout signot --branch both --evidence ot --use-mask-mass
run global_ot_mask_frozen  --readout signot --branch both --evidence ot --use-mask-mass --init-global signot_global_branch --freeze-global

$PY collect_matrix.py --runs-glob "signot_*" --out "$PROJ/outputs/signot_matrix_summary.csv"
echo "=== DONE -> outputs/signot_matrix_summary.csv ==="
cat "$PROJ/outputs/signot_matrix_summary.csv"
