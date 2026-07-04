#!/usr/bin/env bash
# Boost the two weakest (color-defined) signs — TonguePale (~42.2) & Ecchymosis (~47.3) — via an
# OT detector over an explicit COLOR pathway (DINOv2 is semantic, under-encodes color). Trains a
# color-only and a DINOv2+color OT detector with the peak-detection term, selected on rare2
# (mean Pale+Ecchy val-F1). Prints the two target classes' test F1 for each run.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/autodl-tmp/conda/envs/vlm-vllm/bin/python
PROJ=/root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls
cd "$PROJ/scripts"
CSV="$PROJ/outputs/per_class_results_rarecolor.csv"; rm -f "$CSV"
rm -rf "$PROJ"/artifacts/runs/rarecolor_*

# 1) build the color cache (skip if present)
if [ ! -f "$PROJ/artifacts/features/train_patchgrid_color_seed42.pt" ]; then
  echo ">>> caching color features"; $PY cache_color_features.py --splits val test train
fi

COMMON="--readout otgen --seed 42 --epochs 120 --patience 20 --selection-metric rare2 \
  --threshold-step 0.01 --loss bce --pos-weight --per-class-csv $CSV \
  --ot-k 8 --otgen-combine concat --peak-topk 16 --eps-init 0.1 --lam-init 0.05"

echo ">>> color-only OT detector"
$PY train_readout.py $COMMON --tag color_seed42 \
  --run-name rarecolor_coloronly --experiment-name rarecolor_coloronly \
  > "$PROJ/artifacts/logs/rarecolor_coloronly.log" 2>&1

echo ">>> DINOv2 + color OT detector"
$PY train_readout.py $COMMON --tag dinov2grid_seed42 --extra-tag color_seed42 \
  --run-name rarecolor_dinocolor --experiment-name rarecolor_dinocolor \
  > "$PROJ/artifacts/logs/rarecolor_dinocolor.log" 2>&1

echo "=== TARGET CLASS RESULTS (baseline: Pale 42.22, Ecchy 47.25) ==="
for r in rarecolor_coloronly rarecolor_dinocolor; do
  echo -n "$r  "
  $PY - "$PROJ/artifacts/runs/$r/metrics_test.json" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))["per_label"]
p,e=d["TonguePale"],d["Ecchymosis"]
print(f'Pale={p["f1"]*100:.2f}(P{p["precision"]*100:.0f}/R{p["recall"]*100:.0f})  Ecchy={e["f1"]*100:.2f}(P{e["precision"]*100:.0f}/R{e["recall"]*100:.0f})')
EOF
done
