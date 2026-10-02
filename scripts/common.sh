#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SEAT_PYTHON:-python}"
SRC="$REPO_ROOT/src"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
OUT="$("$PY" -c 'from settings import OUTPUTS_DIR; print(OUTPUTS_DIR)')"
RUNS="$("$PY" -c 'from settings import RUNS_DIR; print(RUNS_DIR)')"
FEATURES="$("$PY" -c 'from settings import FEATURES_DIR; print(FEATURES_DIR)')"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

# Fail before training when manifests contain split overlap. Hash/group checks are
# available through audit_dataset.py and must be completed on the actual dataset.
"$PY" "$SRC/audit_dataset.py"
