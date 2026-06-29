#!/bin/bash
# ncu detailed profiling for asymmetric_diamond_trans_safe.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

NCU=/opt/nvidia/nsight-compute/2023.2.2/ncu
PYBIN="${PYBIN:-python3}"
SCRIPT=$PATTERN_DIR/kernels/test_asymmetric_diamond_trans_safe_2d.py
RUN_DIR=/tmp/ncu_runs_adts
TRITON_DEFAULT_CACHE="$HOME/.triton/cache"

mkdir -p "$RUN_DIR"

CONFIGS=(
  "neutral 64  64"
  "mid     128 128"
  "win     128 256"
  "big     256 256"
)
M=8192; N=8192; ITERS=10

set_pass() {
  # Toggle DowncastReorderOptimizer via env var. The variable is exported
  # so subsequent nsys/ncu/python invocations in this shell inherit it.
  if [ "$1" = "OFF" ]; then
    export TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER=1
  else
    unset TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER
  fi
}

for PASS in OFF ON; do
  echo "==================================================================="
  echo "==  Pass state: $PASS"
  echo "==================================================================="
  set_pass "$PASS"
  rm -rf "$TRITON_DEFAULT_CACHE"
  for ENTRY in "${CONFIGS[@]}"; do
    LABEL=$(echo "$ENTRY" | awk '{print $1}')
    BM=$(echo "$ENTRY" | awk '{print $2}')
    BN=$(echo "$ENTRY" | awk '{print $3}')
    TAG="${PASS}_${LABEL}_${BM}x${BN}"
    REP="$RUN_DIR/$TAG.ncu-rep"; CSV="$RUN_DIR/$TAG.csv"; LOG="$RUN_DIR/$TAG.log"
    echo "  -> $TAG  (tile=$((BM * BN)))"
    sudo -n "$NCU" --target-processes all --set detailed \
      --launch-skip 5 --launch-count 1 --force-overwrite \
      --export "$REP" \
      "$PYBIN" "$SCRIPT" --M "$M" --N "$N" --block-m "$BM" --block-n "$BN" --iterations "$ITERS" \
      > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "${REP}" ]; then
      echo "    [FAIL] rc=$rc -- see $LOG"; continue
    fi
    sudo -n "$NCU" --import "$REP" --csv --page raw > "$CSV" 2>/dev/null
    echo "    OK  ($(wc -l < "$CSV") csv lines)"
  done
done

set_pass "ON"
echo "DONE. /tmp/ncu_runs_adts"
