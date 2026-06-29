#!/bin/bash
# ncu detailed profiling for 2-D dual-quant kernel.
# 4 configs (neutral / regr / win / big) x 2 pass states = 8 captures, all at
# M = N = 8192. sudoers rule does NOT allow env-passing, so we use the
# default Triton cache and wipe it between pass toggles.
#
# CONFIGS picked AFTER the nsys sweep completed -- see comments below.
# Update the (BM, BN) tuples to match the highest-signal nsys results
# (typically: best speedup -> "win", worst -> "regr", closest-to-1.00 ->
# "neutral", and a large tile -> "big").

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
SCRIPT=$PATTERN_DIR/kernels/test_dual_quant_2d.py
RUN_DIR=/tmp/ncu_runs_dual_quant
TRITON_DEFAULT_CACHE="$HOME/.triton/cache"

mkdir -p "$RUN_DIR"

# Configs picked from the 8192^2 nsys sweep (see nsys_results_2d.txt):
#   neutral (64, 64)   -- 0.99x, BW-bound small tile, no spill expected
#   regr    (128, 256) -- 0.69x headline regression, 32K-element tile
#   win     (128, 128) -- 1.28x headline win, mid-tile
#   big     (256, 256) -- 1.00x, 64K-element tile, both states heavy
CONFIGS=(
  "neutral 64  64"     # tile = 4096   (~0.99x at 8192^2)
  "regr    128 256"    # tile = 32768  (0.69x at 8192^2 -- headline regression)
  "win     128 128"    # tile = 16384  (1.28x at 8192^2 -- headline win)
  "big     256 256"    # tile = 65536  (~1.00x at 8192^2)
)
M=8192
N=8192
ITERS=10

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
  echo "=================================================================="
  echo "==  Pass state: $PASS"
  echo "=================================================================="
  set_pass "$PASS"
  rm -rf "$TRITON_DEFAULT_CACHE"

  for ENTRY in "${CONFIGS[@]}"; do
    LABEL=$(echo "$ENTRY" | awk '{print $1}')
    BM=$(echo "$ENTRY" | awk '{print $2}')
    BN=$(echo "$ENTRY" | awk '{print $3}')
    TAG="${PASS}_${LABEL}_${BM}x${BN}"
    REP="$RUN_DIR/$TAG.ncu-rep"
    CSV="$RUN_DIR/$TAG.csv"
    LOG="$RUN_DIR/$TAG.log"

    echo "  -> $TAG  (tile=$((BM * BN)))"

    sudo -n "$NCU" \
      --target-processes all \
      --set detailed \
      --launch-skip 5 \
      --launch-count 1 \
      --force-overwrite \
      --export "$REP" \
      "$PYBIN" "$SCRIPT" \
      --M "$M" --N "$N" --block-m "$BM" --block-n "$BN" --iterations "$ITERS" \
      > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "${REP}" ]; then
      echo "    [FAIL] rc=$rc -- see $LOG"
      continue
    fi

    sudo -n "$NCU" --import "$REP" --csv --page raw > "$CSV" 2>/dev/null
    LINES=$(wc -l < "$CSV")
    echo "    OK  ($LINES csv lines)"
  done
done

set_pass "ON"
echo "DONE. /tmp/ncu_runs_dual_quant"
