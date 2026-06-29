#!/bin/bash
# ncu detailed for A2-A7. We pick TWO configs per kernel from the nsys
# results: the highest-speedup tile and the most-regression tile (or, if
# the kernel has only neutral perf, just the heaviest tile to confirm
# no spill change). Configurable via CONFIGS below.

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
ROOT="$PATTERN_DIR"
RUN_DIR=/tmp/ncu_runs_A2A7
TRITON_DEFAULT_CACHE="$HOME/.triton/cache"
M=8192; N=8192; ITERS=10

mkdir -p "$RUN_DIR"

# (tag, script, label, BM, BN). Initially populated with the heaviest
# tile per kernel; can be re-edited after seeing nsys results.
CONFIGS=(
  "A2 test_A2_cachehit_2d.py        win   128 128"
  "A3 test_A3_mixedkind_2d.py       BIGwin 128 256"
  "A3 test_A3_mixedkind_2d.py       win   64  256"
  "A4 test_A4_multilevel_2d.py      win   128 128"
  "A5 test_A5_catofcats_2d.py       BIGwin 64  256"
  "A5 test_A5_catofcats_2d.py       regr  128 256"
  "A6 test_A6_bitwidth_guard_2d.py  mid   128 128"
  "A7 test_A7_yshape_2d.py          mid   128 128"
)

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
  echo "===  Pass state: $PASS  ==="
  set_pass "$PASS"
  rm -rf "$TRITON_DEFAULT_CACHE"
  for ENTRY in "${CONFIGS[@]}"; do
    read -r TAG SCR LABEL BM BN <<< "$ENTRY"
    NAME="${TAG}_${PASS}_${LABEL}_${BM}x${BN}"
    REP="$RUN_DIR/$NAME.ncu-rep"; CSV="$RUN_DIR/$NAME.csv"; LOG="$RUN_DIR/$NAME.log"
    echo "  -> $NAME"
    sudo -n "$NCU" --target-processes all --set detailed \
      --launch-skip 5 --launch-count 1 --force-overwrite \
      --export "$REP" \
      "$PYBIN" "$ROOT/$SCR" --M "$M" --N "$N" --block-m "$BM" --block-n "$BN" --iterations "$ITERS" \
      > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "${REP}" ]; then
      echo "    [FAIL] rc=$rc -- see $LOG"; continue
    fi
    sudo -n "$NCU" --import "$REP" --csv --page raw > "$CSV" 2>/dev/null
    echo "    OK"
  done
done

set_pass ON
echo "DONE. /tmp/ncu_runs_A2A7"
