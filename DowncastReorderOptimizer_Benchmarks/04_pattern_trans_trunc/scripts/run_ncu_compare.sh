#!/bin/bash
# Run ncu --set full for trans_trunc_fused at 3 representative block sizes
# under pass OFF and pass ON. Caches separated per pass so kernels recompile.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

RESULTS_DIR=/tmp/ncu_trans_trunc
SCRIPT="$PATTERN_DIR/kernels/trans_trunc_fused.py"
PYTHON="${PYTHON:-${PYBIN:-python3}}"
NCU="${NCU:-ncu}"

M=5793
N=5793
ITERS=1   # warmup is hardcoded to 3 inside the script

BLOCKS=("64 64" "128 128" "256 256")

mkdir -p "$RESULTS_DIR"

set_pass() {
  # Toggle DowncastReorderOptimizer via env var. The variable is exported
  # so subsequent nsys/ncu/python invocations in this shell inherit it.
  if [ "$1" = "OFF" ]; then
    export TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER=1
  else
    unset TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER
  fi
}

run_one() {
  local pass=$1 bm=$2 bn=$3
  local tag="${pass}_bm${bm}_bn${bn}"
  local rep="$RESULTS_DIR/$tag.ncu-rep"
  local cache="/tmp/triton_tt_cache_${pass,,}"

  echo "=== ncu $tag ==="
  TRITON_CACHE_DIR="$cache" \
    sudo -n -E "$NCU" --set full \
        --launch-skip 3 --launch-count 1 \
        --target-processes application-only \
        --force-overwrite \
        -o "$RESULTS_DIR/$tag" \
        "$PYTHON" "$SCRIPT" --m "$M" --n "$N" --block-m "$bm" --block-n "$bn" \
                         --iterations "$ITERS" --mode nsys \
        > "$RESULTS_DIR/$tag.log" 2>&1
  local rc=$?
  echo "  rc=$rc  rep=$rep"
}

for PASS in OFF ON; do
  echo "=================================================================="
  echo "==  Pass state: $PASS"
  echo "=================================================================="
  set_pass "$PASS"
  rm -rf "/tmp/triton_tt_cache_${PASS,,}"
  mkdir -p "/tmp/triton_tt_cache_${PASS,,}"
  for BLK in "${BLOCKS[@]}"; do
    BM=${BLK% *}; BN=${BLK#* }
    run_one "$PASS" "$BM" "$BN"
  done
done

# Restore pass ON
set_pass "ON"
echo "DONE. Reports under: $RESULTS_DIR"
