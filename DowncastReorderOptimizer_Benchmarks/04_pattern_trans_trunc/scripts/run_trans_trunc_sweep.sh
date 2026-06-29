#!/bin/bash
# Sweep nsys timings for trans_trunc_fused kernel across BLOCK_M/BLOCK_N
# and pass-on vs pass-off. Writes CSV consumed by an aggregator.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

RESULTS_DIR=/tmp/nsys_trans_trunc
CSV="$PATTERN_DIR/nsys/trans_trunc_results.csv"
SCRIPT="$PATTERN_DIR/kernels/trans_trunc_fused.py"

# Square matrix sized so M*N ~= 2^25 (5793^2 = 33,558,849; 2^25 = 33,554,432).
M=5793
N=5793
ITERS=100

BLOCK_MS=(64 128 256 512)
BLOCK_NS=(64 128 256 512)

mkdir -p "$RESULTS_DIR"
echo "pass_state,M,N,block_m,block_n,instances,avg_ns,med_ns,min_ns,max_ns,stddev_ns,total_ns" > "$CSV"

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
  local rep="$RESULTS_DIR/$tag.nsys-rep"
  local cache="/tmp/triton_tt_cache_${pass,,}"

  TRITON_CACHE_DIR="$cache" \
    nsys profile \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      -o "$RESULTS_DIR/$tag" \
      python "$SCRIPT" --m "$M" --n "$N" --block-m "$bm" --block-n "$bn" \
                       --iterations "$ITERS" --mode nsys \
      > "$RESULTS_DIR/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$rep" ]; then
    echo "$pass,$M,$N,$bm,$bn,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
    echo "  [FAIL] $tag (rc=$rc) -- see $RESULTS_DIR/$tag.log"
    return
  fi

  local row
  row=$(nsys stats --report cuda_gpu_kern_sum --format csv "$rep" 2>/dev/null \
        | grep trans_trunc_fused_kernel | head -1)
  if [ -z "$row" ]; then
    echo "$pass,$M,$N,$bm,$bn,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS" >> "$CSV"
    echo "  [NOSTATS] $tag"
    return
  fi

  local total instances avg med mn mx stddev
  IFS=',' read -r _pct total instances avg med mn mx stddev _name <<< "$(echo "$row" | sed 's/"//g')"
  echo "$pass,$M,$N,$bm,$bn,$instances,$avg,$med,$mn,$mx,$stddev,$total" >> "$CSV"
  echo "  $pass bm=$bm bn=$bn  avg=${avg}ns inst=$instances"
}

for PASS in OFF ON; do
  echo "=================================================================="
  echo "==  Pass state: $PASS"
  echo "=================================================================="
  set_pass "$PASS"
  rm -rf "/tmp/triton_tt_cache_${PASS,,}"
  mkdir -p "/tmp/triton_tt_cache_${PASS,,}"
  for BM in "${BLOCK_MS[@]}"; do
    for BN in "${BLOCK_NS[@]}"; do
      run_one "$PASS" "$BM" "$BN"
    done
  done
done

# Leave the pass ON for follow-up testing.
set_pass "ON"

echo "DONE. CSV: $CSV"
