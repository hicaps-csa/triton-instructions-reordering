#!/bin/bash
# nsys sweep: 2-D reshape-quant kernel
#   - 2 matrix sizes : 4096x4096 and 8192x8192
#   - 9 (m, n) tiles : {64, 128, 256} on each axis
#   - 2 pass states  : OFF, ON
# = 36 captures, ~10-15 min wall time.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

KERNEL_NAME=reshape_quant_2d_kernel
RESULTS_DIR=/tmp/nsys_runs_reshape_quant_2d
CSV=$PATTERN_DIR/nsys/nsys_results_2d.csv
PYBIN="${PYBIN:-python3}"
SCRIPT=$PATTERN_DIR/kernels/test_reshape_quant_2d.py

mkdir -p "$RESULTS_DIR"
echo "pass_state,M,N,block_m,block_n,block_lin,instances,avg_ns,med_ns,min_ns,max_ns,stddev_ns,total_ns" > "$CSV"

DIMS=(64 128 256)
SIZES=(4096 8192)
ITERS=100

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
  local pass=$1 size=$2 bm=$3 bn=$4
  local lin=$((bm * bn))
  local tag="${pass}_S${size}_${bm}x${bn}"
  local rep="$RESULTS_DIR/$tag.nsys-rep"
  local cache="/tmp/triton_cache_reshape_quant_2d_${pass,,}"

  TRITON_ALWAYS_COMPILE=1 \
  TRITON_CACHE_DIR="$cache" \
    nsys profile \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --force-overwrite=true \
      -o "$RESULTS_DIR/$tag" \
      "$PYBIN" "$SCRIPT" \
        --M "$size" --N "$size" \
        --block-m "$bm" --block-n "$bn" --iterations "$ITERS" \
      > "$RESULTS_DIR/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$rep" ]; then
    echo "$pass,$size,$size,$bm,$bn,$lin,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
    echo "  [FAIL] $tag (rc=$rc)"
    return
  fi

  local row
  row=$(nsys stats --report cuda_gpu_kern_sum --format csv "$rep" 2>/dev/null \
        | grep "$KERNEL_NAME" | head -1)
  if [ -z "$row" ]; then
    echo "$pass,$size,$size,$bm,$bn,$lin,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS" >> "$CSV"
    echo "  [NOSTATS] $tag"
    return
  fi

  local total instances avg med min_v max_v stddev
  IFS=',' read -r _pct total instances avg med min_v max_v stddev _name <<< "$(echo "$row" | sed 's/"//g')"
  echo "$pass,$size,$size,$bm,$bn,$lin,$instances,$avg,$med,$min_v,$max_v,$stddev,$total" >> "$CSV"
  echo "  $pass S=${size}^2 m=$bm n=$bn (lin=$lin)  avg=${avg}ns inst=$instances"
}

for PASS in OFF ON; do
  echo "=================================================================="
  echo "==  Pass state: $PASS"
  echo "=================================================================="
  set_pass "$PASS"
  rm -rf "/tmp/triton_cache_reshape_quant_2d_${PASS,,}"
  mkdir -p "/tmp/triton_cache_reshape_quant_2d_${PASS,,}"
  for SIZE in "${SIZES[@]}"; do
    for BM in "${DIMS[@]}"; do
      for BN in "${DIMS[@]}"; do
        run_one "$PASS" "$SIZE" "$BM" "$BN"
      done
    done
  done
done

set_pass "ON"
echo "DONE. CSV: $CSV"
