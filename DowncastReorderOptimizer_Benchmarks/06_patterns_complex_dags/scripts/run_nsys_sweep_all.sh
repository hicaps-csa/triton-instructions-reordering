#!/bin/bash
# Unified nsys sweep for A2-A7. For each kernel:
#   sizes  = {4096, 8192}
#   tiles  = {(64,64), (64,256), (128,128), (128,256), (256,128)}  -- 5 representative
#   states = {OFF, ON}
# => 6 kernels x 2 sizes x 5 tiles x 2 states = 120 captures.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

RESULTS_DIR=/tmp/nsys_runs_A2A7
CSV=$PATTERN_DIR/nsys/nsys_results.csv
PYBIN="${PYBIN:-python3}"
ROOT="$PATTERN_DIR"

mkdir -p "$RESULTS_DIR"
echo "tag,kernel_sym,pass_state,M,N,block_m,block_n,block_lin,instances,avg_ns,med_ns,min_ns,max_ns,stddev_ns,total_ns" > "$CSV"

# (tag, script, kernel_name)
KERNELS=(
  "A2 test_A2_cachehit_2d.py        A2_kernel"
  "A3 test_A3_mixedkind_2d.py       A3_kernel"
  "A4 test_A4_multilevel_2d.py      A4_kernel"
  "A5 test_A5_catofcats_2d.py       A5_kernel"
  "A6 test_A6_bitwidth_guard_2d.py  A6_kernel"
  "A7 test_A7_yshape_2d.py          A7_kernel"
)

TILES=(
  "64  64"
  "64  256"
  "128 128"
  "128 256"
  "256 128"
)
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
  local tag=$1 scr=$2 ksym=$3 pass=$4 size=$5 bm=$6 bn=$7
  local lin=$((bm * bn))
  local label="${tag}_${pass}_S${size}_${bm}x${bn}"
  local rep="$RESULTS_DIR/$label.nsys-rep"
  local cache="/tmp/triton_cache_A2A7_${pass,,}"

  TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR="$cache" \
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite=true -o "$RESULTS_DIR/$label" \
      "$PYBIN" "$ROOT/$scr" --M "$size" --N "$size" \
      --block-m "$bm" --block-n "$bn" --iterations "$ITERS" \
      > "$RESULTS_DIR/$label.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$rep" ]; then
    echo "$tag,$ksym,$pass,$size,$size,$bm,$bn,$lin,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$CSV"
    echo "  [FAIL] $label (rc=$rc)"
    return
  fi
  local row
  row=$(nsys stats --report cuda_gpu_kern_sum --format csv "$rep" 2>/dev/null \
        | grep "$ksym" | head -1)
  if [ -z "$row" ]; then
    echo "$tag,$ksym,$pass,$size,$size,$bm,$bn,$lin,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS" >> "$CSV"
    echo "  [NOSTATS] $label"
    return
  fi
  local total instances avg med min_v max_v stddev
  IFS=',' read -r _pct total instances avg med min_v max_v stddev _name <<< "$(echo "$row" | sed 's/"//g')"
  echo "$tag,$ksym,$pass,$size,$size,$bm,$bn,$lin,$instances,$avg,$med,$min_v,$max_v,$stddev,$total" >> "$CSV"
  echo "  $tag $pass S=${size}^2 ${bm}x${bn} avg=${avg}ns"
}

for PASS in OFF ON; do
  echo "==================================================================="
  echo "==  Pass state: $PASS"
  echo "==================================================================="
  set_pass "$PASS"
  rm -rf "/tmp/triton_cache_A2A7_${PASS,,}"
  mkdir -p "/tmp/triton_cache_A2A7_${PASS,,}"
  for KENTRY in "${KERNELS[@]}"; do
    read -r TAG SCR KSYM <<< "$KENTRY"
    for SIZE in "${SIZES[@]}"; do
      for TILE in "${TILES[@]}"; do
        read -r BM BN <<< "$TILE"
        run_one "$TAG" "$SCR" "$KSYM" "$PASS" "$SIZE" "$BM" "$BN"
      done
    done
  done
done

set_pass ON
echo "DONE. CSV: $CSV"
