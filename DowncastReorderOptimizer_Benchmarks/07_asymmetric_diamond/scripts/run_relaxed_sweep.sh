#!/bin/bash
# RELAXED-GATE sweep for asymmetric_diamond_trans.
#
# The current DowncastReorderOptimizer intentionally bails on any producer with a
# non-reducer user (DowncastReorderOptimizer.cpp:339-341). This script:
#   1. Backs up DowncastReorderOptimizer.cpp
#   2. Patches the bail to `continue;` so the diamond rewrite fires
#   3. Rebuilds the pass via ninja
#   4. Runs nsys + ncu sweeps with the relaxed gate
#      (output files tagged *_RELAXED* so they don't clobber the standard run)
#   5. Restores DowncastReorderOptimizer.cpp from backup
#   6. Rebuilds to put the tree back to stock
#
# The trap ensures restore-on-exit so a Ctrl-C or build failure does NOT
# leave the source tree in a modified state.
#
# Purpose: quantify the regression the current strict gate is avoiding,
# on a fresh mover (trans) different from the cat-based pattern that
# originally motivated the gate (diamond_residual_2d / Exp 6). Output
# diff vs the standard ON sweep == the value the gate adds.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

# This sweep patches the C++ pass source, rebuilds, runs the sweep, and
# restores the source. Set the next two env vars to point at your Triton
# checkout before running:
PASS_SRC="${PASS_SRC:?set PASS_SRC=/path/to/triton/lib/Dialect/Triton/Transforms/DowncastReorderOptimizer.cpp}"
BUILD_DIR="${BUILD_DIR:?set BUILD_DIR=/path/to/triton/build/cmake.*}"
BACKUP="${PASS_SRC}.diamond-bak"

PYBIN="${PYBIN:-python3}"
SCRIPT=$PATTERN_DIR/kernels/test_asymmetric_diamond_trans_2d.py
NCU=/opt/nvidia/nsight-compute/2023.2.2/ncu
TRITON_DEFAULT_CACHE="$HOME/.triton/cache"

RESULTS_DIR=/tmp/nsys_runs_adt_relaxed
NCU_RUN_DIR=/tmp/ncu_runs_adt_relaxed
CSV=$PATTERN_DIR/nsys/nsys_results_2d_relaxed.csv

restore_source() {
  if [ -f "$BACKUP" ]; then
    echo ">> restoring $PASS_SRC from backup"
    cp -p "$BACKUP" "$PASS_SRC"
    rm -f "$BACKUP"
    echo ">> rebuilding pass back to stock"
    ( cd "$BUILD_DIR" && ninja ) || echo ">> WARNING: stock rebuild failed; tree may be inconsistent"
  fi
}
trap restore_source EXIT INT TERM

# --- patch ---
echo ">> backing up $PASS_SRC -> $BACKUP"
cp -p "$PASS_SRC" "$BACKUP"

echo ">> patching diamond gate (return {}; -> continue;)"
perl -0777 -i -pe 's/(if \(!isSizeReducingOp\(user\)\)\n)        return \{\};/$1        continue; \/\/ RELAXED_DIAMOND_GATE/' "$PASS_SRC"

if ! grep -q "RELAXED_DIAMOND_GATE" "$PASS_SRC"; then
  echo ">> patch FAILED to apply; trap will restore"
  exit 1
fi
echo ">> patch applied:"; grep -n "RELAXED_DIAMOND_GATE" "$PASS_SRC"

echo ">> rebuilding pass via ninja"
( cd "$BUILD_DIR" && ninja ) || { echo ">> ninja failed"; exit 1; }

# --- pass toggle: env-var based, so subsequent commands inherit it ---
set_pass_on()  { unset TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER; }
set_pass_off() { export TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER=1; }

mkdir -p "$RESULTS_DIR" "$NCU_RUN_DIR"
echo "pass_state,M,N,block_m,block_n,block_lin,instances,avg_ns,med_ns,min_ns,max_ns,stddev_ns,total_ns" > "$CSV"

DIMS=(64 128 256)
SIZES=(4096 8192)
ITERS=100

run_one_nsys() {
  local pass=$1 size=$2 bm=$3 bn=$4
  local lin=$((bm * bn))
  local tag="${pass}_S${size}_${bm}x${bn}"
  local rep="$RESULTS_DIR/$tag.nsys-rep"
  local cache="/tmp/triton_cache_adt_relaxed_${pass,,}"

  TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR="$cache" \
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite=true -o "$RESULTS_DIR/$tag" \
      "$PYBIN" "$SCRIPT" --M "$size" --N "$size" \
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
        | grep asymmetric_diamond_trans_2d_kernel | head -1)
  if [ -z "$row" ]; then
    echo "$pass,$size,$size,$bm,$bn,$lin,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS,NOSTATS" >> "$CSV"
    return
  fi
  local total instances avg med min_v max_v stddev
  IFS=',' read -r _pct total instances avg med min_v max_v stddev _name <<< "$(echo "$row" | sed 's/"//g')"
  echo "$pass,$size,$size,$bm,$bn,$lin,$instances,$avg,$med,$min_v,$max_v,$stddev,$total" >> "$CSV"
  echo "  $pass S=${size}^2 m=$bm n=$bn (lin=$lin)  avg=${avg}ns inst=$instances"
}

echo "==================================================================="
echo "==  RELAXED nsys sweep (pass ON; the relaxed gate now lets the   =="
echo "==  diamond rewrite fire). Compare vs standard nsys_results_2d.  =="
echo "==================================================================="
# Capture both OFF and ON. OFF here is the same kernel without the pass
# at all, which matches the standard OFF (no perf change either way).
# Including OFF makes the relaxed CSV self-contained for speedup math.
for PASS in OFF ON; do
  if [ "$PASS" = "OFF" ]; then set_pass_off; else set_pass_on; fi
  grep -n "add_downcast_reorder_optimizer" "$COMPILER"
  rm -rf "/tmp/triton_cache_adt_relaxed_${PASS,,}"
  mkdir -p "/tmp/triton_cache_adt_relaxed_${PASS,,}"
  for SIZE in "${SIZES[@]}"; do
    for BM in "${DIMS[@]}"; do
      for BN in "${DIMS[@]}"; do
        run_one_nsys "$PASS" "$SIZE" "$BM" "$BN"
      done
    done
  done
done

set_pass_on
echo "DONE nsys. CSV: $CSV"

# --- ncu pass (4 configs, ON only -- OFF is captured by the standard sweep) ---
echo "==================================================================="
echo "==  RELAXED ncu sweep (pass ON only, 4 configs at 8192^2)        =="
echo "==================================================================="
rm -rf "$TRITON_DEFAULT_CACHE"
NCU_CONFIGS=(
  "neutral 64  64"
  "mid     128 128"
  "win     128 256"
  "big     256 256"
)
for ENTRY in "${NCU_CONFIGS[@]}"; do
  LABEL=$(echo "$ENTRY" | awk '{print $1}')
  BM=$(echo "$ENTRY" | awk '{print $2}')
  BN=$(echo "$ENTRY" | awk '{print $3}')
  TAG="ON_RELAXED_${LABEL}_${BM}x${BN}"
  REP="$NCU_RUN_DIR/$TAG.ncu-rep"
  CSV2="$NCU_RUN_DIR/$TAG.csv"
  LOG="$NCU_RUN_DIR/$TAG.log"
  echo "  -> $TAG"
  sudo -n "$NCU" --target-processes all --set detailed \
    --launch-skip 5 --launch-count 1 --force-overwrite \
    --export "$REP" \
    "$PYBIN" "$SCRIPT" --M 8192 --N 8192 \
    --block-m "$BM" --block-n "$BN" --iterations 10 \
    > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "${REP}" ]; then
    echo "    [FAIL] rc=$rc -- see $LOG"
    continue
  fi
  sudo -n "$NCU" --import "$REP" --csv --page raw > "$CSV2" 2>/dev/null
  echo "    OK  ($(wc -l < "$CSV2") csv lines)"
done

echo "DONE. Source restore happens on trap EXIT."
