#!/bin/bash
# Correctness check for asymmetric_diamond_trans_safe.
# Dumps .ttir for both states to confirm:
#   OFF: single tt.trans (fp32) feeds arith.truncf and tt.reshape.
#   ON : strict diamond gate bails -> OFF and ON IR identical (expected).
#        See run_relaxed_sweep.sh for the relaxed-gate variant.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

PYBIN="${PYBIN:-python3}"
SCRIPT=$PATTERN_DIR/kernels/test_asymmetric_diamond_trans_safe_2d.py
RESULTS_DIR=$PATTERN_DIR
LOG=$RESULTS_DIR/verify.log

SHAPES=(
  "1024 1024 64  64"
  "2048 2048 128 128"
  "4096 4096 128 256"
  "4096 4096 256 128"
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

: > "$LOG"
overall_ok=1

for SHAPE in "${SHAPES[@]}"; do
  read -r M N BM BN <<< "$SHAPE"
  declare -A HASH=()
  for PASS in OFF ON; do
    set_pass "$PASS"
    cache="/tmp/triton_cache_adts_${PASS,,}_${M}_${BM}x${BN}"
    rm -rf "$cache"; mkdir -p "$cache"
    line=$(TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR="$cache" \
           "$PYBIN" "$SCRIPT" --M "$M" --N "$N" \
           --block-m "$BM" --block-n "$BN" --verify 2>&1 | tail -1)
    sha=$(echo "$line" | sed -n 's/.*sha256=\([0-9a-f]*\).*/\1/p')
    HASH[$PASS]=$sha
    echo "[$PASS] M=$M N=$N BLOCK=($BM,$BN) $line" | tee -a "$LOG"
    ttir=$(find "$cache" -name "*.ttir" 2>/dev/null | head -1)
    if [ -n "$ttir" ]; then
      cp "$ttir" "$RESULTS_DIR/asymmetric_diamond_trans_safe_${PASS}_${M}_${BM}x${BN}.ttir"
    fi
  done
  if [ "${HASH[OFF]}" = "${HASH[ON]}" ] && [ -n "${HASH[OFF]}" ]; then
    echo "  -> MATCH ($M, $BM, $BN)" | tee -a "$LOG"
  else
    echo "  -> MISMATCH ($M, $BM, $BN): OFF=${HASH[OFF]} ON=${HASH[ON]}" | tee -a "$LOG"
    overall_ok=0
  fi
done

set_pass "ON"
echo
if [ "$overall_ok" -eq 1 ]; then
  echo "ALL SHAPES MATCH (OFF == ON). pass is value-preserving." | tee -a "$LOG"
else
  echo "MISMATCH DETECTED. see $LOG." | tee -a "$LOG"
  exit 1
fi
