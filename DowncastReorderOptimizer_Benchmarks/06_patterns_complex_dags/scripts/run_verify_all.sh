#!/bin/bash
# Unified verify driver for A2-A7. For each kernel, compile OFF then ON,
# dump TTIR for inspection, and verify bit-exactness of all outputs.

# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u

PYBIN="${PYBIN:-python3}"
ROOT="$PATTERN_DIR"
LOG=$ROOT/verify.log

# (Tag, script-basename, M, N, BM, BN) -- each tag identifies the experiment.
TESTS=(
  "A2_cachehit       test_A2_cachehit_2d.py        2048 2048 128 128"
  "A3_mixedkind      test_A3_mixedkind_2d.py       2048 2048 128 128"
  "A4_multilevel     test_A4_multilevel_2d.py      2048 2048 128 128"
  "A5_catofcats      test_A5_catofcats_2d.py       2048 2048 128 128"
  "A6_bitwidth_guard test_A6_bitwidth_guard_2d.py  2048 2048 128 128"
  "A7_yshape         test_A7_yshape_2d.py          2048 2048 128 128"
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

for ENTRY in "${TESTS[@]}"; do
  read -r TAG SCR M N BM BN <<< "$ENTRY"
  echo "=================================================================" | tee -a "$LOG"
  echo "==  $TAG  ($M,$N) BLOCK=($BM,$BN)" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  declare -A HASH=()
  for PASS in OFF ON; do
    set_pass "$PASS"
    cache="/tmp/triton_cache_${TAG}_${PASS,,}"
    rm -rf "$cache"; mkdir -p "$cache"
    line=$(TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR="$cache" \
           "$PYBIN" "$ROOT/$SCR" --M "$M" --N "$N" \
           --block-m "$BM" --block-n "$BN" --verify 2>&1 | tail -1)
    sha=$(echo "$line" | sed -n 's/.*sha256=\([0-9a-f]*\).*/\1/p')
    HASH[$PASS]=$sha
    echo "[$PASS] $line" | tee -a "$LOG"
    ttir=$(find "$cache" -name "*.ttir" 2>/dev/null | head -1)
    if [ -n "$ttir" ]; then
      cp "$ttir" "$ROOT/${TAG}_${PASS}.ttir"
    fi
    ttgir=$(find "$cache" -name "*.ttgir" 2>/dev/null | head -1)
    if [ -n "$ttgir" ]; then
      cp "$ttgir" "$ROOT/${TAG}_${PASS}.ttgir"
    fi
    llir=$(find "$cache" -name "*.llir" 2>/dev/null | head -1)
    if [ -n "$llir" ]; then
      cp "$llir" "$ROOT/${TAG}_${PASS}.llir"
    fi
  done
  if [ "${HASH[OFF]}" = "${HASH[ON]}" ] && [ -n "${HASH[OFF]}" ]; then
    echo "  -> MATCH" | tee -a "$LOG"
  else
    echo "  -> MISMATCH: OFF=${HASH[OFF]} ON=${HASH[ON]}" | tee -a "$LOG"
    overall_ok=0
  fi
  echo "" | tee -a "$LOG"
done

set_pass ON
if [ "$overall_ok" -eq 1 ]; then
  echo "ALL TESTS BIT-EXACT (OFF==ON)." | tee -a "$LOG"
else
  echo "ONE OR MORE MISMATCH; see $LOG" | tee -a "$LOG"
  exit 1
fi
