#!/bin/bash
# Compare OFF/ON SASS opcode counts for the A1 kernel at a given tile.
# Usage:  diff_sass.sh <BLOCK_M> <BLOCK_N>      (default 128 256)
# Compiles both OFF and ON with a fresh cache, then dumps the SASS opcode
# histogram side-by-side for the kernel symbol.


# --- portable path resolution (added by release builder) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATTERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYBIN="${PYBIN:-python3}"
# Override RESULTS_DIR / CACHE_BASE via the environment if you want
# the artifacts to land somewhere other than /tmp.
# -----------------------------------------------------------

set -u
BM=${1:-128}; BN=${2:-256}; M=${M:-8192}; N=${N:-8192}

PYBIN="${PYBIN:-python3}"
SCRIPT=$PATTERN_DIR/kernels/test_multi_consumer_cat_3kind_2d.py
OUT=/tmp/sass_mcc3_${BM}x${BN}
mkdir -p "$OUT"

set_pass() {
  # Toggle DowncastReorderOptimizer via env var. The variable is exported
  # so subsequent nsys/ncu/python invocations in this shell inherit it.
  if [ "$1" = "OFF" ]; then
    export TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER=1
  else
    unset TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER
  fi
}

compile_dump() {
  local pass=$1
  set_pass "$pass"
  local cache="/tmp/triton_cache_mcc3_${pass,,}_sass_${BM}x${BN}"
  rm -rf "$cache"; mkdir -p "$cache"
  TRITON_ALWAYS_COMPILE=1 TRITON_CACHE_DIR="$cache" \
    "$PYBIN" "$SCRIPT" --M "$M" --N "$N" \
    --block-m "$BM" --block-n "$BN" --verify >/dev/null 2>&1
  local cubin
  cubin=$(find "$cache" -name "*.cubin" 2>/dev/null | head -1)
  if [ -z "$cubin" ]; then
    echo "FAILED to find cubin for $pass" >&2
    return 1
  fi
  /usr/local/cuda/bin/cuobjdump --dump-sass "$cubin" > "$OUT/${pass}.sass" 2>/dev/null
  awk '/^\s+\/\*/ {
        for(i=1;i<=NF;i++){
          if ($i ~ /^[A-Z][A-Z0-9_]/ && $i !~ /^R[0-9]/ && $i !~ /^\[/) {
            n=$i; sub(/\..*/,"",n);  # strip variant suffix
            print n; break
          }
        }
      }' "$OUT/${pass}.sass" | sort | uniq -c | sort -k2 > "$OUT/${pass}.hist"
  echo "[$pass] wrote $OUT/${pass}.sass and $OUT/${pass}.hist"
}

compile_dump OFF
compile_dump ON
set_pass ON  # restore

# Side-by-side: join on opcode
echo
echo "Opcode counts at ($BM, $BN):"
echo "  opcode               OFF        ON      delta"
echo "  ----------------- ------- --------- ----------"
join -t'|' -a1 -a2 -e0 -j2 -o '0,1.1,2.1' \
  <(awk '{print $1"|"$2}' "$OUT/OFF.hist" | sort -t'|' -k2,2) \
  <(awk '{print $1"|"$2}' "$OUT/ON.hist"  | sort -t'|' -k2,2) \
| awk -F'|' '{printf "  %-17s %7d %9d %10d\n", $1, $2, $3, $3-$2}' \
| sort -k4 -n
