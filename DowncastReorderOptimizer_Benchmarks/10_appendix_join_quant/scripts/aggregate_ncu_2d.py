"""Aggregate ncu CSVs from the 2-D reshape-quant sweep into a focused report."""
import csv
from pathlib import Path

KERNEL_LABEL = "join_quant_2d_kernel"
RUN_DIR = Path("/tmp/ncu_runs_join_quant_2d")
TXT_PATH = "$PATTERN_DIR/ncu_results_2d.txt"

CONFIGS = [
    ("neutral", 64, 64),
    ("win",     128, 256),
    ("big",     256, 256),
]
PASSES = ["OFF", "ON"]

METRICS = [
    ("gpu__time_duration.sum", "Kernel duration", "ms"),
    ("launch__registers_per_thread", "Regs/thread (used)", ""),
    ("launch__registers_per_thread_allocated", "Regs/thread (alloc)", ""),
    ("launch__occupancy_limit_registers", "Occ. limit (regs)", "warps"),
    ("launch__occupancy_limit_warps", "Occ. limit (max)", "warps"),
    ("launch__waves_per_multiprocessor", "Waves / SM", ""),
    ("sm__warps_active.avg.pct_of_peak_sustained_active",
     "Achieved occupancy", "%"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
     "SM (compute) thrpt", "%"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
     "Memory thrpt", "%"),
    ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
     "DRAM thrpt", "%"),
    ("lts__throughput.avg.pct_of_peak_sustained_elapsed",
     "L2 thrpt", "%"),
    ("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
     "Local-mem LD reqs (spill rd)", ""),
    ("l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
     "Local-mem ST reqs (spill wr)", ""),
    ("smsp__inst_executed.sum", "Total inst executed", ""),
    ("launch__block_size", "Block size (threads)", ""),
    ("launch__grid_size", "Grid size (blocks)", ""),
]

STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
STALL_REASONS = [
    "long_scoreboard", "short_scoreboard", "wait", "drain",
    "lg_throttle", "math_pipe_throttle", "barrier", "branch_resolving",
    "membar", "selected", "tex_throttle", "no_instruction",
    "dispatch_stall", "imc_miss",
]


def to_float(v):
    if v is None or v == "":
        return None
    s = str(v).replace(",", "").strip()
    if s.lower() in ("nan", "inf", "-inf"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv(path: Path):
    with open(path) as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r and any(c.strip() for c in r)]
    if len(rows) < 2:
        return {}
    header = rows[0]
    data = rows[-1]
    if len(data) != len(header):
        for r in rows[1:]:
            if len(r) == len(header):
                data = r; break
    return dict(zip(header, data))


def fmt(v, unit):
    if v is None: return "    n/a"
    if unit == "ms": return f"{v*1000:>9.2f} us"
    if unit == "%": return f"{v:>8.2f} %"
    if v >= 1e6 or (v >= 1e3 and v == int(v)):
        return f"{v:>13,.0f}"
    if v == int(v): return f"{v:>13.0f}"
    return f"{v:>13.2f}"


def main():
    if not RUN_DIR.exists():
        raise SystemExit(f"missing {RUN_DIR}")

    data = {}
    for ps in PASSES:
        for label, m, n in CONFIGS:
            tag = f"{ps}_{label}_{m}x{n}"
            path = RUN_DIR / f"{tag}.csv"
            if not path.exists():
                print(f"missing {path}"); continue
            data[(ps, label)] = load_csv(path)

    with open(TXT_PATH, "w") as f:
        w = f.write
        w("=" * 96 + "\n")
        w(f"  ncu detailed profiling -- 2-D {KERNEL_LABEL} @ M = N = 8192\n")
        w("  3 configs (neutral / win / big) x 2 pass states (one launch profiled per run)\n")
        w("=" * 96 + "\n\n")

        w("Configurations\n")
        w("-" * 96 + "\n")
        for label, m, n in CONFIGS:
            w(f"  {label:<8} : (BLOCK_M, BLOCK_N) = ({m}, {n})  tile = {m*n}\n")
        w("\n")

        for label, m, n in CONFIGS:
            off = data.get(("OFF", label), {})
            on = data.get(("ON", label), {})
            if not off or not on: continue
            w(f"=== {label.upper()}  (BLOCK_M, BLOCK_N) = ({m}, {n})  tile = {m*n} ===\n\n")

            w(f"  {'Metric':<32}  {'OFF':>17}  {'ON':>17}  {'Delta':>14}\n")
            w("  " + "-" * 90 + "\n")
            for col, lbl, unit in METRICS:
                vo = to_float(off.get(col))
                vn = to_float(on.get(col))
                delta = ""
                if vo is not None and vn is not None and vo != 0:
                    if unit == "%":
                        delta = f"{vn - vo:>+7.2f} ppt"
                    else:
                        delta = f"{vn / vo:>9.2f}x"
                w(f"  {lbl:<32}  {fmt(vo, unit):>17}  {fmt(vn, unit):>17}  {delta:>14}\n")
            w("\n")

            stalls_off, stalls_on = {}, {}
            for r in STALL_REASONS:
                col = STALL_PREFIX + r
                vo = to_float(off.get(col))
                vn = to_float(on.get(col))
                if vo is not None: stalls_off[r] = vo
                if vn is not None: stalls_on[r] = vn
            if stalls_off or stalls_on:
                w("  Top warp-stall reasons (raw counts; higher = more stalled)\n")
                w(f"    {'Reason':<24}  {'OFF':>14}  {'ON':>14}\n")
                all_r = set(stalls_off) | set(stalls_on)
                ranked = sorted(all_r,
                                key=lambda r: max(stalls_off.get(r, 0), stalls_on.get(r, 0)),
                                reverse=True)
                for r in ranked[:8]:
                    vo = stalls_off.get(r); vn = stalls_on.get(r)
                    w(f"    {r:<24}  {fmt(vo, ''):>14}  {fmt(vn, ''):>14}\n")
                w("\n")

        w("Compact summary (one row per config)\n")
        w("-" * 96 + "\n")
        w(f"  {'Config':<9} {'Pass':<4} {'Dur us':>9}  {'Reg':>4}  "
          f"{'Occ %':>6}  {'Mem %':>6}  {'DRAM %':>6}  "
          f"{'L2 %':>6}  {'Spill LD':>10}  {'Spill ST':>10}\n")
        for label, m, n in CONFIGS:
            for ps in PASSES:
                d = data.get((ps, label), {})
                if not d: continue
                dur = to_float(d.get("gpu__time_duration.sum"))
                reg = to_float(d.get("launch__registers_per_thread"))
                occ = to_float(d.get("sm__warps_active.avg.pct_of_peak_sustained_active"))
                mem = to_float(d.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"))
                dram = to_float(d.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"))
                l2 = to_float(d.get("lts__throughput.avg.pct_of_peak_sustained_elapsed"))
                sld = to_float(d.get("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum"))
                sst = to_float(d.get("l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"))

                def cell(v, w_, fp=2):
                    if v is None: return f"{'n/a':>{w_}}"
                    if v >= 1e6: return f"{v/1e6:>{w_-1}.1f}M"
                    if v >= 1e3: return f"{v/1e3:>{w_-1}.1f}K"
                    return f"{v:>{w_}.{fp}f}"

                w(f"  {label:<9} {ps:<4} {(dur*1000 if dur else float('nan')):>9.2f}  "
                  f"{(int(reg) if reg else 0):>4}  "
                  f"{(occ if occ else 0):>5.1f}%  {(mem if mem else 0):>5.1f}%  "
                  f"{(dram if dram else 0):>5.1f}%  {(l2 if l2 else 0):>5.1f}%  "
                  f"{cell(sld, 10, 0)}  {cell(sst, 10, 0)}\n")
        w("\n")

        w("Analysis\n")
        w("-" * 96 + "\n")
        for label, m, n in CONFIGS:
            off = data.get(("OFF", label), {})
            on = data.get(("ON", label), {})
            if not (off and on): continue
            reg_off = to_float(off.get("launch__registers_per_thread")) or 0
            reg_on = to_float(on.get("launch__registers_per_thread")) or 0
            sld_off = to_float(off.get("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum")) or 0
            sld_on = to_float(on.get("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum")) or 0
            sst_off = to_float(off.get("l1tex__t_requests_pipe_lsu_mem_local_op_st.sum")) or 0
            sst_on = to_float(on.get("l1tex__t_requests_pipe_lsu_mem_local_op_st.sum")) or 0
            dur_off = to_float(off.get("gpu__time_duration.sum")) or 0
            dur_on = to_float(on.get("gpu__time_duration.sum")) or 0
            occ_off = to_float(off.get("sm__warps_active.avg.pct_of_peak_sustained_active")) or 0
            occ_on = to_float(on.get("sm__warps_active.avg.pct_of_peak_sustained_active")) or 0

            w(f"  [{label.upper()}] (BLOCK_M, BLOCK_N) = ({m}, {n})  tile = {m*n}\n")
            speedup = dur_off / dur_on if dur_on else float("nan")
            w(f"    speedup (OFF/ON)        : {speedup:.2f}x\n")
            w(f"    regs/thread             : OFF={reg_off:.0f}  ON={reg_on:.0f}\n")
            w(f"    occupancy               : OFF={occ_off:.1f}%  ON={occ_on:.1f}%\n")
            spill_off = sld_off + sst_off
            spill_on = sld_on + sst_on
            w(f"    spill traffic (LD+ST)   : OFF={spill_off:>14,.0f}  ON={spill_on:>14,.0f}\n")
            if spill_off > 1e5 or spill_on > 1e5:
                if spill_on < spill_off * 0.5:
                    w("    -> pass eliminates most spill traffic\n")
                elif spill_off > 1e5 and spill_on > 1e5:
                    w("    -> both versions spill heavily\n")
            else:
                w("    -> negligible spill traffic in either version\n")
            w("\n")

        w("=" * 96 + "\n")

    print(f"wrote {TXT_PATH}")


if __name__ == "__main__":
    main()
