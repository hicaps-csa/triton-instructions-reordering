"""Turn nsys_results_2d.csv into a human-readable report.

For each matrix size, render 3x3 m-by-n tables (OFF us / ON us / speedup /
BW) so the (m, n) split is visible at a glance now that block_m and block_n
are real, independent parameters of the kernel.

cat_trunc_2d variant: dst is FP16 (2 bytes/elem) so each launch moves
4*M*N + 4*M*N + 2*M*2*N = 12*M*N bytes (vs 10 for the i8-quant variant).
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "nsys_results_2d.csv")
TXT_PATH = os.path.join(HERE, "nsys_results_2d.txt")

NUM_SMS = 68
DIMS = [64, 128, 256]
BYTES_PER_LAUNCH_PER_MN = 12  # 4 (read a fp32) + 4 (read b fp32) + 4 (write 2N fp16)


def bw_gb_s(M: int, N: int, ns: float) -> float:
    if ns <= 0:
        return 0.0
    return (BYTES_PER_LAUNCH_PER_MN * M * N) / (ns * 1e-9) / 1e9


def parse_csv(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                for k in ("M", "N", "block_m", "block_n", "block_lin",
                         "instances"):
                    r[k] = int(r[k])
                for k in ("avg_ns", "med_ns", "min_ns", "max_ns",
                         "stddev_ns"):
                    r[k] = float(r[k])
                rows.append(r)
            except (ValueError, KeyError):
                rows.append({**r, "_failed": True})
    return rows


def render_matrix(f, dims, cell):
    f.write("           " + " ".join(f"n={n:>5}" for n in dims) + "\n")
    for m in dims:
        row = [cell(m, n) for n in dims]
        f.write(f"   m={m:>4} " + " ".join(row) + "\n")


def write_report(rows):
    valid = [r for r in rows if not r.get("_failed")]
    failed = [r for r in rows if r.get("_failed")]
    sizes = sorted({r["M"] for r in valid})

    idx = {(r["pass_state"], r["M"], r["block_m"], r["block_n"]): r
           for r in valid}

    with open(TXT_PATH, "w") as f:
        w = f.write
        w("=" * 96 + "\n")
        w("  Triton 2-D cat_trunc_2d_kernel -- nsys timing sweep\n")
        w("  Pass under test: DowncastReorderOptimizer (push fp32->fp16 truncf before tt.cat)\n")
        w("=" * 96 + "\n\n")

        w("Setup\n")
        w("-" * 96 + "\n")
        w("  GPU         : RTX 2080 Ti (68 SMs)\n")
        w("  Stack       : Triton 3.6.0 / nsys 2023.2.3 / torch 2.9.1+cu128\n")
        w("  Iterations  : 100 between cudaProfilerStart/Stop, +3 warmup\n")
        w("  Reported    : per-launch avg from `nsys cuda_gpu_kern_sum`\n")
        w("  Block sizes : (BLOCK_M, BLOCK_N) in {64, 128, 256}^2 -- 9 distinct\n")
        w("                kernels per matrix size (no aliasing).\n")
        w("  Op chain    : cat(fp32, fp32) -> truncf -> fp16  (sibling of\n")
        w("                cat_quant_2d which uses cat -> fptosi -> i8).\n")
        w("  Bytes/launch: 4*M*N (read a) + 4*M*N (read b) + 2*M*2*N (write fp16)\n")
        w("              = 12*M*N\n\n")

        for size in sizes:
            n_elems = size * size
            w(f"=== Matrix size: {size} x {size}  ({n_elems:,} elements/matrix) ===\n\n")

            w("  Pass OFF -- avg per-launch time (us)\n")
            render_matrix(f, DIMS, lambda m, n: (
                f"{idx[('OFF', size, m, n)]['avg_ns']/1000:>7.1f}"
                if ('OFF', size, m, n) in idx else "    n/a"))
            w("\n")

            w("  Pass ON  -- avg per-launch time (us)\n")
            render_matrix(f, DIMS, lambda m, n: (
                f"{idx[('ON', size, m, n)]['avg_ns']/1000:>7.1f}"
                if ('ON', size, m, n) in idx else "    n/a"))
            w("\n")

            w("  Speedup (OFF/ON)  -- >1.00x means pass is faster\n")
            def sp_cell(m, n):
                off = idx.get(('OFF', size, m, n))
                on = idx.get(('ON', size, m, n))
                if not (off and on) or on['avg_ns'] <= 0:
                    return "    n/a"
                return f" {off['avg_ns']/on['avg_ns']:>5.2f}x"
            render_matrix(f, DIMS, sp_cell)
            w("\n")

            w("  Pass ON  -- effective bandwidth (GB/s, 12*M*N bytes/launch)\n")
            render_matrix(f, DIMS, lambda m, n: (
                f"{bw_gb_s(size, size, idx[('ON', size, m, n)]['avg_ns']):>7.1f}"
                if ('ON', size, m, n) in idx else "    n/a"))
            w("\n")

            w("  Detailed per-tile table\n")
            w(f"  {'(m,n)':>10} | {'tile':>7} | {'grid':>11} | "
              f"{'OFF us':>9} | {'ON us':>9} | {'speedup':>8} | "
              f"{'BW ON GB/s':>10}\n")
            w("  " + "-" * 90 + "\n")
            for m in DIMS:
                for n in DIMS:
                    off = idx.get(('OFF', size, m, n))
                    on = idx.get(('ON', size, m, n))
                    if not (off and on):
                        continue
                    tile = m * n
                    gx = -(-size // m); gy = -(-size // n)
                    grid = gx * gy
                    sp = off['avg_ns'] / on['avg_ns']
                    bw = bw_gb_s(size, size, on['avg_ns'])
                    w(f"  {f'{m}x{n}':>10} | {tile:>7} | "
                      f"{f'{gx}x{gy}={grid}':>11} | "
                      f"{off['avg_ns']/1000:>9.2f} | "
                      f"{on['avg_ns']/1000:>9.2f} | "
                      f"{sp:>7.2f}x | {bw:>10.1f}\n")
            w("\n")

        if failed:
            w("Failures / no-stat configurations\n")
            w("-" * 96 + "\n")
            for r in failed:
                w(f"  pass={r.get('pass_state')} M={r.get('M')} "
                  f"N={r.get('N')} m={r.get('block_m')} n={r.get('block_n')}\n")
            w("\n")

        # ---- analysis ----
        speedups = []
        for r in valid:
            if r["pass_state"] != "ON":
                continue
            off = idx.get(('OFF', r['M'], r['block_m'], r['block_n']))
            if not off:
                continue
            sp = off['avg_ns'] / r['avg_ns']
            speedups.append((r['M'], r['block_m'], r['block_n'], sp, off, r))
        speedups.sort(key=lambda x: x[3], reverse=True)

        w("Findings\n")
        w("-" * 96 + "\n")
        if speedups:
            best = speedups[0]
            worst = speedups[-1]
            w(f"  - Best speedup     : M=N={best[0]}  ({best[1]}, {best[2]})  "
              f"-> {best[3]:.2f}x  ({best[4]['avg_ns']/1000:.1f}us -> {best[5]['avg_ns']/1000:.1f}us)\n")
            w(f"  - Worst (regression): M=N={worst[0]}  ({worst[1]}, {worst[2]})  "
              f"-> {worst[3]:.2f}x  ({worst[4]['avg_ns']/1000:.1f}us -> {worst[5]['avg_ns']/1000:.1f}us)\n\n")

            wins = [s for s in speedups if s[3] >= 1.10]
            neutrals = [s for s in speedups if 0.95 < s[3] < 1.10]
            regressions = [s for s in speedups if s[3] <= 0.95]
            w(f"  - Wins (>=1.10x)        : {len(wins)} configs\n")
            w(f"  - Neutral (0.95-1.10x)  : {len(neutrals)} configs\n")
            w(f"  - Regressions (<=0.95x) : {len(regressions)} configs\n\n")

            w("  m vs n asymmetry (do different (m, n) with the same product\n")
            w("  show the same or different timing?)\n")
            w(f"    {'M=N':>5}  {'m*n':>7}  {'pairs':>34}  {'OFF us range':>20}  {'ON us range':>20}\n")
            buckets = defaultdict(list)
            for r in valid:
                if r["pass_state"] != "ON":
                    continue
                off = idx.get(('OFF', r['M'], r['block_m'], r['block_n']))
                if not off:
                    continue
                buckets[(r['M'], r['block_m'] * r['block_n'])].append(
                    (r['block_m'], r['block_n'], off['avg_ns'], r['avg_ns']))
            for (M, prod) in sorted(buckets.keys()):
                items = buckets[(M, prod)]
                if len(items) < 2:
                    continue
                pairs = ",".join(f"({m}x{n})" for m, n, *_ in items)
                off_min = min(it[2] for it in items)
                off_max = max(it[2] for it in items)
                on_min = min(it[3] for it in items)
                on_max = max(it[3] for it in items)
                w(f"    {M:>5}  {prod:>7}  {pairs:>34}  "
                  f"{off_min/1000:>7.1f}-{off_max/1000:>7.1f}us  "
                  f"{on_min/1000:>7.1f}-{on_max/1000:>7.1f}us\n")
            w("\n")

        # ---- recommendations ----
        w("Recommendations for follow-up experiments / ncu drill-down\n")
        w("-" * 96 + "\n")
        if speedups:
            target_size = max(sizes)
            candidates_for_size = [s for s in speedups if s[0] == target_size]
            if candidates_for_size:
                best_for = candidates_for_size[0]
                worst_for = candidates_for_size[-1]
                near_one = sorted(candidates_for_size,
                                  key=lambda s: abs(s[3] - 1.0))[0]
                w(f"  At M=N={target_size}, recommended ncu configs:\n")
                w(f"    win    : ({best_for[1]:>3}, {best_for[2]:>3})  speedup {best_for[3]:.2f}x\n")
                w(f"    regr   : ({worst_for[1]:>3}, {worst_for[2]:>3})  speedup {worst_for[3]:.2f}x\n")
                w(f"    neutral: ({near_one[1]:>3}, {near_one[2]:>3})  speedup {near_one[3]:.2f}x\n")

        w("\n  vs cat_quant_2d: dst is fp16 (2 bytes vs 1), so the post-cast\n")
        w("  working set is 2x bigger. Expect the win cliff in the same place\n")
        w("  but the regr/big regime slightly worse. Compare the speedup\n")
        w("  matrix here against `cat_quant_2d/nsys_results_2d.txt`.\n\n")

        w("=" * 96 + "\n")

    print(f"wrote {TXT_PATH}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"missing {CSV_PATH}")
    write_report(parse_csv(CSV_PATH))


if __name__ == "__main__":
    main()
