"""Turn nsys_results_2d.csv (dual_quant_2d sweep) into a human-readable report.

For each matrix size, render 3x3 m-by-n tables (OFF us / ON us / speedup /
BW) and a per-tile detailed table. The kernel here is the dual-precision
quantize fanout: cat (1-D fp32) feeds both fptosi (i8 store) and truncf
(f16 store). Both reducer paths are rewritten by DAG extension A; the
original fp32 cat dies post-pass.
"""
import csv
import os

CSV_PATH = "$PATTERN_DIR/nsys_results_2d.csv"
TXT_PATH = "$PATTERN_DIR/nsys_results_2d.txt"

NUM_SMS = 68
DIMS = [64, 128, 256]


def bw_gb_s(M: int, N: int, ns: float) -> float:
    """Bytes/launch:
        4*M*N (read a) + 4*M*N (read b)
      + 1*M*2*N (write i8) + 2*M*2*N (write f16)
      = 14*M*N
    """
    if ns <= 0:
        return 0.0
    return (14 * M * N) / (ns * 1e-9) / 1e9


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
        w("  Triton 2-D dual_quant_2d_kernel -- nsys timing sweep\n")
        w("  Pass under test: DowncastReorderOptimizer DAG extension A (multi-reducer-kind dedup cache)\n")
        w("  Diamond head : tt.cat (1-D fp32) -> { fptosi -> i8 store, truncf -> f16 store }\n")
        w("  Both users size-reducing -> guard does not bail; original fp32 cat dies post-pass.\n")
        w("=" * 96 + "\n\n")

        w("Setup\n")
        w("-" * 96 + "\n")
        w("  GPU         : RTX 2080 Ti (68 SMs)\n")
        w("  Stack       : Triton 3.6.0 / nsys 2023.2.3 / torch 2.9.1+cu128\n")
        w("  Iterations  : 100 between cudaProfilerStart/Stop, +3 warmup\n")
        w("  Reported    : per-launch avg from `nsys cuda_gpu_kern_sum`\n")
        w("  Block sizes : (BLOCK_M, BLOCK_N) in {64, 128, 256}^2 -- 9 distinct\n")
        w("                kernels per matrix size.\n")
        w("  Bytes/launch: 4*M*N (read a) + 4*M*N (read b)\n")
        w("              + 1*M*2*N (write i8) + 2*M*2*N (write f16)\n")
        w("              = 14*M*N\n\n")

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

            w("  Pass ON  -- effective bandwidth (GB/s, 14*M*N bytes/launch)\n")
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

        if speedups:
            target_size = max(sizes)
            cands = [s for s in speedups if s[0] == target_size]
            if cands:
                best_for = cands[0]
                worst_for = cands[-1]
                near_one = sorted(cands, key=lambda s: abs(s[3] - 1.0))[0]
                w(f"  At M=N={target_size}, recommended ncu configs:\n")
                w(f"    win    : ({best_for[1]:>3}, {best_for[2]:>3})  speedup {best_for[3]:.2f}x\n")
                w(f"    regr   : ({worst_for[1]:>3}, {worst_for[2]:>3})  speedup {worst_for[3]:.2f}x\n")
                w(f"    neutral: ({near_one[1]:>3}, {near_one[2]:>3})  speedup {near_one[3]:.2f}x\n")
        w("\n" + "=" * 96 + "\n")

    print(f"wrote {TXT_PATH}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"missing {CSV_PATH}")
    write_report(parse_csv(CSV_PATH))


if __name__ == "__main__":
    main()
