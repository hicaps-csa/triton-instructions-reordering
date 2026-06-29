"""Aggregate nsys_results.csv for A2-A7 into a single human-readable report.

For each kernel tag, render OFF/ON tables and speedup matrices across the
shared (size, tile) grid.
"""
import csv, os

CSV_PATH = "$PATTERN_DIR/nsys_results.csv"
TXT_PATH = "$PATTERN_DIR/nsys_results.txt"

TAGS = ["A2", "A3", "A4", "A5", "A6", "A7"]
TAG_NAMES = {
    "A2": "cache-HIT (same-kind, same-type, different operands)",
    "A3": "MIXED-KIND, SAME-TYPE (fptosi vs fptoui to i8)",
    "A4": "MULTI-LEVEL DIAMOND (cat then 2 movers then truncf, cache hit at cat)",
    "A5": "CAT-OF-CATS (3 cats, single truncf, 1-sweep convergence)",
    "A6": "BITWIDTH GUARD (fp16->i16 guard FAILS, fp32->i16 PASSES)",
    "A7": "Y-SHAPE UPSTREAM (shared load, redundant truncfs, CSE catches)",
}


def parse(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                for k in ("M","N","block_m","block_n","block_lin","instances"):
                    r[k] = int(r[k])
                for k in ("avg_ns","med_ns","min_ns","max_ns","stddev_ns"):
                    r[k] = float(r[k])
                rows.append(r)
            except (ValueError, KeyError):
                pass
    return rows


def write_report(rows):
    idx = {(r["tag"], r["pass_state"], r["M"], r["block_m"], r["block_n"]): r
           for r in rows}
    sizes_present = sorted({r["M"] for r in rows})
    tiles_present = sorted({(r["block_m"], r["block_n"]) for r in rows},
                          key=lambda t: t[0]*1000 + t[1])

    with open(TXT_PATH, "w") as f:
        w = f.write
        w("=" * 96 + "\n")
        w("  A2-A7 unified perf sweep -- DowncastReorderOptimizer DAG-topology gap tests\n")
        w("  GPU         : RTX 2080 Ti (68 SMs)\n")
        w("  Stack       : Triton 3.6.0 / nsys 2023.2.3\n")
        w("  Iterations  : 100 per capture, +3 warmup\n")
        w("=" * 96 + "\n\n")

        for tag in TAGS:
            tag_rows = [r for r in rows if r["tag"] == tag]
            if not tag_rows:
                w(f"--- {tag} : NO DATA ---\n\n")
                continue
            w("-" * 96 + "\n")
            w(f"  {tag} : {TAG_NAMES[tag]}\n")
            w("-" * 96 + "\n\n")
            for size in sizes_present:
                if not any(r["M"] == size for r in tag_rows):
                    continue
                w(f"  M=N={size}\n")
                w(f"    {'tile':>10} | {'OFF us':>10} | {'ON us':>10} | {'speedup':>8}\n")
                w(f"    " + "-" * 50 + "\n")
                for (bm, bn) in tiles_present:
                    off = idx.get((tag, "OFF", size, bm, bn))
                    on  = idx.get((tag, "ON",  size, bm, bn))
                    if not (off and on):
                        continue
                    sp = off["avg_ns"] / on["avg_ns"] if on["avg_ns"] > 0 else 0.0
                    w(f"    {f'{bm}x{bn}':>10} | {off['avg_ns']/1000:>10.2f} | "
                      f"{on['avg_ns']/1000:>10.2f} | {sp:>7.2f}x\n")
                w("\n")
            # Per-tag summary stats
            ops = []
            for (tag2, st, M2, bm2, bn2), r in idx.items():
                if tag2 != tag or st != "ON":
                    continue
                off = idx.get((tag2, "OFF", M2, bm2, bn2))
                if not off:
                    continue
                sp = off["avg_ns"]/r["avg_ns"] if r["avg_ns"] > 0 else 0
                ops.append((M2, bm2, bn2, sp))
            if ops:
                ops.sort(key=lambda x: x[3], reverse=True)
                wins = [o for o in ops if o[3] >= 1.10]
                neut = [o for o in ops if 0.95 < o[3] < 1.10]
                regr = [o for o in ops if o[3] <= 0.95]
                w(f"  {tag} summary across {len(ops)} configs:\n")
                w(f"    best   : {ops[0][1]:>3}x{ops[0][2]:<3} @ {ops[0][0]}^2 -> {ops[0][3]:.2f}x\n")
                w(f"    worst  : {ops[-1][1]:>3}x{ops[-1][2]:<3} @ {ops[-1][0]}^2 -> {ops[-1][3]:.2f}x\n")
                w(f"    wins / neutral / regr : {len(wins)} / {len(neut)} / {len(regr)}\n\n")

        w("=" * 96 + "\n")
    print(f"wrote {TXT_PATH}")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"missing {CSV_PATH}")
    write_report(parse(CSV_PATH))


if __name__ == "__main__":
    main()
