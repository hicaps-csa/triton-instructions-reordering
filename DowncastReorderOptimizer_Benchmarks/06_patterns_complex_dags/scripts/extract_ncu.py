"""Extract key NCU metrics from /tmp/ncu_runs_A2A7/*.csv into a single table.
"""
import csv, os, sys, glob, re

ROOT = "/tmp/ncu_runs_A2A7"

WANT = {
    "duration_us":      "gpu__time_duration.sum",
    "dram_rd_MB":       "dram__bytes_read.sum",
    "dram_wr_MB":       "dram__bytes_write.sum",
    "inst_M":           "smsp__inst_executed.sum",
    "regs":             "launch__registers_per_thread",
    "spill_ld":         "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
    "spill_st":         "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
    "occ_pct":          "sm__warps_active.avg.pct_of_peak_sustained_active",
}


def parse(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return None
    header = rows[0]; units = rows[1]; body = rows[2]
    h2i = {h.strip(): i for i, h in enumerate(header)}
    out = {}
    for friendly, mname in WANT.items():
        col = h2i.get(mname)
        if col is None:
            for k, i in h2i.items():
                if k.startswith(mname):
                    col = i; break
        if col is None:
            out[friendly] = ("?", "MISS"); continue
        out[friendly] = (body[col].strip(), units[col].strip())
    return out


def fmt(v):
    s, u = v
    try:
        x = float(s.replace(",", ""))
        if u == "msecond":   return f"{x*1000:.1f}us"
        if u == "second":    return f"{x*1e6:.1f}us"
        if u == "Gbyte":     return f"{x*1024:.1f}MB"
        if u == "Mbyte":     return f"{x:.1f}MB"
        if u == "Kbyte":     return f"{x/1024:.3f}MB"
        if u == "byte":      return f"{x/1024/1024:.3f}MB"
        if u == "%":         return f"{x:.2f}%"
        if abs(x) >= 1e9:    return f"{x/1e9:.2f}G"
        if abs(x) >= 1e6:    return f"{x/1e6:.2f}M"
        if abs(x) >= 1e3:    return f"{x/1e3:.2f}k"
        return f"{x:.2f}"
    except (ValueError, AttributeError):
        return s


PATTERN = re.compile(r"^(A\d)_(OFF|ON)_(\w+?)_(\d+)x(\d+)\.csv$")


def main():
    files = sorted(glob.glob(f"{ROOT}/*.csv"))
    rows = []
    for fp in files:
        name = os.path.basename(fp)
        m = PATTERN.match(name)
        if not m:
            continue
        tag, state, label, bm, bn = m.groups()
        d = parse(fp)
        if d is None:
            continue
        rows.append({"tag":tag, "state":state, "label":label,
                     "tile":f"{bm}x{bn}", "data":d})

    # Print one section per (tag, label) showing OFF vs ON
    by_key = {}
    for r in rows:
        k = (r["tag"], r["label"], r["tile"])
        by_key.setdefault(k, {})[r["state"]] = r["data"]

    print(f"{'tag':<3} {'label':<8} {'tile':<8} {'state':<5} | " +
          " | ".join(f"{m:<12}" for m in WANT))
    print("-" * 130)
    for k in sorted(by_key):
        for state in ("OFF", "ON"):
            d = by_key[k].get(state)
            if d is None:
                continue
            cells = " | ".join(f"{fmt(d[m]):<12}" for m in WANT)
            print(f"{k[0]:<3} {k[1]:<8} {k[2]:<8} {state:<5} | {cells}")
        print()


if __name__ == "__main__":
    main()
