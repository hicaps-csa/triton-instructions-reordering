# DowncastReorderOptimizer — benchmark suite

Per-experiment kernels, sweep scripts, and recorded results that
accompany the MTech report on `DowncastReorderOptimizer`, a Triton TTIR
pass that hoists element size-reducing casts upstream of data-movement
ops so the mover runs on the smaller element type.

The Triton compiler with the pass installed is assumed to already be
on this machine (drop this folder anywhere inside or next to the
Triton checkout); these artifacts do not rebuild the compiler.

## Layout

Each pattern lives in its own top-level folder and follows the same
internal layout:

```
<pattern>/
  INFO.txt        report section + source folder mapping
  kernels/        Triton kernel under test (plus any torch baseline)
  scripts/        sweep / verify scripts and aggregator helpers
  nsys/           recorded nsys results (.txt and .csv)
  ncu/            recorded ncu results
  summary/        writeups, spill analyses, per-pattern notes
```

## Pattern index (mapped to `Final_Report.tex`)

Folders are numbered in report order, so `ls -1` matches the
report's table of contents.

| Folder                                       | Report section                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| `01_pattern_cat_trunc/`                      | §6.5 — Pattern 1: Concatenate -- Truncate                                      |
| `02_pattern_deep_chain/`                     | §6.6 — Pattern 2: Deep Mover Chain                                             |
| `03_pattern_dual_quant/`                     | §6.7 — Pattern 3: Multi-consumer (two-way)                                     |
| `04_pattern_trans_trunc/`                    | §6.8 — Pattern 4: Transpose -- Truncate                                        |
| `05_pattern_multi_consumer_three_way/`       | §6.9 — Pattern 5: Multi-consumer (three-way)                                   |
| `06_patterns_complex_dags/`                  | §6.10 — Patterns 6-11: Complex DAGs (cache hit, mixed-kind, multi-level diamond, cat-of-cats, bit-width guard, Y-shape) |
| `07_asymmetric_diamond/`                     | §6.10.6 — Asymmetric Diamond (relaxed gate sweep)                              |
| `08_asymmetric_diamond_safe/`                | §6.10.6 — Asymmetric Diamond (safe-side control)                               |
| `09_appendix_reshape_quant/`                 | §A.1 — `reshape` + quant                                                       |
| `10_appendix_join_quant/`                    | §A.2 — `join` + quant                                                          |
| `11_appendix_multi_consumer_trans/`          | §A.3 — Multi-consumer `trans`                                                  |
| `12_pattern_cat_quant/`                      | §A.5 — Pattern 12: Concatenate -- Quantize                                     |
| `13_appendix_broadcast_trunc/`               | §A.6 — `broadcast` + truncate                                                  |

Patterns 6-11 share their kernels and sweep scripts under
`06_patterns_complex_dags/` because the original `A2_to_A7/` driver
iterates over all six kernels in a single sweep.

## Running a sweep

Every sweep script is a single Bash file under `<pattern>/scripts/`
and iterates both pass states (`OFF` then `ON`) in one invocation:

```
bash <pattern>/scripts/run_nsys_sweep_2d.sh
```

The script resolves its own location, finds the kernel under
`<pattern>/kernels/`, and writes results into `<pattern>/nsys/`.
Override the defaults via these environment variables if needed:

| Variable        | Default                | Meaning                                       |
| --------------- | ---------------------- | --------------------------------------------- |
| `PYBIN`         | `python3`              | Python interpreter to use                     |
| `RESULTS_DIR`   | `/tmp/nsys_runs_*`     | Where nsys captures land                      |
| `TRITON_CACHE_DIR` | `/tmp/triton_cache_*` | Per-pass-state Triton cache                  |

## Pass toggle

The pass is toggled at runtime via a single environment variable read
inside `make_ttir`:

```
TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER=1   # pass OFF
unset TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER  # pass ON
```

Every `set_pass` function inside the sweep scripts simply
exports / unsets this variable, so the toggle takes effect on the
next compile (`TRITON_ALWAYS_COMPILE=1` is also set inside the
scripts to bypass the disk cache).

## Relaxed-gate sweep — extra prerequisites

`07_asymmetric_diamond/scripts/run_relaxed_sweep.sh` and its safe-side
sibling temporarily patch the C++ pass source, rebuild Triton, run the
sweep, and restore the source. These two scripts (only) require
pointing at your Triton checkout:

```
PASS_SRC=/path/to/triton/lib/Dialect/Triton/Transforms/DowncastReorderOptimizer.cpp \
BUILD_DIR=/path/to/triton/build/cmake.linux-x86_64-cpython-3.* \
bash 07_asymmetric_diamond/scripts/run_relaxed_sweep.sh
```

Other sweeps do not touch the compiler source.

## Requirements

- A Triton build with `DowncastReorderOptimizer` registered in the
  TTIR pipeline (`add_downcast_reorder_optimizer`).
- An NVIDIA GPU (the report's numbers are on an RTX 2080 Ti, Turing,
  68 SMs, CC 7.5).
- `nsys` (NVIDIA Nsight Systems, 2023.2.x tested) for timing sweeps.
- `ncu` (NVIDIA Nsight Compute, 2023.2.x tested) for the perf-counter
  sweeps. `ncu` is invoked as `sudo ncu` in some scripts because the
  perf counters need root; see the per-script comments.
- Python 3 with `torch` (2.9.1+cu128 used for the report), `numpy`,
  and `pandas` (for the aggregator scripts).

## What's preserved as-is

The `summary/*.txt` writeups are historical artifacts of the original
runs. They may still contain absolute paths from the original user's
machine; these are prose references, not anything the scripts read.
