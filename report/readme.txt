DowncastReorderOptimizer -- MTech Final Report
================================================

This folder contains Final_Report.pdf, the final MTech report on the
DowncastReorderOptimizer pass added to Triton in this branch (v1).

Summary of changes in this branch (v1) on top of upstream main:

  1. New TTIR pass: DowncastReorderOptimizer
       Hoists element size-reducing casts (arith.truncf, arith.trunci,
       arith.fptosi, arith.fptoui, narrowing tt.fp_to_fp, narrowing
       arith.index_cast) upstream of safe data-movement ops (tt.cat,
       tt.trans, tt.broadcast, tt.reshape, tt.expand_dims, tt.join),
       so the data-movement op runs on the smaller element type.
       Worklist-driven with upstream propagation; dedup cache for
       multi-consumer fan-out; diamond / mixed-fanout bail-out.

  2. Pass integration:
       lib/Dialect/Triton/Transforms/DowncastReorderOptimizer.cpp
       include/triton/Dialect/Triton/Transforms/Passes.h
       include/triton/Dialect/Triton/Transforms/Passes.td
       lib/Dialect/Triton/Transforms/CMakeLists.txt
       python/src/passes.cc        (binding: add_downcast_reorder_optimizer)
       third_party/nvidia/backend/compiler.py
                                   (pipeline insertion in make_ttir)

  3. Runtime toggle:
       TRITON_DISABLE_DOWNCAST_REORDER_OPTIMIZER  (env var)
         unset    -> pass ON
         =1       -> pass OFF
       Read inside make_ttir at compile time; no rebuild required.

  4. Benchmark suite (DowncastReorderOptimizer_Benchmarks/):
       13 pattern folders mirroring the report's Patterns 1-12, the
       asymmetric-diamond control pair, and the appendix experiments.
       Each pattern is self-contained: kernels/, scripts/, nsys/,
       ncu/, summary/, INFO.txt. Sweep scripts iterate OFF then ON
       in a single invocation via the env var toggle, with
       script-relative paths so the tree runs from any location.

  5. Report (this folder):
       Final_Report.pdf  -- algorithm description, per-pattern
                            walkthroughs (Patterns 1-12, asymmetric
                            diamond, appendix experiments), and the
                            performance analysis (best-vs-best,
                            per-tile peak rescue, viable-tile counts,
                            DRAM-bandwidth model, stall-reason and
                            L1-absorption analyses).
