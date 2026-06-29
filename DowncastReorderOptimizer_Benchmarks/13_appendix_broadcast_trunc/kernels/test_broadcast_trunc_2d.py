"""2-D broadcast-trunc benchmark.

Loads a 1-D fp32 row of length N, broadcasts it across BLOCK_M rows of a
[BLOCK_M, BLOCK_N] tile, then narrows to fp16 (arith.truncf). Tile shape
is parameterized by independent BLOCK_M / BLOCK_N constexprs, so every
(m, n) compiles to a unique kernel.

The IR contains the tt.broadcast -> arith.truncf pattern that DowncastReorderOptimizer
targets (it sinks the cast above the broadcast so the broadcast emits the
narrowed dtype). Sibling kernels: cat_trunc_2d, splat_trunc_2d.

CLI mirrors test_cat_trunc_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse


@triton.jit
def broadcast_trunc_2d_kernel(
    ptr_in,   # [N] fp32 row, source of broadcast
    ptr_out,  # [M, N] fp16, broadcast result
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    row = tl.load(ptr_in + offs_n, mask=n_mask, other=0.0).to(tl.float32)   # [BLOCK_N]
    # Explicit broadcast_to keeps the truncf as the *immediate* consumer of
    # tt.broadcast. The `row[None, :] + tl.zeros(...)` idiom inserts an
    # arith.addf between the broadcast and the truncf, blocking the pass.
    bcast = tl.broadcast_to(row[None, :], (BLOCK_M, BLOCK_N))               # tt.broadcast
    out = bcast.to(tl.float16)                                               # arith.truncf

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptr_out + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src = torch.randn(N, device=device, dtype=torch.float32)
    dst = torch.empty(M, N, device=device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        broadcast_trunc_2d_kernel[grid](src, dst, M, N,
                                        BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        broadcast_trunc_2d_kernel[grid](src, dst, M, N,
                                        BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n}, fp16 truncation)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D broadcast->trunc Triton nsys benchmark")
    parser.add_argument("--M", type=int, default=8192, help="rows of the destination matrix")
    parser.add_argument("--N", type=int, default=8192, help="cols of source row / destination matrix")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
