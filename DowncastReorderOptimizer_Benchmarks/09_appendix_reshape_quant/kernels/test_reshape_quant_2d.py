"""2-D reshape-quant benchmark.

Loads an M*N fp32 matrix A as 2-D tiles, reshapes each tile to 1-D fp32,
truncates to int8, then reshapes back to 2-D before storing. The mover
op under test is tt.reshape; the reducer is arith.fptosi (fp32 -> i8).

The IR contains the tt.reshape -> arith.fptosi pattern that DowncastReorderOptimizer
sinks. Tile shape is parameterized by BLOCK_M and BLOCK_N as independent
constexprs to mirror the cat_quant_2d harness.

CLI mirrors test_cat_quant_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse


@triton.jit
def reshape_quant_2d_kernel(
    ptr_a, ptr_out,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(ptr_a + offs_m[:, None] * N + offs_n[None, :],
                mask=mask, other=0.0).to(tl.float32)

    a_flat = tl.reshape(a, (BLOCK_M * BLOCK_N,))   # tt.reshape  (mover under test)
    a_i8 = a_flat.to(tl.int8)                      # arith.fptosi (reducer)

    a_2d = tl.reshape(a_i8, (BLOCK_M, BLOCK_N))    # collapse back; consumer = store, not a reducer
    tl.store(ptr_out + offs_m[:, None] * N + offs_n[None, :], a_2d, mask=mask)


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src_a = torch.randn(M, N, device=device, dtype=torch.float32)
    dst = torch.empty(M, N, device=device, dtype=torch.int8)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        reshape_quant_2d_kernel[grid](src_a, dst, M, N,
                                      BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        reshape_quant_2d_kernel[grid](src_a, dst, M, N,
                                      BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D reshape->quant Triton nsys benchmark")
    parser.add_argument("--M", type=int, default=8192, help="rows of A")
    parser.add_argument("--N", type=int, default=8192, help="cols of A")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
