"""2-D join-quant benchmark.

Loads two M*N fp32 matrices A, B as 2-D tiles, joins them along a new
minor dim (BM, BN, 2), truncates to int8, then collapses the last two
dims to (BM, 2*BN) before storing. The mover op under test is tt.join;
the reducer is arith.fptosi.

This is the closest analog to cat_quant_2d: tt.join doubles the element
count just like tt.cat (lhs + rhs material), so the same register-spill
cliff should appear in the OFF case at large tiles. The pass moves the
fptosi onto the operands of join, halving the post-join register
footprint.

CLI mirrors test_cat_quant_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse


@triton.jit
def join_quant_2d_kernel(
    ptr_a, ptr_b, ptr_out,
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
    b = tl.load(ptr_b + offs_m[:, None] * N + offs_n[None, :],
                mask=mask, other=0.0).to(tl.float32)

    j = tl.join(a, b)                                # tt.join (mover under test) -> [BM, BN, 2]
    j_i8 = j.to(tl.int8)                             # arith.fptosi (reducer)

    j_2d = tl.reshape(j_i8, (BLOCK_M, 2 * BLOCK_N))  # collapse last two dims; consumer = store

    out_offs_n = pid_n * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    out_mask = (offs_m[:, None] < M) & (out_offs_n[None, :] < 2 * N)
    tl.store(ptr_out + offs_m[:, None] * (2 * N) + out_offs_n[None, :],
             j_2d, mask=out_mask)


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src_a = torch.randn(M, N, device=device, dtype=torch.float32)
    src_b = torch.randn(M, N, device=device, dtype=torch.float32)
    dst = torch.empty(M, 2 * N, device=device, dtype=torch.int8)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        join_quant_2d_kernel[grid](src_a, src_b, dst, M, N,
                                   BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        join_quant_2d_kernel[grid](src_a, src_b, dst, M, N,
                                   BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D join->quant Triton nsys benchmark")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
