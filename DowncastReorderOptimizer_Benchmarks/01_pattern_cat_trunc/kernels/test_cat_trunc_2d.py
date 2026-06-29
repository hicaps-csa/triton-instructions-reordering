"""2-D cat-trunc benchmark.

Loads two M*N fp32 matrices A, B as 2-D tiles, concatenates them, and
narrows to fp16 (arith.truncf). Tile shape is parameterized by BLOCK_M
and BLOCK_N as independent constexprs, so every (m, n) compiles to a
unique kernel.

Triton's tl.cat is 1-D only, so we reshape the 2-D tiles to 1-D, do the
cat, truncate, and reshape back. The IR still contains the
tt.cat -> arith.truncf pattern that DowncastReorderOptimizer targets (the
fp32->fp16 sibling of the cat_quant_2d fp32->i8 version).

CLI arguments mirror test_cat_quant_2d.py so the existing sweep harness
needs minimal changes.
"""
import torch
import triton
import triton.language as tl
import argparse


@triton.jit
def cat_trunc_2d_kernel(
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

    a_flat = tl.reshape(a, (BLOCK_M * BLOCK_N,))
    b_flat = tl.reshape(b, (BLOCK_M * BLOCK_N,))

    c = tl.cat(a_flat, b_flat, can_reorder=True)   # fp32, length 2*BLOCK_M*BLOCK_N
    d = c.to(tl.float16)                           # fp16, same length (truncf)

    d_2d = tl.reshape(d, (BLOCK_M, 2 * BLOCK_N))

    out_offs_n = pid_n * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    out_mask = (offs_m[:, None] < M) & (out_offs_n[None, :] < 2 * N)
    tl.store(ptr_out + offs_m[:, None] * (2 * N) + out_offs_n[None, :],
             d_2d, mask=out_mask)


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src_a = torch.randn(M, N, device=device, dtype=torch.float32)
    src_b = torch.randn(M, N, device=device, dtype=torch.float32)
    dst = torch.empty(M, 2 * N, device=device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        cat_trunc_2d_kernel[grid](src_a, src_b, dst, M, N,
                                  BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        cat_trunc_2d_kernel[grid](src_a, src_b, dst, M, N,
                                  BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n}, fp16 truncation)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D cat->trunc Triton nsys benchmark")
    parser.add_argument("--M", type=int, default=8192, help="rows of A and B")
    parser.add_argument("--N", type=int, default=8192, help="cols of A and B")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
