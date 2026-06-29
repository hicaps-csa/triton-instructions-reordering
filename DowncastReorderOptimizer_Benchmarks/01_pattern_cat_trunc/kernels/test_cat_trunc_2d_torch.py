"""Torch-eager equivalent of test_cat_trunc_2d.py.

Concatenates two M*N fp32 matrices A, B along dim=1 and narrows the
result to fp16. This is the eager-mode counterpart of the
cat_trunc_2d Triton kernel (tt.cat -> arith.truncf), used as a
performance baseline.

CLI mirrors test_cat_trunc_2d.py; block-m/block-n are accepted but
ignored (tiling is implicit in torch eager mode).
"""
import torch
import argparse


def cat_trunc_2d_torch(src_a: torch.Tensor, src_b: torch.Tensor) -> torch.Tensor:
    return torch.cat([src_a, src_b], dim=1).to(torch.float16)


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src_a = torch.randn(M, N, device=device, dtype=torch.float32)
    src_b = torch.randn(M, N, device=device, dtype=torch.float32)

    for _ in range(3):
        dst = cat_trunc_2d_torch(src_a, src_b)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        dst = cat_trunc_2d_torch(src_a, src_b)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n}, fp16 truncation, torch eager)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D cat->trunc torch eager nsys benchmark")
    parser.add_argument("--M", type=int, default=8192, help="rows of A and B")
    parser.add_argument("--N", type=int, default=8192, help="cols of A and B")
    parser.add_argument("--block-m", type=int, default=128, help="ignored in eager mode")
    parser.add_argument("--block-n", type=int, default=128, help="ignored in eager mode")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
