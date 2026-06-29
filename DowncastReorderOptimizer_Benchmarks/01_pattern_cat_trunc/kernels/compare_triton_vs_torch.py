"""Wall-clock comparison: Triton cat_trunc_2d vs torch eager equivalents.

Uses CUDA events to time the full op (all kernel launches included) over
many iterations. Prints per-iter avg in microseconds and torch/triton
speedup ratios.

Torch variants timed:
  - torch_cat_cast  : torch.cat([A,B], dim=1).to(fp16)
                      (2 launches, materializes fp32 M*2N intermediate)
  - torch_copy_cast : pre-allocate fp16 M*2N, dst[:, :N].copy_(A), dst[:, N:].copy_(B)
                      (2 launches, fp16-only writes, matches Triton's 12*M*N bytes)
"""
import argparse
import torch
import triton
import triton.language as tl


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

    c = tl.cat(a_flat, b_flat, can_reorder=True)
    d = c.to(tl.float16)
    d_2d = tl.reshape(d, (BLOCK_M, 2 * BLOCK_N))

    out_offs_n = pid_n * (2 * BLOCK_N) + tl.arange(0, 2 * BLOCK_N)
    out_mask = (offs_m[:, None] < M) & (out_offs_n[None, :] < 2 * N)
    tl.store(ptr_out + offs_m[:, None] * (2 * N) + out_offs_n[None, :],
             d_2d, mask=out_mask)


def time_op(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us per iter


def run(M, N, block_m, block_n, iters):
    device = torch.device("cuda")
    A = torch.randn(M, N, device=device, dtype=torch.float32)
    B = torch.randn(M, N, device=device, dtype=torch.float32)

    # --- Triton ---
    dst_tri = torch.empty(M, 2 * N, device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    def run_tri():
        cat_trunc_2d_kernel[grid](A, B, dst_tri, M, N,
                                  BLOCK_M=block_m, BLOCK_N=block_n)

    # --- torch.cat + .to(fp16) ---
    def run_cat_cast():
        torch.cat([A, B], dim=1).to(torch.float16)

    # --- preallocated fp16 dst, two copy_ casts ---
    dst_copy = torch.empty(M, 2 * N, device=device, dtype=torch.float16)

    def run_copy_cast():
        dst_copy[:, :N].copy_(A)
        dst_copy[:, N:].copy_(B)

    t_tri = time_op(run_tri, iters)
    t_cat = time_op(run_cat_cast, iters)
    t_cpy = time_op(run_copy_cast, iters)

    bytes_per_iter = 12 * M * N  # 4*M*N read A + 4*M*N read B + 2*M*2*N write fp16
    bw = lambda us: bytes_per_iter / (us * 1e-6) / 1e9  # GB/s

    print(f"\n=== M={M} N={N}  BLOCK_M={block_m} BLOCK_N={block_n}  iters={iters} ===")
    print(f"{'impl':<22} {'us/iter':>10} {'GB/s':>10} {'vs triton':>12}")
    print("-" * 58)
    print(f"{'triton':<22} {t_tri:>10.2f} {bw(t_tri):>10.1f} {'1.00x':>12}")
    print(f"{'torch_cat_cast':<22} {t_cat:>10.2f} {bw(t_cat):>10.1f} {t_cat/t_tri:>11.2f}x")
    print(f"{'torch_copy_cast':<22} {t_cpy:>10.2f} {bw(t_cpy):>10.1f} {t_cpy/t_tri:>11.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--block-m", type=int, default=64)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    run(args.M, args.N, args.block_m, args.block_n, args.iters)
