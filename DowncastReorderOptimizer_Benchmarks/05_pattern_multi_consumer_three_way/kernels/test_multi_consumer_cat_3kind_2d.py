"""2-D 3-DISTINCT-KIND multi-consumer cat benchmark (Experiment A1).

Extends multi_consumer_cat_2d from 2 consumers to 3 distinct-kind consumers
on a single fp32 tt.cat. The point is to stress the dedup cache key
(OperationName, ResultType) in applyMultiConsumerOptimization: with three
different OperationName values feeding the same producer, the cache must
produce THREE independent rewritten chains -- no false sharing, no
collisions.

Kinds chosen (verified via TTIR probing on this Triton build):
  arith.fptosi : fp32 -> int8
  arith.truncf : fp32 -> float16
  tt.fp_to_fp  : fp32 -> float8e5  (this is the only path that exercises
                                    the triton::FpToFpOp branch of
                                    createSizeReducingClone, which carries
                                    getRoundingAttr() through -- a code
                                    path otherwise untested in the suite)

NOTE: fp32 -> bf16 (the original A1 spec) emits arith.truncf, not tt.fp_to_fp,
so it would NOT exercise a third distinct kind. fp8e5 is the only consumer
that forces tt.fp_to_fp on this build.

    load A (fp32)          load B (fp32)
        |                       |
     reshape                 reshape          <- 2D -> 1D movers
        \\__________ cat _________/           <- SHARED mover (1-D fp32)
                      |
            +---------+---------+
            |         |         |
         fptosi    truncf    fp_to_fp         <- 3 reducers, 3 distinct kinds
         (i8)      (fp16)    (fp8e5)
            |         |         |
         reshape   reshape   reshape          <- 1D -> 2D movers
            |         |         |
         store i8  store f16 store fp8

Pass OFF: single fp32 tt.cat feeds three reducers.
Pass ON : exactly three rewritten cat chains, one per (OperationName,
          ResultType). Original fp32 cat dies because all uses are now on
          one of the three rewritten chains.

CLI mirrors test_multi_consumer_cat_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse
import hashlib


@triton.jit
def multi_consumer_cat_3kind_2d_kernel(
    ptr_a, ptr_b, ptr_out_i8, ptr_out_f16, ptr_out_fp8,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(ptr_a + offs_m[:, None] * N + offs_n[None, :],
                mask=in_mask, other=0.0).to(tl.float32)
    b = tl.load(ptr_b + offs_m[:, None] * N + offs_n[None, :],
                mask=in_mask, other=0.0).to(tl.float32)

    a_flat = tl.reshape(a, (BLOCK_M * BLOCK_N,))
    b_flat = tl.reshape(b, (BLOCK_M * BLOCK_N,))

    c = tl.cat(a_flat, b_flat, can_reorder=True)   # diamond head: 1-D fp32

    d_i8  = c.to(tl.int8)        # arith.fptosi
    d_f16 = c.to(tl.float16)     # arith.truncf
    d_fp8 = c.to(tl.float8e5)    # tt.fp_to_fp

    d_i8_2d  = tl.reshape(d_i8,  (2 * BLOCK_M, BLOCK_N))
    d_f16_2d = tl.reshape(d_f16, (2 * BLOCK_M, BLOCK_N))
    d_fp8_2d = tl.reshape(d_fp8, (2 * BLOCK_M, BLOCK_N))

    out_row = pid_m * (2 * BLOCK_M) + tl.arange(0, 2 * BLOCK_M)
    out_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (out_row[:, None] < 2 * M) & (out_col[None, :] < N)

    tl.store(ptr_out_i8 + out_row[:, None] * N + out_col[None, :],
             d_i8_2d, mask=out_mask)
    tl.store(ptr_out_f16 + out_row[:, None] * N + out_col[None, :],
             d_f16_2d, mask=out_mask)
    tl.store(ptr_out_fp8 + out_row[:, None] * N + out_col[None, :],
             d_fp8_2d, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    b = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a, b


def verify(M, N, block_m, block_n):
    device = torch.device("cuda")
    a, b = _make_inputs(M, N, device)
    out_i8  = torch.empty(2 * M, N, device=device, dtype=torch.int8)
    out_f16 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    out_fp8 = torch.empty(2 * M, N, device=device, dtype=torch.float8_e5m2)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    multi_consumer_cat_3kind_2d_kernel[grid](
        a, b, out_i8, out_f16, out_fp8, M, N,
        BLOCK_M=block_m, BLOCK_N=block_n,
    )
    torch.cuda.synchronize()
    # combined hash + per-output hashes
    h = hashlib.sha256()
    h.update(out_i8.cpu().numpy().tobytes())
    h.update(out_f16.cpu().numpy().tobytes())
    # float8 has no numpy dtype; bit-cast to uint8
    h.update(out_fp8.view(torch.uint8).cpu().numpy().tobytes())
    h_i8  = hashlib.sha256(out_i8.cpu().numpy().tobytes()).hexdigest()
    h_f16 = hashlib.sha256(out_f16.cpu().numpy().tobytes()).hexdigest()
    h_fp8 = hashlib.sha256(out_fp8.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({block_m},{block_n}) "
          f"sha256={h.hexdigest()} "
          f"sha_i8={h_i8} sha_f16={h_f16} sha_fp8={h_fp8}")


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!"); return
    device = torch.device("cuda")

    src_a = torch.randn(M, N, device=device, dtype=torch.float32)
    src_b = torch.randn(M, N, device=device, dtype=torch.float32)
    dst_i8  = torch.empty(2 * M, N, device=device, dtype=torch.int8)
    dst_f16 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    dst_fp8 = torch.empty(2 * M, N, device=device, dtype=torch.float8_e5m2)

    grid = lambda meta: (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    for _ in range(3):
        multi_consumer_cat_3kind_2d_kernel[grid](
            src_a, src_b, dst_i8, dst_f16, dst_fp8, M, N,
            BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        multi_consumer_cat_3kind_2d_kernel[grid](
            src_a, src_b, dst_i8, dst_f16, dst_fp8, M, N,
            BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"Completed {num_iterations} iterations for nsys profiling "
          f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2-D 3-kind multi-consumer cat (A1) Triton benchmark")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify(args.M, args.N, args.block_m, args.block_n)
    else:
        run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n,
                           args.iterations)
