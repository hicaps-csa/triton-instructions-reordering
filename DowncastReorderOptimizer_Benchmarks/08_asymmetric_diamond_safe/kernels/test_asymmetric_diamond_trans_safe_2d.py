"""2-D asymmetric-diamond benchmark on tt.trans with no arithmetic on the
wide leg. Compared to asymmetric_diamond_trans_2d (which used arith.mulf
to keep the wide leg alive) this kernel drops mulf entirely, isolating
the diamond-gate behavior from any arithmetic cost.

Python source has reshape(reshape(trans)) on the wide leg as a no-op, but
Triton's frontend elides both reshapes -- the post-lowering IR for the
wide leg is just `tt.store %trans` directly. The diamond shape that
reaches the pass is therefore:

    load (fp32)
        |
      tt.trans (fp32)              <- DIAMOND HEAD (shared)
        |
   +----+----+
   |         |
 truncf    (tt.store, fp32)        <- reducer / direct sink
 (fp16)
   |
 store fp16

This is the minimal diamond: trans has two users, one reducer (truncf)
and one non-reducer (tt.store). The strict diamond gate at
DowncastReorderOptimizer.cpp:339-341 still bails because tt.store is not size-reducing.
ON and OFF IR are byte-identical (confirmed). See run_relaxed_sweep.sh for
the relaxed-gate variant that flips the bail and lets the diamond rewrite
fire.

CLI mirrors test_diamond_residual_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse
import hashlib


@triton.jit
def asymmetric_diamond_trans_safe_2d_kernel(
    ptr_in, ptr_out_narrow, ptr_out_wide,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    in_ptrs = ptr_in + rm[:, None] * N + rn[None, :]
    in_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tile = tl.load(in_ptrs, mask=in_mask, other=0.0).to(tl.float32)

    trans = tl.trans(tile)                            # (BLOCK_N, BLOCK_M) fp32  -- diamond head

    narrow = trans.to(tl.float16)                     # reducer: truncf
    wide_flat = tl.reshape(trans, (BLOCK_N * BLOCK_M,))  # non-reducer mover: reshape
    wide_2d = tl.reshape(wide_flat, (BLOCK_N, BLOCK_M))  # rehydrate for the store

    out_row = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_col = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_mask = (out_row[:, None] < N) & (out_col[None, :] < M)

    tl.store(ptr_out_narrow + out_row[:, None] * M + out_col[None, :],
             narrow, mask=out_mask)
    tl.store(ptr_out_wide + out_row[:, None] * M + out_col[None, :],
             wide_2d, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a


def verify(M, N, block_m, block_n):
    device = torch.device("cuda")
    src = _make_inputs(M, N, device)
    out_narrow = torch.empty(N, M, device=device, dtype=torch.float16)
    out_wide = torch.empty(N, M, device=device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    asymmetric_diamond_trans_safe_2d_kernel[grid](
        src, out_narrow, out_wide, M, N, BLOCK_M=block_m, BLOCK_N=block_n
    )
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(out_narrow.cpu().numpy().tobytes())
    h.update(out_wide.cpu().numpy().tobytes())
    h_n = hashlib.sha256(out_narrow.cpu().numpy().tobytes()).hexdigest()
    h_w = hashlib.sha256(out_wide.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({block_m},{block_n}) "
          f"sha256={h.hexdigest()} sha_narrow={h_n} sha_wide={h_w}")


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!"); return
    device = torch.device("cuda")

    src = torch.randn(M, N, device=device, dtype=torch.float32)
    dst_narrow = torch.empty(N, M, device=device, dtype=torch.float16)
    dst_wide = torch.empty(N, M, device=device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    for _ in range(3):
        asymmetric_diamond_trans_safe_2d_kernel[grid](
            src, dst_narrow, dst_wide, M, N, BLOCK_M=block_m, BLOCK_N=block_n
        )
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        asymmetric_diamond_trans_safe_2d_kernel[grid](
            src, dst_narrow, dst_wide, M, N, BLOCK_M=block_m, BLOCK_N=block_n
        )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"Completed {num_iterations} iterations for nsys profiling "
          f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D asymmetric-diamond trans (truncf + reshape) Triton benchmark")
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
        run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
