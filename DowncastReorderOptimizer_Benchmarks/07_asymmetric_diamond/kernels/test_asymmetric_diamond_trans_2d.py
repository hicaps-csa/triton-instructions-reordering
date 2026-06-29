"""2-D asymmetric-diamond benchmark on tt.trans.

Builds a diamond at the trans output where one user is a size-reducer
(truncf -> fp16) and the other is a non-reducer (mulf, kept in fp32).
Distinct from diamond_residual_2d (which sits on tt.cat) and dual_quant_2d
(both users are reducers, distinct kinds). This kernel measures whether
DAG-style diamond handling regresses on a mover that was empirically
neutral in the single-consumer case (Exp 2, trans+truncf).

    load (fp32)
        |
      tt.trans (fp32)              <- DIAMOND HEAD (shared mover)
        |
   +----+----+
   |         |
 truncf    mulf (* 0.5)            <- reducer / non-reducer
   |         |
 store fp16 store fp32

Pass OFF: single fp32 tt.trans feeds both users.
Pass ON : DAG extension A clones the reducer chain backward through trans,
          producing a tt.trans on fp16. Because mulf still consumes the
          original fp32 trans, use_empty() is false and the original
          tt.trans is kept alive. Both trans ops coexist post-pass.

The cost-of-duplication regime: now the kernel materializes TWO tt.trans
ops on the same tile, where the single-consumer version was already
empirically neutral. Most likely outcome to watch for: ON regresses
relative to OFF at tile sizes where trans duplication doesn't fit.

CLI mirrors test_diamond_residual_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse
import hashlib


@triton.jit
def asymmetric_diamond_trans_2d_kernel(
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

    trans = tl.trans(tile)                       # (BLOCK_N, BLOCK_M) fp32  -- diamond head

    narrow = trans.to(tl.float16)                # reducer (truncf)
    wide = trans * 0.5                           # non-reducer (mulf, stays fp32)

    out_row = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_col = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_mask = (out_row[:, None] < N) & (out_col[None, :] < M)

    tl.store(ptr_out_narrow + out_row[:, None] * M + out_col[None, :],
             narrow, mask=out_mask)
    tl.store(ptr_out_wide + out_row[:, None] * M + out_col[None, :],
             wide, mask=out_mask)


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
    asymmetric_diamond_trans_2d_kernel[grid](
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
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src = torch.randn(M, N, device=device, dtype=torch.float32)
    dst_narrow = torch.empty(N, M, device=device, dtype=torch.float16)
    dst_wide = torch.empty(N, M, device=device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        asymmetric_diamond_trans_2d_kernel[grid](
            src, dst_narrow, dst_wide, M, N, BLOCK_M=block_m, BLOCK_N=block_n
        )
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        asymmetric_diamond_trans_2d_kernel[grid](
            src, dst_narrow, dst_wide, M, N, BLOCK_M=block_m, BLOCK_N=block_n
        )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D asymmetric-diamond trans Triton benchmark")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--verify", action="store_true",
                        help="run once with deterministic input, print sha256 of combined outputs")
    args = parser.parse_args()

    if args.verify:
        verify(args.M, args.N, args.block_m, args.block_n)
    else:
        run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
