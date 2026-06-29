"""A4: MULTI-LEVEL DIAMOND.

One fp32 tt.cat fans out into two branches, EACH of which contains MORE
movers before terminating in arith.truncf-fp16:

    load a (fp32)         load b (fp32)
        |                       |
    reshape                  reshape
        \\__________ cat ________/
                     |
            +--------+--------+
            |                 |
         reshape           reshape          <- another layer of movers
         (2D shape A)      (2D shape B)
            |                 |
         trans              (truncf-fp16)
            |                 |
         truncf-fp16        reshape
            |                 |
         reshape            store fp16
            |
         store fp16

Tests:
  (1) Worklist propagation correctly re-queues cat AFTER each branch's
      mover has been rewritten. Initial walk seeds only the inner-most
      movers whose users are truncfs. Cat itself has NON-reducer users
      (reshape, reshape) at walk time, so it is NOT seeded; it can only
      become eligible after the lower movers are rewritten.

  (2) When cat becomes eligible, both its users are arith.truncf-fp16
      ops (cloned during the lower-mover rewrites). They are SAME-KIND,
      SAME-TYPE -- the cache MUST hit at cat, producing ONE rewritten
      cat-on-fp16 shared by both branches. This is a cache-hit
      consequence of multi-level propagation, distinct from A2's direct
      same-kind fan-out.

  (3) Cat erasure: original fp32 cat must be erased once both upstream
      uses are redirected. use_empty() must be true.
"""
import torch, triton, triton.language as tl
import argparse, hashlib


@triton.jit
def A4_kernel(
    ptr_a, ptr_b, ptr_out_trans, ptr_out_reshape,
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

    c = tl.cat(a_flat, b_flat, can_reorder=True)   # fp32, len = 2*BLOCK_M*BLOCK_N

    # Branch 1: cat -> reshape(2BM, BN) -> trans -> truncf -> reshape(2BM, BN)
    c1 = tl.reshape(c, (2 * BLOCK_M, BLOCK_N))
    c1_t = tl.trans(c1, 1, 0)
    d1 = c1_t.to(tl.float16)
    out1 = tl.trans(d1, 1, 0)         # back to (2BM, BN)

    # Branch 2: cat -> reshape(BM, 2BN) -> truncf -> reshape(2BM, BN)
    c2 = tl.reshape(c, (BLOCK_M, 2 * BLOCK_N))
    d2 = c2.to(tl.float16)
    out2 = tl.reshape(d2, (2 * BLOCK_M, BLOCK_N))

    out_row = pid_m * (2 * BLOCK_M) + tl.arange(0, 2 * BLOCK_M)
    out_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (out_row[:, None] < 2 * M) & (out_col[None, :] < N)
    tl.store(ptr_out_trans + out_row[:, None] * N + out_col[None, :], out1, mask=out_mask)
    tl.store(ptr_out_reshape + out_row[:, None] * N + out_col[None, :], out2, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    b = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a, b


def verify(M, N, bm, bn):
    device = torch.device("cuda")
    a, b = _make_inputs(M, N, device)
    out1 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    out2 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    A4_kernel[grid](a, b, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(out1.cpu().numpy().tobytes()); h.update(out2.cpu().numpy().tobytes())
    h1 = hashlib.sha256(out1.cpu().numpy().tobytes()).hexdigest()
    h2 = hashlib.sha256(out2.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({bm},{bn}) sha256={h.hexdigest()} "
          f"sha_trans={h1} sha_reshape={h2}")


def run_benchmark_nsys(M, N, bm, bn, iters):
    device = torch.device("cuda")
    a = torch.randn(M, N, device=device, dtype=torch.float32)
    b = torch.randn(M, N, device=device, dtype=torch.float32)
    out1 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    out2 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, bm), triton.cdiv(N, bn))
    for _ in range(3):
        A4_kernel[grid](a, b, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        A4_kernel[grid](a, b, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(f"Completed {iters} iterations (M={M}, N={N}, BLOCK_M={bm}, BLOCK_N={bn})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--block-m", type=int, default=128)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.verify:
        verify(args.M, args.N, args.block_m, args.block_n)
    else:
        run_benchmark_nsys(args.M, args.N, args.block_m, args.block_n, args.iterations)
