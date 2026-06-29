"""A5: CAT-OF-CATS / NESTED MOVERS.

Three tt.cats in series, terminating in arith.truncf-fp16. Tests the
worklist iteration: starting from a SINGLE seed (the outer cat, which
is the only one with a reducer user), the worklist must propagate up
through TWO MORE cat layers in one runOnce sweep -- never falling back
to the outer driver's defensive iteration loop.

   load a (fp32)  load b   load c   load d
        \\        /          \\      /
         cat_ab                cat_cd       <- inner cats (no reducer users)
            \\                 /
              ____ cat_abcd ___              <- outer cat (only reducer user)
                       |
                    truncf-fp16
                       |
                    reshape
                       |
                    store fp16

Initial walk seeds ONLY cat_abcd (its user is truncf). Worklist trace:
  pop cat_abcd  -> rewrite truncf above cat_abcd. cat_abcd becomes cat_abcd-on-fp16.
                   upstream re-queue: cat_ab, cat_cd (both are safe movers feeding cat_abcd).
  pop cat_cd    -> its user is now the cloned truncf from above. Rewrite.
  pop cat_ab    -> its user is now the cloned truncf from above. Rewrite.

End state: 3 fp16 cats (cat_ab on fp16, cat_cd on fp16, cat_abcd on fp16);
1 fp16 cat_abcd in the IR. The fp32 cats are all gone. All this must
happen in a SINGLE runOnce call.
"""
import torch, triton, triton.language as tl
import argparse, hashlib


@triton.jit
def A5_kernel(
    ptr_a, ptr_b, ptr_c, ptr_d, ptr_out,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    a = tl.load(ptr_a + offs_m[:, None] * N + offs_n[None, :], mask=in_mask, other=0.0).to(tl.float32)
    b = tl.load(ptr_b + offs_m[:, None] * N + offs_n[None, :], mask=in_mask, other=0.0).to(tl.float32)
    c = tl.load(ptr_c + offs_m[:, None] * N + offs_n[None, :], mask=in_mask, other=0.0).to(tl.float32)
    d = tl.load(ptr_d + offs_m[:, None] * N + offs_n[None, :], mask=in_mask, other=0.0).to(tl.float32)

    a_flat = tl.reshape(a, (BLOCK_M * BLOCK_N,))
    b_flat = tl.reshape(b, (BLOCK_M * BLOCK_N,))
    c_flat = tl.reshape(c, (BLOCK_M * BLOCK_N,))
    d_flat = tl.reshape(d, (BLOCK_M * BLOCK_N,))

    ab = tl.cat(a_flat, b_flat, can_reorder=True)        # fp32 cat #1
    cd = tl.cat(c_flat, d_flat, can_reorder=True)        # fp32 cat #2
    abcd = tl.cat(ab, cd, can_reorder=True)              # fp32 cat #3 (outer)

    e = abcd.to(tl.float16)                              # truncf
    e2d = tl.reshape(e, (4 * BLOCK_M, BLOCK_N))

    out_row = pid_m * (4 * BLOCK_M) + tl.arange(0, 4 * BLOCK_M)
    out_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (out_row[:, None] < 4 * M) & (out_col[None, :] < N)
    tl.store(ptr_out + out_row[:, None] * N + out_col[None, :], e2d, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    b = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    c = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    d = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a, b, c, d


def verify(M, N, bm, bn):
    device = torch.device("cuda")
    a, b, c, d = _make_inputs(M, N, device)
    out = torch.empty(4 * M, N, device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    A5_kernel[grid](a, b, c, d, out, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    h = hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({bm},{bn}) sha256={h}")


def run_benchmark_nsys(M, N, bm, bn, iters):
    device = torch.device("cuda")
    a = torch.randn(M, N, device=device, dtype=torch.float32)
    b = torch.randn(M, N, device=device, dtype=torch.float32)
    c = torch.randn(M, N, device=device, dtype=torch.float32)
    d = torch.randn(M, N, device=device, dtype=torch.float32)
    out = torch.empty(4 * M, N, device=device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, bm), triton.cdiv(N, bn))
    for _ in range(3):
        A5_kernel[grid](a, b, c, d, out, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        A5_kernel[grid](a, b, c, d, out, M, N, BLOCK_M=bm, BLOCK_N=bn)
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
