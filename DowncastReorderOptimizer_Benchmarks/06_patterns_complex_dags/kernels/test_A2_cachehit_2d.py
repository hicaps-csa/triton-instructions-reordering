"""A2: SAME-KIND, SAME-TYPE FAN-OUT (cache-HIT path).

Important framing finding (documented during construction):

  The naive A2 pattern -- a single fp32 tt.cat consumed by two textual
  `c.to(tl.float16)` calls -- does NOT exercise DowncastReorderOptimizer's cache-hit
  path on real Triton. The frontend / canonicalize step folds the two
  identical `arith.truncf %c` ops into a single SSA value BEFORE
  DowncastReorderOptimizer runs, so the pass sees only ONE consumer of the cat. The
  applyMultiConsumerOptimization cache stores one entry, the lookup
  never hits.

  To actually drive the cache-hit code path we need TWO truncf ops with
  the SAME (OperationName, ResultType) but DIFFERENT operand SSA values
  so the frontend cannot fold them. This kernel inserts an extra mover
  layer between cat and each truncf: the two truncf ops then consume
  the results of TWO DIFFERENT reshape ops (different output shapes),
  which frontend CSE can't merge.

   load a (fp32)         load b (fp32)
       |                       |
   reshape                  reshape
       \\__________ cat ________/                 <- fp32 cat
                    |
        +-----------+-----------+
        |                       |
   reshape (2BM,BN)        reshape (BM,2BN)       <- two DIFFERENT reshapes
        |                       |
     truncf-fp16            truncf-fp16            <- two truncf-fp16 ops,
        |                       |                    SAME kind+type, DIFFERENT operands
     reshape (2BM,BN)       reshape (2BM,BN)
        |                       |
     store fp16              store fp16

Worklist trace:
  pop reshape #A (2BM,BN): user truncf-fp16. Push truncf above reshape.
    Cat becomes user of a cloned truncf-fp16 (#1).
  pop reshape #B (BM,2BN): user truncf-fp16. Push truncf above reshape.
    Cat becomes user of a cloned truncf-fp16 (#2).
  pop cat: users are #1 and #2 -- two ops with same (OpName, ResultType)
    cache key. CACHE MISS on first, CACHE HIT on second. ONE rewritten
    cat-on-fp16 shared by both branches.

The TTIR check: ON should have exactly ONE `tt.cat ... fp16`, NOT two.
If we see two, the cache hit failed.
"""
import torch, triton, triton.language as tl
import argparse, hashlib


@triton.jit
def A2_kernel(
    ptr_a, ptr_b, ptr_out1, ptr_out2,
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

    c = tl.cat(a_flat, b_flat, can_reorder=True)   # fp32

    # Two DIFFERENT-shape reshapes feeding two truncf-fp16 ops.
    # Different SSA operands -> frontend CSE cannot merge the two truncfs.
    c_r1 = tl.reshape(c, (2 * BLOCK_M, BLOCK_N))   # reshape #A
    c_r2 = tl.reshape(c, (BLOCK_M, 2 * BLOCK_N))   # reshape #B  (different shape)

    d1 = c_r1.to(tl.float16)        # truncf #1 (operand = c_r1)
    d2 = c_r2.to(tl.float16)        # truncf #2 (operand = c_r2)

    # Store both; reshape #B's output is rearranged into the same (2BM, BN) layout.
    out1 = d1
    out2 = tl.reshape(d2, (2 * BLOCK_M, BLOCK_N))

    out_row = pid_m * (2 * BLOCK_M) + tl.arange(0, 2 * BLOCK_M)
    out_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (out_row[:, None] < 2 * M) & (out_col[None, :] < N)
    tl.store(ptr_out1 + out_row[:, None] * N + out_col[None, :], out1, mask=out_mask)
    tl.store(ptr_out2 + out_row[:, None] * N + out_col[None, :], out2, mask=out_mask)


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
    A2_kernel[grid](a, b, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(out1.cpu().numpy().tobytes()); h.update(out2.cpu().numpy().tobytes())
    h1 = hashlib.sha256(out1.cpu().numpy().tobytes()).hexdigest()
    h2 = hashlib.sha256(out2.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({bm},{bn}) sha256={h.hexdigest()} "
          f"sha_out1={h1} sha_out2={h2}")


def run_benchmark_nsys(M, N, bm, bn, iters):
    device = torch.device("cuda")
    a = torch.randn(M, N, device=device, dtype=torch.float32)
    b = torch.randn(M, N, device=device, dtype=torch.float32)
    out1 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    out2 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, bm), triton.cdiv(N, bn))
    for _ in range(3):
        A2_kernel[grid](a, b, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        A2_kernel[grid](a, b, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
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
