"""A7: Y-SHAPE UPSTREAM.

A SHARED fp32 source (the loaded tensor) feeds TWO independent
mover -> truncf chains. The pass rewrites each chain independently,
which means BOTH cloned truncf ops end up consuming the same SHARED
fp32 value -- producing two SSA-distinct truncf ops with identical
operands and result types.

    load a (fp32)  <- shared SSA value with TWO users below
        |
        +-------+----------+
        |                  |
     trans (2D)        reshape (2D->1D)         <- two different movers
        |                  |
     truncf-fp16        truncf-fp16
        |                  |
     trans (back)       reshape (->2D)
        |                  |
     store fp16         store fp16

After pass: each cloned truncf goes on the load result directly:

    load a (fp32)
        |
        +-------+----------+
        |                  |
     truncf-fp16        truncf-fp16        <- redundant! two identical truncfs
        |                  |
     trans              reshape
        |                  |
     ...                ...

The test: do downstream MLIR / TritonGPU passes (CSE, canonicalize) de-
duplicate the two truncfs once both consume the same %a? Inspect ttgir
and llir to confirm only ONE conversion shows up in the final SASS.
"""
import torch, triton, triton.language as tl
import argparse, hashlib


@triton.jit
def A7_kernel(
    ptr_a, ptr_out_trans, ptr_out_reshape,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # SHARED fp32 load
    a = tl.load(ptr_a + offs_m[:, None] * N + offs_n[None, :],
                mask=in_mask, other=0.0).to(tl.float32)

    # Branch 1: trans -> truncf
    a_t = tl.trans(a, 1, 0)                       # mover1 (trans)
    d1 = a_t.to(tl.float16)                        # truncf
    out1 = tl.trans(d1, 1, 0)                      # back to (M, N) for store

    # Branch 2: reshape -> truncf
    a_r = tl.reshape(a, (BLOCK_M * BLOCK_N,))      # mover2 (reshape)
    d2 = a_r.to(tl.float16)                        # truncf
    out2 = tl.reshape(d2, (BLOCK_M, BLOCK_N))

    tl.store(ptr_out_trans + offs_m[:, None] * N + offs_n[None, :], out1, mask=in_mask)
    tl.store(ptr_out_reshape + offs_m[:, None] * N + offs_n[None, :], out2, mask=in_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a


def verify(M, N, bm, bn):
    device = torch.device("cuda")
    a = _make_inputs(M, N, device)
    out1 = torch.empty(M, N, device=device, dtype=torch.float16)
    out2 = torch.empty(M, N, device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    A7_kernel[grid](a, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
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
    out1 = torch.empty(M, N, device=device, dtype=torch.float16)
    out2 = torch.empty(M, N, device=device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, bm), triton.cdiv(N, bn))
    for _ in range(3):
        A7_kernel[grid](a, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        A7_kernel[grid](a, out1, out2, M, N, BLOCK_M=bm, BLOCK_N=bn)
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
