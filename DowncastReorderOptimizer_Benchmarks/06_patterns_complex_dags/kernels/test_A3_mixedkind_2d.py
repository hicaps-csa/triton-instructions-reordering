"""A3: MIXED-KIND, SAME-TYPE FAN-OUT.

One fp32 tt.cat feeds TWO consumers with DIFFERENT OperationName but
nominally the SAME result type:
  - arith.fptosi to int8
  - arith.fptoui to uint8

Question A3 asks: in Triton TTIR, do these have the SAME MLIR result
type (both `tensor<...xi8>` if signless), or different (signed vs
unsigned). If same, the dedup cache (which keys on (OperationName,
ResultType)) ends up with two entries because OpName differs -- two
rewritten chains. Confirms that the current policy ("separate chains
for different kinds") is what we observe; the perf row tells us whether
that policy is leaving free perf on the table by NOT sharing chains.

   load a (fp32)         load b (fp32)
       |                       |
   reshape                  reshape
       \\__________ cat ________/
                    |
            +-------+-------+
            |               |
         fptosi           fptoui          <- diff kinds, same/related dest
         (i8)             (u8)
            |               |
         store i8         store u8
"""
import torch, triton, triton.language as tl
import argparse, hashlib


@triton.jit
def A3_kernel(
    ptr_a, ptr_b, ptr_out_i8, ptr_out_u8,
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

    d_i8 = c.to(tl.int8)         # arith.fptosi
    d_u8 = c.to(tl.uint8)        # arith.fptoui

    d_i8_2d = tl.reshape(d_i8, (2 * BLOCK_M, BLOCK_N))
    d_u8_2d = tl.reshape(d_u8, (2 * BLOCK_M, BLOCK_N))

    out_row = pid_m * (2 * BLOCK_M) + tl.arange(0, 2 * BLOCK_M)
    out_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (out_row[:, None] < 2 * M) & (out_col[None, :] < N)
    tl.store(ptr_out_i8 + out_row[:, None] * N + out_col[None, :], d_i8_2d, mask=out_mask)
    tl.store(ptr_out_u8 + out_row[:, None] * N + out_col[None, :], d_u8_2d, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    b = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a, b


def verify(M, N, bm, bn):
    device = torch.device("cuda")
    a, b = _make_inputs(M, N, device)
    out_i8 = torch.empty(2 * M, N, device=device, dtype=torch.int8)
    out_u8 = torch.empty(2 * M, N, device=device, dtype=torch.uint8)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    A3_kernel[grid](a, b, out_i8, out_u8, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(out_i8.cpu().numpy().tobytes()); h.update(out_u8.cpu().numpy().tobytes())
    h1 = hashlib.sha256(out_i8.cpu().numpy().tobytes()).hexdigest()
    h2 = hashlib.sha256(out_u8.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({bm},{bn}) sha256={h.hexdigest()} "
          f"sha_i8={h1} sha_u8={h2}")


def run_benchmark_nsys(M, N, bm, bn, iters):
    device = torch.device("cuda")
    a = torch.randn(M, N, device=device, dtype=torch.float32)
    b = torch.randn(M, N, device=device, dtype=torch.float32)
    out_i8 = torch.empty(2 * M, N, device=device, dtype=torch.int8)
    out_u8 = torch.empty(2 * M, N, device=device, dtype=torch.uint8)
    grid = lambda meta: (triton.cdiv(M, bm), triton.cdiv(N, bn))
    for _ in range(3):
        A3_kernel[grid](a, b, out_i8, out_u8, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        A3_kernel[grid](a, b, out_i8, out_u8, M, N, BLOCK_M=bm, BLOCK_N=bn)
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
