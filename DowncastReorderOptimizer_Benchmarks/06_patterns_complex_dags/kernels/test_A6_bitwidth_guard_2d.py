"""A6: BITWIDTH-EQUAL GUARD TEST.

The pass's per-operand bitwidth guard (eligibleReducers, DowncastReorderOptimizer.cpp:
356-372) requires every producer operand's element-type bitwidth to be
STRICTLY GREATER than the consumer's destination bitwidth, otherwise the
consumer is skipped:

    elemTy.getIntOrFloatBitWidth() <= dstType.getElementTypeBitWidth()
        -> allOperandsReduce = false; break;

A6 as originally framed (cat of mixed-bitwidth operands) cannot be
expressed in Triton because all cat operands must share a dtype. Instead
we test the same guard via a single-operand mover (reshape) with an
operand whose bitwidth EQUALS the destination -- which is the boundary
case `width <= dst_width`.

This kernel runs TWO independent paths in the same launch:

  Path A (guard FAILS):
    load fp16 -> reshape (mover, 16-bit) -> fptosi -> i16 (16-bit)
    Operand bitwidth 16 == dest 16. Strictly-greater check fails.
    -> Pass SHOULD NOT rewrite. Expect ON IR identical to OFF IR for
       this chain.

  Path B (guard PASSES, control case):
    load fp32 -> reshape (mover, 32-bit) -> fptosi -> i16 (16-bit)
    Operand bitwidth 32 > 16. Guard passes.
    -> Pass SHOULD rewrite. Expect ON IR with fptosi pushed above
       reshape on this chain.

Both paths produce i16 stores so the verifier checks both at once. The
TTIR diff between OFF and ON should show ONE rewrite (Path B), not two.
"""
import torch, triton, triton.language as tl
import argparse, hashlib


@triton.jit
def A6_kernel(
    ptr_a_f16, ptr_a_f32,
    ptr_out_a, ptr_out_b,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    in_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Path A: fp16 input -> reshape -> fptosi-i16  (GUARD FAILS, 16 == 16)
    a16 = tl.load(ptr_a_f16 + offs_m[:, None] * N + offs_n[None, :],
                  mask=in_mask, other=0.0).to(tl.float16)
    a16_flat = tl.reshape(a16, (BLOCK_M * BLOCK_N,))   # safe mover, operand fp16
    pa = a16_flat.to(tl.int16)                          # fptosi to i16
    pa_2d = tl.reshape(pa, (BLOCK_M, BLOCK_N))

    # Path B: fp32 input -> reshape -> fptosi-i16  (GUARD PASSES, 32 > 16)
    a32 = tl.load(ptr_a_f32 + offs_m[:, None] * N + offs_n[None, :],
                  mask=in_mask, other=0.0).to(tl.float32)
    a32_flat = tl.reshape(a32, (BLOCK_M * BLOCK_N,))   # safe mover, operand fp32
    pb = a32_flat.to(tl.int16)                          # fptosi to i16
    pb_2d = tl.reshape(pb, (BLOCK_M, BLOCK_N))

    tl.store(ptr_out_a + offs_m[:, None] * N + offs_n[None, :], pa_2d, mask=in_mask)
    tl.store(ptr_out_b + offs_m[:, None] * N + offs_n[None, :], pb_2d, mask=in_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a16 = torch.randn(M, N, generator=g, dtype=torch.float32).to(device).to(torch.float16)
    a32 = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a16, a32


def verify(M, N, bm, bn):
    device = torch.device("cuda")
    a16, a32 = _make_inputs(M, N, device)
    outa = torch.empty(M, N, device=device, dtype=torch.int16)
    outb = torch.empty(M, N, device=device, dtype=torch.int16)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    A6_kernel[grid](a16, a32, outa, outb, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(outa.cpu().numpy().tobytes()); h.update(outb.cpu().numpy().tobytes())
    h1 = hashlib.sha256(outa.cpu().numpy().tobytes()).hexdigest()
    h2 = hashlib.sha256(outb.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({bm},{bn}) sha256={h.hexdigest()} "
          f"sha_pathA_fail={h1} sha_pathB_pass={h2}")


def run_benchmark_nsys(M, N, bm, bn, iters):
    device = torch.device("cuda")
    a16 = torch.randn(M, N, device=device, dtype=torch.float16)
    a32 = torch.randn(M, N, device=device, dtype=torch.float32)
    outa = torch.empty(M, N, device=device, dtype=torch.int16)
    outb = torch.empty(M, N, device=device, dtype=torch.int16)
    grid = lambda meta: (triton.cdiv(M, bm), triton.cdiv(N, bn))
    for _ in range(3):
        A6_kernel[grid](a16, a32, outa, outb, M, N, BLOCK_M=bm, BLOCK_N=bn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        A6_kernel[grid](a16, a32, outa, outb, M, N, BLOCK_M=bm, BLOCK_N=bn)
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
