import argparse
import hashlib

import torch
import triton
import triton.language as tl


@triton.jit
def trans_trunc_fused_kernel(
    ptr_in, ptr_out,
    M, N,
    stride_in_m, stride_in_n,
    stride_out_n, stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    in_ptrs = ptr_in + rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
    mask_in = (rm[:, None] < M) & (rn[None, :] < N)
    tile = tl.load(in_ptrs, mask=mask_in, other=0.0)

    transposed = tl.trans(tile)
    out = transposed.to(tl.float16)

    out_ptrs = ptr_out + rn[:, None] * stride_out_n + rm[None, :] * stride_out_m
    mask_out = (rn[:, None] < N) & (rm[None, :] < M)
    tl.store(out_ptrs, out, mask=mask_out)


def launch(src_fp32, dst_fp16, BLOCK_M, BLOCK_N):
    M, N = src_fp32.shape
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    trans_trunc_fused_kernel[grid](
        src_fp32, dst_fp16,
        M, N,
        src_fp32.stride(0), src_fp32.stride(1),
        dst_fp16.stride(0), dst_fp16.stride(1),
        BLOCK_M, BLOCK_N,
    )


def sha256_of_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def run_correctness(M, N, BLOCK_M, BLOCK_N, seed):
    device = torch.device("cuda")
    torch.manual_seed(seed)

    src = torch.randn(M, N, device=device, dtype=torch.float32)
    dst = torch.empty(N, M, device=device, dtype=torch.float16)

    launch(src, dst, BLOCK_M, BLOCK_N)
    torch.cuda.synchronize()

    ref = src.T.contiguous().to(torch.float16)
    bit_exact = torch.equal(dst, ref)
    max_abs_diff = (dst.float() - ref.float()).abs().max().item()

    digest = sha256_of_tensor(dst)
    ref_digest = sha256_of_tensor(ref)

    print(f"M={M} N={N} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} seed={seed}")
    print(f"  bit-exact vs torch: {bit_exact}")
    print(f"  max |dst-ref|     : {max_abs_diff}")
    print(f"  sha256(dst)       : {digest}")
    print(f"  sha256(ref)       : {ref_digest}")
    return bit_exact, digest


def run_nsys(M, N, BLOCK_M, BLOCK_N, num_iterations, seed):
    device = torch.device("cuda")
    torch.manual_seed(seed)

    src = torch.randn(M, N, device=device, dtype=torch.float32)
    dst = torch.empty(N, M, device=device, dtype=torch.float16)

    for _ in range(3):
        launch(src, dst, BLOCK_M, BLOCK_N)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        launch(src, dst, BLOCK_M, BLOCK_N)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK={BLOCK_M}x{BLOCK_N}, fp32->fp16 trans+truncate)"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available!")

    parser = argparse.ArgumentParser(description="Trans+Truncate fused Triton kernel")
    parser.add_argument("--m", type=int, default=4096, help="rows of input matrix")
    parser.add_argument("--n", type=int, default=4096, help="cols of input matrix")
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100,
                        help="profiled iterations (nsys mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["correctness", "nsys"], default="correctness")
    args = parser.parse_args()

    if args.mode == "correctness":
        ok, _ = run_correctness(args.m, args.n, args.block_m, args.block_n, args.seed)
        raise SystemExit(0 if ok else 1)
    else:
        run_nsys(args.m, args.n, args.block_m, args.block_n, args.iterations, args.seed)
