"""2-D dual-precision-quantize benchmark (case #3).

Builds a diamond at the cat output where BOTH users are size-reducing:

    load a (fp32)              load b (fp32)
        |                          |
     reshape                    reshape         <- movers
        |____________ cat ________|             <- DIAMOND HEAD (1-D fp32)
                       |
              +--------+--------+
              |                 |
           fptosi -> i8       truncf -> f16     <- two reducers, distinct kinds
              |                 |
            reshape           reshape
              |                 |
           store i8          store f16

Pass OFF: a single fp32 cat feeds both reducers.
Pass ON : DAG extension A clones both reducer chains backward through the
          movers. Because both users are size-reducing, the guard does NOT
          bail. The dedup cache keyed on (OperationName, Type) distinguishes
          (fptosi, i8) from (truncf, f16), so we get TWO rewritten chains.
          The original fp32 cat dies (use_empty after rewrites).

CLI mirrors test_diamond_residual_2d.py: --M --N --block-m --block-n
--iterations plus --verify for deterministic seed + sha256 over both outputs.
"""
import torch
import triton
import triton.language as tl
import argparse
import hashlib


@triton.jit
def dual_quant_2d_kernel(
    ptr_a, ptr_b, ptr_out_i8, ptr_out_f16,
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

    c = tl.cat(a_flat, b_flat, can_reorder=True)        # diamond head (1-D fp32)

    d_i8 = c.to(tl.int8)                                 # fptosi reducer
    d_i8_2d = tl.reshape(d_i8, (2 * BLOCK_M, BLOCK_N))

    d_f16 = c.to(tl.float16)                             # truncf reducer
    d_f16_2d = tl.reshape(d_f16, (2 * BLOCK_M, BLOCK_N))

    out_row = pid_m * (2 * BLOCK_M) + tl.arange(0, 2 * BLOCK_M)
    out_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (out_row[:, None] < 2 * M) & (out_col[None, :] < N)

    tl.store(ptr_out_i8 + out_row[:, None] * N + out_col[None, :],
             d_i8_2d, mask=out_mask)
    tl.store(ptr_out_f16 + out_row[:, None] * N + out_col[None, :],
             d_f16_2d, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    b = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a, b


def verify(M, N, block_m, block_n):
    device = torch.device("cuda")
    a, b = _make_inputs(M, N, device)
    out_i8 = torch.empty(2 * M, N, device=device, dtype=torch.int8)
    out_f16 = torch.empty(2 * M, N, device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    dual_quant_2d_kernel[grid](a, b, out_i8, out_f16, M, N,
                               BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    h = hashlib.sha256()
    h.update(out_i8.cpu().numpy().tobytes())
    h.update(out_f16.cpu().numpy().tobytes())
    h_i8 = hashlib.sha256(out_i8.cpu().numpy().tobytes()).hexdigest()
    h_f16 = hashlib.sha256(out_f16.cpu().numpy().tobytes()).hexdigest()
    print(f"verify M={M} N={N} BLOCK=({block_m},{block_n}) "
          f"sha256={h.hexdigest()} sha_i8={h_i8} sha_f16={h_f16}")


def run_benchmark_nsys(M, N, block_m, block_n, num_iterations):
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    device = torch.device("cuda")

    src_a = torch.randn(M, N, device=device, dtype=torch.float32)
    src_b = torch.randn(M, N, device=device, dtype=torch.float32)
    dst_i8 = torch.empty(2 * M, N, device=device, dtype=torch.int8)
    dst_f16 = torch.empty(2 * M, N, device=device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        dual_quant_2d_kernel[grid](src_a, src_b, dst_i8, dst_f16, M, N,
                                   BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        dual_quant_2d_kernel[grid](src_a, src_b, dst_i8, dst_f16, M, N,
                                   BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D dual-precision-quantize Triton benchmark")
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
