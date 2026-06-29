"""2-D multi-consumer trans benchmark.

Tests the dedup cache in applyMultiConsumerOptimization when the shared
mover is tt.trans (rather than tt.cat as in dual_quant_2d). A single
tt.trans on fp32 feeds TWO size-reducing consumers of different kinds:
fptosi -> i8 and truncf -> fp16. The cache keyed on
(OperationName, Type) should produce TWO independent rewritten chains
(one per (kind, type) pair); after both rewrites the original fp32
tt.trans is dead (use_empty()) and is erased.

    load (fp32)
        |
      tt.trans (fp32)              <- SHARED mover
        |
   +----+----+
   |         |
 fptosi    truncf                  <- two reducers, different kinds
 (i8)      (fp16)
   |         |
 store i8  store f16

Note on kernel design: identical same-kind same-type reducers (e.g., two
.to(fp16) calls) get CSE-collapsed by Triton's frontend before the pass
sees the IR, so they cannot exercise the dedup cache. Different-kind
reducers stay distinct at TTIR level and DO exercise the multi-reducer-
kind branch of the cache.

Pass OFF: single fp32 tt.trans feeds both fptosi and truncf.
Pass ON : tt.trans of fp32 dies; one fptosi runs on the input tile and
          a fresh tt.trans on i8 stores to ptr_out_i8; symmetrically for
          truncf and fp16. Two parallel rewritten chains, no original
          trans left.

CLI mirrors test_diamond_residual_2d.py.
"""
import torch
import triton
import triton.language as tl
import argparse
import hashlib


@triton.jit
def multi_consumer_trans_trunc_2d_kernel(
    ptr_in, ptr_out_i8, ptr_out_f16,
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

    trans = tl.trans(tile)                          # (BLOCK_N, BLOCK_M) fp32  -- shared mover

    narrow_i8 = trans.to(tl.int8)                   # fptosi  reducer
    narrow_f16 = trans.to(tl.float16)               # truncf  reducer

    out_row = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_col = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_mask = (out_row[:, None] < N) & (out_col[None, :] < M)
    tl.store(ptr_out_i8 + out_row[:, None] * M + out_col[None, :],
             narrow_i8, mask=out_mask)
    tl.store(ptr_out_f16 + out_row[:, None] * M + out_col[None, :],
             narrow_f16, mask=out_mask)


def _make_inputs(M, N, device):
    g = torch.Generator(device="cpu").manual_seed(0xBEEFCAFE)
    a = torch.randn(M, N, generator=g, dtype=torch.float32).to(device)
    return a


def verify(M, N, block_m, block_n):
    device = torch.device("cuda")
    src = _make_inputs(M, N, device)
    out_i8 = torch.empty(N, M, device=device, dtype=torch.int8)
    out_f16 = torch.empty(N, M, device=device, dtype=torch.float16)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    multi_consumer_trans_trunc_2d_kernel[grid](
        src, out_i8, out_f16, M, N, BLOCK_M=block_m, BLOCK_N=block_n
    )
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

    src = torch.randn(M, N, device=device, dtype=torch.float32)
    dst_i8 = torch.empty(N, M, device=device, dtype=torch.int8)
    dst_f16 = torch.empty(N, M, device=device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    for _ in range(3):
        multi_consumer_trans_trunc_2d_kernel[grid](
            src, dst_i8, dst_f16, M, N, BLOCK_M=block_m, BLOCK_N=block_n
        )
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iterations):
        multi_consumer_trans_trunc_2d_kernel[grid](
            src, dst_i8, dst_f16, M, N, BLOCK_M=block_m, BLOCK_N=block_n
        )
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(
        f"Completed {num_iterations} iterations for nsys profiling "
        f"(M={M}, N={N}, BLOCK_M={block_m}, BLOCK_N={block_n})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-D multi-consumer trans (fptosi + truncf) Triton benchmark")
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
