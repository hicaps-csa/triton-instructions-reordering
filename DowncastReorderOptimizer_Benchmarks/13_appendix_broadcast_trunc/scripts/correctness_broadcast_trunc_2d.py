"""Correctness check for the 2-D broadcast_trunc_2d_kernel.

Compares the Triton output against a torch reference (broadcast then cast)
across a representative tile/grid set. Also reports the post-make_ttir
element type of the broadcast op so you can confirm whether DowncastReorderOptimizer
sank the cast (pass ON: tensor<...xf16>; pass OFF: tensor<...xf32>).

Run once with the pass ON and once with the pass OFF -- both should report
PASS for every config.
"""
import re
import torch
import triton
import triton.language as tl

from test_broadcast_trunc_2d import broadcast_trunc_2d_kernel


def _extract_mover_dtype(ttir: str, mover_op: str) -> str:
    for line in ttir.splitlines():
        if mover_op not in line:
            continue
        m = re.findall(r"tensor<([^>]+)>", line)
        if not m:
            continue
        shape_and_type = m[-1].split(",")[0]
        return shape_and_type.rsplit("x", 1)[-1].strip()
    return "?"


def run_one(M, N, block_m, block_n):
    device = torch.device("cuda")
    src = torch.randn(N, device=device, dtype=torch.float16).to(torch.float32)
    dst = torch.empty(M, N, device=device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    broadcast_trunc_2d_kernel[grid](src, dst, M, N,
                                    BLOCK_M=block_m, BLOCK_N=block_n)
    torch.cuda.synchronize()

    ref = src.unsqueeze(0).expand(M, N).to(torch.float16)
    max_err = (dst.float() - ref.float()).abs().max().item()
    ok = max_err == 0.0

    cache = next(iter(broadcast_trunc_2d_kernel.device_caches.values()))[0]
    compiled = next(iter(cache.values()))
    ttir = compiled.asm.get("ttir", "")
    mover_dtype = _extract_mover_dtype(ttir, "tt.broadcast")

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] M={M:<5} N={N:<5} ({block_m:>3}, {block_n:>3}) "
          f"max_err={max_err}  tt.broadcast out=tensor<...x{mover_dtype}>")
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        raise SystemExit(1)

    print("broadcast_trunc_2d correctness")
    print("-" * 80)
    configs = [
        (512, 512, 64, 64),
        (1024, 1024, 64, 128),
        (1024, 1024, 128, 64),
        (2048, 2048, 128, 128),
        (4096, 4096, 128, 256),
        (4096, 4096, 256, 256),
    ]
    results = [run_one(*c) for c in configs]
    n_pass = sum(results)
    print("-" * 80)
    print(f"{n_pass}/{len(results)} configs passed")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
