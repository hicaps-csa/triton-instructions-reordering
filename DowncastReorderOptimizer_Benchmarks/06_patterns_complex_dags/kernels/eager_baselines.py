"""Torch-eager baselines for Experiments 7, 8, and A1-A7.

Each function below is the natural unfused-PyTorch equivalent of the
corresponding Triton kernel: every op launches its own kernel and
materializes its intermediate(s) in global memory. Timed with CUDA
events, 100 iterations + 10 warmup, at M=N in {4096, 8192}.

Run:  python eager_baselines.py
"""
import torch

DEVICE = torch.device("cuda")
WARMUP = 10
ITERS = 100
SIZES = [4096, 8192]


# ----------------------------------------------------------------------
# Eager equivalents.  Each takes pre-allocated inputs and returns the
# output tuple, mirroring the Triton kernel's outputs.
# ----------------------------------------------------------------------

def exp7_eager(a):
    # multi_consumer_trans_trunc: trans(fp32) -> {i8, fp16}
    t = a.t().contiguous()
    return t.to(torch.int8), t.to(torch.float16)


def exp8_eager(a):
    # asymmetric_diamond_trans: trans(fp32) -> {fp16, fp32 passthrough}
    t = a.t().contiguous()           # the fp32 (wide) output
    return t.to(torch.float16), t


def a1_eager(a, b):
    # 3-kind cat diamond: cat -> {i8, fp16, fp8e5}
    c = torch.cat([a, b], dim=0)
    return c.to(torch.int8), c.to(torch.float16), c.to(torch.float8_e5m2)


def a2_eager(a, b):
    # cache-hit: cat -> two fp16 outputs
    c = torch.cat([a, b], dim=0)
    return c.to(torch.float16), c.to(torch.float16)


def a3_eager(a, b):
    # mixed-kind: cat -> {i8 (signed), u8 (unsigned)}
    c = torch.cat([a, b], dim=0)
    return c.to(torch.int8), c.to(torch.uint8)


def a4_eager(a, b):
    # multi-level diamond: cat -> two fp16 outputs (one via transpose)
    c = torch.cat([a, b], dim=0)
    o1 = c.t().contiguous().to(torch.float16)
    o2 = c.to(torch.float16)
    return o1, o2


def a5_eager(a, b, c, d):
    # cat-of-cats: cat(cat(a,b), cat(c,d)) -> fp16
    ab = torch.cat([a, b], dim=0)
    cd = torch.cat([c, d], dim=0)
    abcd = torch.cat([ab, cd], dim=0)
    return (abcd.to(torch.float16),)


def a6_eager(a16, a32):
    # bitwidth guard: fp16->i16 path A, fp32->i16 path B
    return a16.to(torch.int16), a32.to(torch.int16)


def a7_eager(a):
    # Y-shape: shared fp32 load -> two fp16 outputs
    return a.to(torch.float16), a.to(torch.float16)


# ----------------------------------------------------------------------
# Timing harness
# ----------------------------------------------------------------------

def time_eager(fn, make_inputs, M, N):
    inputs = make_inputs(M, N)
    for _ in range(WARMUP):
        fn(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*inputs)
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    return ms_total / ITERS * 1000.0  # us per iteration


def f32(M, N):
    return (torch.randn(M, N, device=DEVICE, dtype=torch.float32),)

def f32x2(M, N):
    return (torch.randn(M, N, device=DEVICE, dtype=torch.float32),
            torch.randn(M, N, device=DEVICE, dtype=torch.float32))

def f32x4(M, N):
    return tuple(torch.randn(M, N, device=DEVICE, dtype=torch.float32)
                 for _ in range(4))

def f16_and_f32(M, N):
    return (torch.randn(M, N, device=DEVICE, dtype=torch.float16),
            torch.randn(M, N, device=DEVICE, dtype=torch.float32))


CASES = [
    ("Exp7",  exp7_eager, f32),
    ("Exp8",  exp8_eager, f32),
    ("A1",    a1_eager,   f32x2),
    ("A2",    a2_eager,   f32x2),
    ("A3",    a3_eager,   f32x2),
    ("A4",    a4_eager,   f32x2),
    ("A5",    a5_eager,   f32x4),
    ("A6",    a6_eager,   f16_and_f32),
    ("A7",    a7_eager,   f32),
]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!"); return
    print(f"{'case':<6} | {'M=N=4096 (us)':>15} | {'M=N=8192 (us)':>15}")
    print("-" * 44)
    for name, fn, mk in CASES:
        row = []
        for size in SIZES:
            us = time_eager(fn, mk, size, size)
            row.append(us)
        print(f"{name:<6} | {row[0]:>15.1f} | {row[1]:>15.1f}")


if __name__ == "__main__":
    main()
