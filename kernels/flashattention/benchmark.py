from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
INCLUDE_DIR = REPO_ROOT / "include"

BATCH_SIZE = 1
NUM_HEADS = 32
QUERY_LEN = 4096
KEY_LEN = 4096
HEAD_DIM = 64
WARMUP = 2
REPEAT = 5
VERIFY = True

KERNELS = [
    ("naive", "flashattention_naive.cu", "flash_attention", False),
    ("mma", "flashattention_mma.cu", "flash_attention_mma", True),
]


@lru_cache(maxsize=None)
def load_kernel(name: str, source: str, symbol: str, use_half: bool):
    module_name = f"cuda_kernel_bench_flashattention_{name}"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    flags = [f"-DKERNEL_SYMBOL={symbol}"]
    if use_half:
        flags.append("-DUSE_HALF_INPUT")
    common_flags = ["-O3", "-std=c++17", *flags]
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / "flashattention_single_binding.cpp"), str(KERNEL_DIR / source)],
        extra_include_paths=[str(INCLUDE_DIR), str(KERNEL_DIR)],
        extra_cflags=common_flags,
        extra_cuda_cflags=common_flags,
        build_directory=str(build_dir),
        verbose=False,
    )


def make_launch(name: str, source: str, symbol: str, use_half: bool) -> Callable[..., None]:
    cache = {}

    def launch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor) -> None:
        if "fn" not in cache:
            cache["fn"] = getattr(load_kernel(name, source, symbol, use_half), symbol)
        cache["fn"](q, k, v, out)

    return launch


def make_input(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    numel = math.prod(shape)
    return torch.arange(numel, dtype=torch.float32).reshape(shape).add(offset).remainder(257).sub(128).mul(0.0078125)


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v.float()).to(dtype=q.dtype)


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float) -> bool:
    if torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
        return True
    mismatch = (~torch.isclose(lhs, rhs, atol=atol, rtol=rtol)).flatten().nonzero()[0].item()
    print(f"Mismatch at index {mismatch}: lhs={lhs.flatten()[mismatch].item():.6f} rhs={rhs.flatten()[mismatch].item():.6f}")
    return False


def measure_ms(launch: Callable[[], None], prepare: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(REPEAT):
        prepare()
        torch.cuda.synchronize()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best


def run_impl(name: str, backend: str, launch: Callable[[], None], prepare: Callable[[], None], verify: Callable[[], bool], num_bytes: float) -> bool:
    for _ in range(WARMUP):
        prepare()
        launch()
        torch.cuda.synchronize()
    best_ms = measure_ms(launch, prepare)
    passed = None
    if VERIFY:
        prepare()
        launch()
        torch.cuda.synchronize()
        passed = verify()
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    attention_scores = float(BATCH_SIZE * NUM_HEADS * QUERY_LEN * KEY_LEN)
    ops = (attention_scores * float(HEAD_DIM) * 4.0 + attention_scores * 5.0) / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[flashattention/{backend}:{name}] best={best_ms:.4f} ms, {ops:.3f} GOP/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    q_shape = (BATCH_SIZE, NUM_HEADS, QUERY_LEN, HEAD_DIM)
    kv_shape = (BATCH_SIZE, NUM_HEADS, KEY_LEN, HEAD_DIM)
    base_q = make_input(q_shape, 0).cuda()
    base_k = make_input(kv_shape, 17).cuda()
    base_v = make_input(kv_shape, 31).cuda()

    implementations = [
        (name, "cuda", make_launch(name, source, symbol, use_half), use_half)
        for name, source, symbol, use_half in KERNELS
    ]
    implementations.append(("torch", "pytorch", lambda q, k, v, out: out.copy_(reference(q, k, v)), False))

    all_passed = True
    for name, backend, fn, use_half in implementations:
        dtype = torch.float16 if use_half else torch.float32
        q = base_q.to(dtype=dtype)
        k = base_k.to(dtype=dtype)
        v = base_v.to(dtype=dtype)
        out = torch.empty_like(q)
        ref = reference(q, k, v) if VERIFY else None
        atol = 2e-3 if dtype == torch.float16 else 1e-4
        rtol = 2e-3 if dtype == torch.float16 else 1e-4
        num_bytes = float(q.numel() + k.numel() + v.numel() + out.numel()) * q.element_size()

        passed = run_impl(
            name,
            backend,
            lambda fn=fn: fn(q, k, v, out),
            lambda: out.zero_(),
            lambda ref=ref, atol=atol, rtol=rtol: True if ref is None else compare_tensors(out, ref, atol, rtol),
            num_bytes,
        )
        all_passed = passed and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
