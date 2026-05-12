from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
INCLUDE_DIR = REPO_ROOT / "include"

TOKENS = 4096
HIDDEN = 8192
EPS = 1e-6
WARMUP = 2
REPEAT = 5
VERIFY = True

FLOAT_KERNELS = [
    ("naive", "rms_naive.cu", "rms_naive", "rms_single_binding.cpp", []),
    ("naive_v2", "rms_naive_v2.cu", "rms_naive_v2", "rms_single_binding.cpp", []),
    ("shared_memory", "rms_shared_memory.cu", "rms_shared_memory", "rms_single_binding.cpp", ["-DREQUIRE_SHARED_MEMORY_HIDDEN_SIZE"]),
]
HALF_KERNELS = [
    ("half", "rms_half.cu", "rms_half", "rms_half_single_binding.cpp", []),
    ("half2", "rms_half2.cu", "rms_half2", "rms_half_single_binding.cpp", []),
]


@lru_cache(maxsize=None)
def load_kernel(name: str, source: str, symbol: str, binding: str, flags: tuple[str, ...]):
    module_name = f"cuda_kernel_bench_rmsnorm_{name}"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    common_flags = ["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}", *flags]
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / binding), str(KERNEL_DIR / source)],
        extra_include_paths=[str(INCLUDE_DIR), str(KERNEL_DIR)],
        extra_cflags=common_flags,
        extra_cuda_cflags=common_flags,
        build_directory=str(build_dir),
        verbose=False,
    )


def make_launch(name: str, source: str, symbol: str, binding: str, flags: list[str]) -> Callable[..., None]:
    cache = {}

    def launch(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float) -> None:
        if "fn" not in cache:
            cache["fn"] = getattr(load_kernel(name, source, symbol, binding, tuple(flags)), symbol)
        cache["fn"](x, weight, out, eps)

    return launch


def make_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    host_x = torch.arange(TOKENS * HIDDEN, dtype=torch.float32).reshape(TOKENS, HIDDEN).remainder(97).mul(0.01)
    host_w = torch.arange(HIDDEN, dtype=torch.float32).remainder(13).mul(0.1)
    return host_x.cuda().to(dtype), host_w.cuda().to(dtype)


def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    wf = weight.float()
    denom = torch.sqrt((xf * xf).mean(dim=1, keepdim=True) + EPS)
    return ((xf / denom) * wf).to(dtype=x.dtype)


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float) -> bool:
    if torch.allclose(lhs, rhs, atol=atol, rtol=rtol):
        return True
    mismatch = (~torch.isclose(lhs, rhs, atol=atol, rtol=rtol)).flatten().nonzero()[0].item()
    print(f"Mismatch at index {mismatch}: lhs={lhs.flatten()[mismatch].item():.6f} rhs={rhs.flatten()[mismatch].item():.6f}")
    return False


def measure_ms(launch: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best


def run_impl(label: str, name: str, backend: str, launch: Callable[[], None], verify: Callable[[], bool], num_bytes: float) -> bool:
    for _ in range(WARMUP):
        launch()
        torch.cuda.synchronize()
    best_ms = measure_ms(launch)
    passed = None
    if VERIFY:
        launch()
        torch.cuda.synchronize()
        passed = verify()
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    ops = float(TOKENS * HIDDEN) / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[rmsnorm/{label}/{backend}:{name}] best={best_ms:.4f} ms, {ops:.3f} GOP/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def benchmark_path(label: str, dtype: torch.dtype, specs: list[tuple[str, str, str, str, list[str]]], atol: float, rtol: float) -> bool:
    x, weight = make_inputs(dtype)
    out = torch.empty_like(x)
    ref = reference(x, weight) if VERIFY else None
    num_bytes = float(TOKENS * HIDDEN) * x.element_size() * 3.0

    implementations = [
        (name, "cuda", make_launch(name, source, symbol, binding, flags))
        for name, source, symbol, binding, flags in specs
    ]
    implementations.append((f"torch_{label}", "pytorch", lambda a, w, o, eps: o.copy_(reference(a, w))))

    all_passed = True
    for name, backend, fn in implementations:
        passed = run_impl(
            label,
            name,
            backend,
            lambda fn=fn: fn(x, weight, out, EPS),
            lambda: True if ref is None else compare_tensors(out, ref, atol, rtol),
            num_bytes,
        )
        all_passed = passed and all_passed
    return all_passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    all_passed = benchmark_path("float", torch.float32, FLOAT_KERNELS, 1e-5, 1e-5)
    all_passed = benchmark_path("half", torch.float16, HALF_KERNELS, 1e-2, 1e-2) and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
