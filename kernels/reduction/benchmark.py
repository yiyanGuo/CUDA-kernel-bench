from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
INCLUDE_DIR = REPO_ROOT / "include"

N = 1 << 24
WARMUP = 2
REPEAT = 5
VERIFY = True

KERNELS = [
    ("naive", ["reduction_naive.cu"], "reduction_naive"),
    ("presum", ["reduction_presum.cu"], "reduction_presum"),
    ("presum_float4", ["reduction_presum_float4.cu", "reduction_presum.cu"], "reduction_presum_float4"),
    ("shuffle", ["reduction_shuffle.cu"], "reduction_shuffle"),
    ("grid_stride", ["reduction_grid_stride.cu"], "reduction_grid_stride"),
    ("integrate", ["reduction_integrate.cu"], "reduction_integrate"),
]


@lru_cache(maxsize=None)
def load_kernel(name: str, sources: tuple[str, ...], symbol: str):
    module_name = f"cuda_kernel_bench_reduction_{name}"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / "reduction_single_binding.cpp"), *[str(KERNEL_DIR / source) for source in sources]],
        extra_include_paths=[str(INCLUDE_DIR), str(KERNEL_DIR)],
        extra_cflags=["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}"],
        extra_cuda_cflags=["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}"],
        build_directory=str(build_dir),
        verbose=False,
    )


def make_launch(name: str, sources: list[str], symbol: str) -> Callable[[torch.Tensor, torch.Tensor], None]:
    cache = {}

    def launch(x: torch.Tensor, out: torch.Tensor) -> None:
        if "fn" not in cache:
            cache["fn"] = getattr(load_kernel(name, tuple(sources), symbol), symbol)
        cache["fn"](x, out)

    return launch


def compare_scalar(lhs: float, rhs: float, atol: float = 1e-2, rtol: float = 1e-3) -> bool:
    diff = abs(lhs - rhs)
    if diff <= atol or diff <= rtol * max(abs(lhs), abs(rhs)):
        return True
    print(f"Mismatch: lhs={lhs:.6f} rhs={rhs:.6f}")
    return False


def measure_ms(launch: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        launch()
        torch.cuda.synchronize()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best


def run_impl(name: str, backend: str, launch: Callable[[], None], verify: Callable[[], bool], num_bytes: float) -> bool:
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
    gadds = float(N - 1) / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[reduction/{backend}:{name}] best={best_ms:.4f} ms, {gadds:.3f} GAdd/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    host_input = (torch.arange(N, dtype=torch.float32).remainder(113) - 56).mul(0.03125)
    ref = host_input.sum(dtype=torch.float64).to(dtype=torch.float32).item() if VERIFY else None
    device_input = host_input.cuda()
    device_output = torch.zeros(1, device="cuda", dtype=torch.float32)
    num_bytes = float(N) * device_input.element_size() + device_output.element_size()

    implementations = [(name, "cuda", make_launch(name, source, symbol)) for name, source, symbol in KERNELS]
    implementations.append(("torch", "pytorch", lambda x, out: out.copy_(torch.sum(x).reshape(1))))

    all_passed = True
    for name, backend, fn in implementations:
        def launch(fn=fn) -> None:
            device_output.zero_()
            fn(device_input, device_output)

        passed = run_impl(
            name,
            backend,
            launch,
            lambda: True if ref is None else compare_scalar(device_output.item(), ref),
            num_bytes,
        )
        all_passed = passed and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
