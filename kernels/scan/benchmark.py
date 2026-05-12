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
    ("naive", "scan_naive.cu", "scan_naive"),
    ("one_block", "scan_one_block.cu", "scan_one_block"),
    ("multi_block", "scan_multi_block.cu", "scan_multi_block"),
    ("warp", "scan_warp.cu", "scan_warp"),
    ("thrust", "thrust_exclusive_scan.cu", "scan_thrust_exclusive"),
    ("memory_buffer", "scan_memory_buffer.cu", "scan_memory_buffer"),
]


@lru_cache(maxsize=None)
def load_kernel(name: str, source: str, symbol: str):
    module_name = f"cuda_kernel_bench_scan_{name}"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / "scan_single_binding.cpp"), str(KERNEL_DIR / source)],
        extra_include_paths=[str(INCLUDE_DIR), str(KERNEL_DIR)],
        extra_cflags=["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}"],
        extra_cuda_cflags=["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}"],
        build_directory=str(build_dir),
        verbose=False,
    )


def make_launch(name: str, source: str, symbol: str) -> Callable[[torch.Tensor, torch.Tensor], None]:
    cache = {}

    def launch(x: torch.Tensor, out: torch.Tensor) -> None:
        if "fn" not in cache:
            cache["fn"] = getattr(load_kernel(name, source, symbol), symbol)
        cache["fn"](x, out)

    return launch


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if torch.allclose(lhs, rhs, atol=1e-3, rtol=1e-3):
        return True
    mismatch = (~torch.isclose(lhs, rhs, atol=1e-3, rtol=1e-3)).flatten().nonzero()[0].item()
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
    gadds = float(max(N - 1, 0)) / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[scan/{backend}:{name}] best={best_ms:.4f} ms, {gadds:.3f} GAdd/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    host_input = (torch.arange(N, dtype=torch.float32).remainder(37) - 18).mul(0.125)
    ref = None
    if VERIFY:
        ref = torch.zeros_like(host_input)
        ref[1:] = torch.cumsum(host_input[:-1], dim=0)
    device_input = host_input.cuda()
    device_output = torch.empty_like(device_input)
    num_bytes = float(N) * 2.0 * device_input.element_size()

    implementations = [(name, "cuda", make_launch(name, source, symbol)) for name, source, symbol in KERNELS]
    implementations.append(("torch", "pytorch", lambda x, out: out.copy_(torch.cat((torch.zeros_like(x[:1]), torch.cumsum(x[:-1], dim=0))))))

    all_passed = True
    for name, backend, fn in implementations:
        passed = run_impl(
            name,
            backend,
            lambda fn=fn: fn(device_input, device_output),
            lambda: True if ref is None else compare_tensors(device_output.cpu(), ref),
            num_bytes,
        )
        all_passed = passed and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
