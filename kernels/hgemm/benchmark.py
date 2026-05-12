from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
INCLUDE_DIR = REPO_ROOT / "include"
CUTLASS_INCLUDE_DIR = Path(os.environ.get("CUTLASS_INCLUDE_DIR", "/root/code/cutlass/include"))

M = 4096
N = 4096
K = 4096
WARMUP = 2
REPEAT = 5
VERIFY = True

KERNELS = [
    ("simple", "hgemm_simple.cu", "hgemm_simple"),
    ("shared_memory", "hgemm_shared_memory.cu", "hgemm_shared_memory"),
    ("2_pipeline", "hgemm_2_pipeline.cu", "hgemm_2_pipeline"),
]


def include_paths() -> list[str]:
    paths = [str(INCLUDE_DIR), str(KERNEL_DIR)]
    if CUTLASS_INCLUDE_DIR.exists():
        paths.append(str(CUTLASS_INCLUDE_DIR))
    cutlass_util = CUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include"
    if cutlass_util.exists():
        paths.append(str(cutlass_util))
    return paths


@lru_cache(maxsize=None)
def load_kernel(name: str, source: str):
    module_name = f"cuda_kernel_bench_hgemm_{name}"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / source)],
        extra_include_paths=include_paths(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )


def make_launch(name: str, source: str, symbol: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    cache = {}

    def launch(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        if "fn" not in cache:
            cache["fn"] = getattr(load_kernel(name, source), symbol)
        cache["fn"](a, b, out)

    return launch


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if torch.allclose(lhs, rhs, atol=1e-2, rtol=1e-2):
        return True
    mismatch = (~torch.isclose(lhs, rhs, atol=1e-2, rtol=1e-2)).flatten().nonzero()[0].item()
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
    flops = float(2 * M * N * K)
    gflops = flops / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[hgemm/{backend}:{name}] best={best_ms:.4f} ms, {gflops:.3f} GFLOP/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if M % 128 != 0 or N % 128 != 0 or K % 32 != 0:
        raise ValueError("hgemm requires M%128 == 0, N%128 == 0, and K%32 == 0.")

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    kernel_ref = torch.matmul(a.float(), b.float().t()).half() if VERIFY else None
    torch_ref = torch.matmul(a, b.t()) if VERIFY else None
    num_bytes = float((M * K + N * K + M * N) * a.element_size())

    implementations = [(name, "cuda", make_launch(name, source, symbol)) for name, source, symbol in KERNELS]
    implementations.append(("torch", "pytorch", lambda lhs, rhs, output: torch.matmul(lhs, rhs.t(), out=output)))

    all_passed = True
    for name, backend, fn in implementations:
        ref = torch_ref if backend == "pytorch" else kernel_ref
        passed = run_impl(
            name,
            backend,
            lambda fn=fn: fn(a, b, out),
            lambda ref=ref: True if ref is None else compare_tensors(out, ref),
            num_bytes,
        )
        all_passed = passed and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
