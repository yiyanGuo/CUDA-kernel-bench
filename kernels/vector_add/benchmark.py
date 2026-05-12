from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
INCLUDE_DIR = REPO_ROOT / "include"
CUTLASS_INCLUDE_DIR = Path(os.environ.get("CUTLASS_INCLUDE_DIR", "/root/code/cutlass/include"))

N = 1 << 24
WARMUP = 2
REPEAT = 5
VERIFY = True
DTYPES = (torch.float32, torch.float16)


def include_paths() -> list[str]:
    paths = [str(INCLUDE_DIR), str(KERNEL_DIR)]
    if CUTLASS_INCLUDE_DIR.exists():
        paths.append(str(CUTLASS_INCLUDE_DIR))
    cutlass_util = CUTLASS_INCLUDE_DIR.parent / "tools" / "util" / "include"
    if cutlass_util.exists():
        paths.append(str(cutlass_util))
    return paths


@lru_cache(maxsize=None)
def load_cuda_extension(module_name: str, sources: tuple[str, ...]):
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / source) for source in sources],
        extra_include_paths=include_paths(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )


def cuda_launch(module_name: str, sources: list[str], function_name: str) -> Callable[..., None]:
    cache: dict[str, Callable[..., None]] = {}

    def launch(*args) -> None:
        if "fn" not in cache:
            module = load_cuda_extension(module_name, tuple(sources))
            cache["fn"] = getattr(module, function_name)
        cache["fn"](*args)

    return launch


if triton is not None and tl is not None:
    @triton.jit
    def vector_add_triton_kernel(a_ptr, b_ptr, out_ptr, numel, block_size: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, a + b, mask=mask)


def triton_launch(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("triton is not installed.")
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["block_size"]),)
    vector_add_triton_kernel[grid](a, b, out, numel, block_size=1024)


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


def run_impl(name: str, backend: str, launch: Callable[[], None], verify: Callable[[], bool], bytes_moved: float) -> bool:
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
    gflops = float(N) / (best_ms * 1e6)
    bandwidth = bytes_moved / (best_ms * 1e6)
    print(f"[vector_add/{backend}:{name}] best={best_ms:.4f} ms, {gflops:.3f} GFLOP/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def make_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    host_a = torch.arange(N, dtype=torch.float32).remainder(97).mul(0.5)
    host_b = torch.arange(N, dtype=torch.float32).remainder(53).mul(0.25)
    return host_a.cuda().to(dtype), host_b.cuda().to(dtype)


def benchmark_dtype(dtype: torch.dtype) -> bool:
    a, b = make_inputs(dtype)
    out = torch.empty_like(a)
    ref = a + b if VERIFY else None
    tol = 1e-2 if dtype == torch.float16 else 1e-5

    implementations: list[tuple[str, str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]]] = [
        ("dispatch", "cuda", cuda_launch("cuda_kernel_bench_vector_add", ["vector_add.cu"], "vector_add")),
        ("cute_x4", "cutlass", cuda_launch("cuda_kernel_bench_vector_add_cutlass", ["vector_add_cutlass.cu"], "vector_add_cute_x4")),
        ("torch", "pytorch", lambda x, y, z: torch.add(x, y, out=z)),
    ]
    if dtype == torch.float32:
        implementations.insert(1, ("f32x4", "cuda", cuda_launch("cuda_kernel_bench_vector_add", ["vector_add.cu"], "vector_add_f32x4")))
    if dtype == torch.float16:
        implementations.insert(1, ("f16x2", "cuda", cuda_launch("cuda_kernel_bench_vector_add", ["vector_add.cu"], "vector_add_f16x2")))
    if triton is not None:
        implementations.insert(-1, ("elementwise", "triton", triton_launch))

    bytes_moved = float(N) * 3.0 * a.element_size()
    all_passed = True
    print(f"== vector_add {dtype} N={N} ==")
    for name, backend, fn in implementations:
        passed = run_impl(
            name,
            backend,
            lambda fn=fn: fn(a, b, out),
            lambda: True if ref is None else compare_tensors(out, ref, tol, tol),
            bytes_moved,
        )
        all_passed = passed and all_passed
    return all_passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    all_passed = True
    for dtype in DTYPES:
        all_passed = benchmark_dtype(dtype) and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
