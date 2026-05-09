from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.common import (  # noqa: E402
    BenchmarkConfig,
    compare_tensors,
    ensure_cuda_available,
    filter_implementations,
    run_implementation,
)
from kernel.api import KernelImplementation  # noqa: E402
from kernel.cuda_extension import load_extension_from_iterable  # noqa: E402


DEFAULT_M = 4096
DEFAULT_N = 4096
DEFAULT_K = 4096
TILE_M = 128
TILE_N = 128
TILE_K = 32


def get_module():
    return load_extension_from_iterable(
        "cuda_kernel_bench_hgemm_simple",
        ["kernel/hgemm/hgemm_simple.cu"],
    )


def _resolve_dims(dims: list[int]) -> tuple[int, int, int]:
    if not dims:
        return DEFAULT_M, DEFAULT_N, DEFAULT_K
    if len(dims) != 3:
        raise ValueError("hgemm expects dimensions: hgemm <M> <N> <K>")
    m, n, k = dims
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("M, N, and K must be > 0.")
    if m % TILE_M != 0 or n % TILE_N != 0 or k % TILE_K != 0:
        raise ValueError("hgemm_simple requires M%128 == 0, N%128 == 0, and K%32 == 0.")
    return m, n, k


def _make_inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((n, k), device=device, dtype=torch.float16)
    return a, b


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    m, n, k = _resolve_dims(dims)

    module = get_module()
    a, b = _make_inputs(m, n, k)
    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    kernel_reference = torch.matmul(a.float(), b.float().t()).half() if config.verify else None
    torch_reference = torch.matmul(a, b.t()) if config.verify else None

    implementations = [
        KernelImplementation(
            name="simple",
            backend="cuda",
            launch=module.hgemm_simple,
            source="kernel/hgemm/hgemm_simple.cu",
        ),
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda lhs, rhs, out: torch.matmul(lhs, rhs.t(), out=out),
            source="kernel/hgemm/benchmark.py",
        ),
    ]
    implementations = filter_implementations(implementations, config)

    def verify(implementation: KernelImplementation) -> bool:
        reference = torch_reference if implementation.backend == "pytorch" else kernel_reference
        if reference is None:
            return True
        return compare_tensors(output, reference, abs_tolerance=1e-2, rel_tolerance=1e-2)

    all_passed = True
    flops = float(2 * m * n * k)
    num_bytes = float((m * k + n * k + m * n) * a.element_size())
    for implementation in implementations:
        passed = run_implementation(
            op_name=f"hgemm/{m}x{n}x{k}",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(a, b, output),
            verify=lambda impl=implementation: verify(impl),
            work_units=flops,
            work_unit_name="FLOP",
            num_bytes=num_bytes,
        )
        all_passed = passed and all_passed
    return all_passed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the hgemm_simple benchmark.",
    )
    parser.add_argument("m", nargs="?", type=int, default=DEFAULT_M)
    parser.add_argument("n", nargs="?", type=int, default=DEFAULT_N)
    parser.add_argument("k", nargs="?", type=int, default=DEFAULT_K)
    parser.add_argument("--impl", help="Implementation name, e.g. simple, cuda:simple, torch.")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0.")

    config = BenchmarkConfig(
        warmup=args.warmup,
        repeat=args.repeat,
        mode="single" if args.impl else "compare",
        implementation=args.impl,
        verify=not args.no_verify,
    )
    try:
        passed = run_benchmark([args.m, args.n, args.k], config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
