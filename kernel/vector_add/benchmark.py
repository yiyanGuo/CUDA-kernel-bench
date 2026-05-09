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
    load_backend_implementations,
    run_implementation,
)
from kernel.api import KernelImplementation  # noqa: E402


DEFAULT_N = 1 << 24


def _resolve_n(dims: list[int]) -> int:
    if not dims:
        return DEFAULT_N
    if len(dims) != 1:
        raise ValueError("vector_add expects exactly one dimension: vector_add <N>")
    if dims[0] <= 0:
        raise ValueError("vector_add dimension must be > 0.")
    return dims[0]


def _make_inputs(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    host_a = torch.arange(n, dtype=torch.float32).remainder(97).mul(0.5)
    host_b = torch.arange(n, dtype=torch.float32).remainder(53).mul(0.25)
    device = torch.device("cuda")
    return host_a.to(device=device, dtype=dtype), host_b.to(device=device, dtype=dtype)


def _implementation_supports_dtype(
    implementation: KernelImplementation,
    dtype: torch.dtype,
) -> bool:
    if implementation.name == "f32x4":
        return dtype == torch.float32
    if implementation.name == "f16x2":
        return dtype == torch.float16
    return dtype in {torch.float32, torch.float16}


def run_benchmark(dims: list[int], config: BenchmarkConfig, dtype: torch.dtype = torch.float32) -> bool:
    ensure_cuda_available()
    n = _resolve_n(dims)

    device_a, device_b = _make_inputs(n, dtype)
    device_output = torch.empty_like(device_a)
    device_ref = device_a + device_b if config.verify else None

    implementations = load_backend_implementations(
        [
            "kernel.vector_add.vector_add_cuda",
            "kernel.vector_add.vector_add_cutlass",
            "kernel.vector_add.vector_add_triton",
        ]
    )
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda a, b, out: torch.add(a, b, out=out),
            source="kernel/vector_add/benchmark.py",
        )
    )
    implementations = [
        implementation
        for implementation in implementations
        if _implementation_supports_dtype(implementation, dtype)
    ]
    implementations = filter_implementations(implementations, config)

    def verify() -> bool:
        if device_ref is None:
            return True
        tolerance = 1e-2 if dtype == torch.float16 else 1e-5
        return compare_tensors(device_output, device_ref, tolerance, tolerance)

    all_passed = True
    for implementation in implementations:
        passed = run_implementation(
            op_name=f"vector_add/{str(dtype).removeprefix('torch.')}",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(
                device_a, device_b, device_output
            ),
            verify=verify,
            work_units=float(n),
            work_unit_name="FLOP",
            num_bytes=float(n) * 3.0 * device_a.element_size(),
        )
        all_passed = passed and all_passed
    return all_passed


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"float", "float32", "f32", "fp32"}:
        return torch.float32
    if normalized in {"half", "float16", "f16", "fp16"}:
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be one of: float32, float16")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run only the vector_add benchmark.",
    )
    parser.add_argument("n", nargs="?", type=int, default=DEFAULT_N)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.float32)
    parser.add_argument("--impl", help="Implementation name, e.g. dispatch, f32x4, f16x2, triton:elementwise.")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("n must be > 0.")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0.")

    config = BenchmarkConfig(
        warmup=args.warmup,
        repeat=args.repeat,
        mode="single" if args.impl else "compare",
        implementation=args.impl,
        verify=not args.no_verify,
    )
    return 0 if run_benchmark([args.n], config, args.dtype) else 2


if __name__ == "__main__":
    raise SystemExit(main())
