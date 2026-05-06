from __future__ import annotations

import torch

from benchmark.common import (
    BenchmarkConfig,
    compare_tensors,
    ensure_cuda_available,
    filter_implementations,
    load_backend_implementations,
    run_implementation,
)
from kernel.api import KernelImplementation


DEFAULT_N = 1 << 24


def _resolve_n(dims: list[int]) -> int:
    if not dims:
        return DEFAULT_N
    if len(dims) != 1:
        raise ValueError("vector_add expects exactly one dimension: vector_add <N>")
    if dims[0] <= 0:
        raise ValueError("vector_add dimension must be > 0.")
    return dims[0]


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    n = _resolve_n(dims)

    host_a = torch.arange(n, dtype=torch.float32).remainder(97).mul(0.5)
    host_b = torch.arange(n, dtype=torch.float32).remainder(53).mul(0.25)
    host_ref = host_a + host_b if config.verify else None

    device = torch.device("cuda")
    device_a = host_a.to(device=device)
    device_b = host_b.to(device=device)
    device_output = torch.empty_like(device_a)

    implementations = load_backend_implementations(
        [
            "kernel.vector_add.vector_add_cuda",
            "kernel.vector_add.vector_add_triton",
        ]
    )
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda a, b, out: torch.add(a, b, out=out),
            source="benchmark/bench_vector_add.py",
        )
    )
    implementations = filter_implementations(implementations, config)

    def verify() -> bool:
        if host_ref is None:
            return True
        return compare_tensors(device_output, host_ref)

    all_passed = True
    for implementation in implementations:
        passed = run_implementation(
            op_name="vector_add",
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
