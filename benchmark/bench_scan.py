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
        raise ValueError("scan expects exactly one dimension: scan <N>")
    if dims[0] <= 0:
        raise ValueError("scan dimension must be > 0.")
    return dims[0]


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    n = _resolve_n(dims)

    host_input = (torch.arange(n, dtype=torch.float32).remainder(37) - 18).mul(0.125)
    host_ref = torch.zeros_like(host_input)
    if n > 1:
        host_ref[1:] = torch.cumsum(host_input[:-1], dim=0)

    device_input = host_input.to(device="cuda")
    device_output = torch.empty_like(device_input)

    implementations = load_backend_implementations(["kernel.scan.scan_cuda"])
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda x, out: out.copy_(
                torch.cat((torch.zeros_like(x[:1]), torch.cumsum(x[:-1], dim=0)))
            ),
            source="benchmark/bench_scan.py",
        )
    )
    implementations = filter_implementations(implementations, config)

    all_passed = True
    for implementation in implementations:
        passed = run_implementation(
            op_name="scan",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(device_input, device_output),
            verify=lambda: compare_tensors(
                device_output,
                host_ref,
                abs_tolerance=1e-3,
                rel_tolerance=1e-3,
            ),
            work_units=float(max(n - 1, 0)),
            work_unit_name="Add",
            num_bytes=float(n) * 2.0 * device_input.element_size(),
        )
        all_passed = passed and all_passed
    return all_passed
