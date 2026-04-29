from __future__ import annotations

import torch

from benchmark.common import (
    BenchmarkConfig,
    compare_scalars,
    ensure_cuda_available,
    load_backend_implementations,
    run_implementation,
)


DEFAULT_N = 1 << 24


def _resolve_n(dims: list[int]) -> int:
    if not dims:
        return DEFAULT_N
    if len(dims) != 1:
        raise ValueError("reduction expects exactly one dimension: reduction <N>")
    if dims[0] <= 0:
        raise ValueError("reduction dimension must be > 0.")
    return dims[0]


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    n = _resolve_n(dims)

    host_input = (torch.arange(n, dtype=torch.float32).remainder(113) - 56).mul(0.03125)
    host_ref = host_input.sum(dtype=torch.float64).to(dtype=torch.float32)

    device_input = host_input.to(device="cuda")
    device_output = torch.zeros(1, device="cuda", dtype=torch.float32)

    implementations = load_backend_implementations(["kernel.reduction.reduction_cuda"])

    all_passed = True
    for implementation in implementations:
        def launch(impl=implementation) -> None:
            device_output.zero_()
            impl.launch(device_input, device_output)

        passed = run_implementation(
            op_name="reduction",
            implementation=implementation,
            config=config,
            launch=launch,
            verify=lambda: compare_scalars(
                device_output.item(),
                host_ref.item(),
                abs_tolerance=1e-2,
                rel_tolerance=1e-3,
            ),
            work_units=float(n - 1),
            work_unit_name="Add",
            num_bytes=float(n) * device_input.element_size() + device_output.element_size(),
        )
        all_passed = passed and all_passed
    return all_passed
