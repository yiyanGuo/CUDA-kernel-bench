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


DEFAULT_ROWS = 2048
DEFAULT_COLS = 2048


def _resolve_shape(dims: list[int]) -> tuple[int, int]:
    if not dims:
        return DEFAULT_ROWS, DEFAULT_COLS
    if len(dims) != 2:
        raise ValueError(
            "transpose expects two dimensions: transpose <rows> <cols>"
        )
    rows, cols = dims
    if rows <= 0 or cols <= 0:
        raise ValueError("transpose dimensions must both be > 0.")
    return rows, cols


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    rows, cols = _resolve_shape(dims)
    element_count = rows * cols

    host_input = (
        torch.arange(element_count, dtype=torch.float32).remainder(113).mul(0.125)
    ).view(rows, cols)
    host_ref = host_input.transpose(0, 1).contiguous()

    device_input = host_input.to(device="cuda")
    device_output = torch.empty((cols, rows), device="cuda", dtype=torch.float32)

    implementations = load_backend_implementations(["kernel.transpose.transpose_cuda"])
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda x, out: out.copy_(x.transpose(0, 1).contiguous()),
            source="benchmark/bench_transpose.py",
        )
    )
    implementations = filter_implementations(implementations, config)

    all_passed = True
    for implementation in implementations:
        passed = run_implementation(
            op_name="transpose",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(device_input, device_output),
            verify=lambda: compare_tensors(device_output, host_ref),
            work_units=float(element_count),
            work_unit_name="Elem",
            num_bytes=float(element_count) * 2.0 * device_input.element_size(),
        )
        all_passed = passed and all_passed
    return all_passed
