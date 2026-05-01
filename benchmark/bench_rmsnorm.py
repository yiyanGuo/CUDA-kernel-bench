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


DEFAULT_TOKENS = 4096
DEFAULT_HIDDEN = 8192
EPS = 1e-6


def _resolve_dims(dims: list[int]) -> tuple[int, int]:
    if not dims:
        return DEFAULT_TOKENS, DEFAULT_HIDDEN
    if len(dims) != 2:
        raise ValueError("rmsnorm expects two dimensions: rmsnorm <tokens> <hidden>")
    if dims[0] <= 0 or dims[1] <= 0:
        raise ValueError("rmsnorm dimensions must be > 0.")
    return dims[0], dims[1]


def _make_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    host_in = (
        torch.arange(num_tokens * hidden_size, dtype=torch.float32)
        .reshape(num_tokens, hidden_size)
        .remainder(97)
        .mul(0.01)
    )
    host_w = torch.arange(hidden_size, dtype=torch.float32).remainder(13).mul(0.1)
    return host_in.to(dtype=dtype), host_w.to(dtype=dtype)


def _reference(
    dev_in: torch.Tensor,
    dev_w: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    input_f32 = dev_in.float()
    weight_f32 = dev_w.float()
    sq = input_f32 * input_f32
    mean_sq = sq.mean(dim=1, keepdim=True)
    denom = torch.sqrt(mean_sq + EPS)
    return ((input_f32 / denom) * weight_f32).to(dtype=output_dtype)


def _run_path(
    *,
    path_name: str,
    module_names: list[str],
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    config: BenchmarkConfig,
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[bool, int]:
    device = torch.device("cuda")
    host_in, host_w = _make_inputs(num_tokens, hidden_size, dtype)
    dev_in = host_in.to(device=device)
    dev_w = host_w.to(device=device)
    dev_out = torch.empty_like(dev_in)

    implementations = load_backend_implementations(module_names)
    implementations.append(
        KernelImplementation(
            name=f"torch_{path_name}",
            backend="pytorch",
            launch=lambda x, w, out, eps: out.copy_(_reference(x, w, x.dtype)),
            source="benchmark/bench_rmsnorm.py",
        )
    )
    implementations = filter_implementations(
        implementations,
        config,
        allow_empty=True,
    )
    if not implementations:
        return True, 0

    ref = _reference(dev_in, dev_w, dev_in.dtype)

    all_passed = True
    work_units = float(num_tokens * hidden_size)
    num_bytes = work_units * dev_in.element_size() * 3.0

    for implementation in implementations:
        passed = run_implementation(
            op_name=f"rmsnorm/{path_name}",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(dev_in, dev_w, dev_out, EPS),
            verify=lambda: compare_tensors(
                dev_out,
                ref,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            ),
            work_units=work_units,
            work_unit_name="OP",
            num_bytes=num_bytes,
        )
        all_passed = passed and all_passed
    return all_passed, len(implementations)


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    num_tokens, hidden_size = _resolve_dims(dims)

    float_passed, float_count = _run_path(
        path_name="float",
        module_names=["kernel.RMSNorm.rms_cuda"],
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        dtype=torch.float32,
        config=config,
        abs_tolerance=1e-5,
        rel_tolerance=1e-5,
    )

    half_passed, half_count = _run_path(
        path_name="half",
        module_names=["kernel.RMSNorm.rms_half_cuda"],
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        dtype=torch.float16,
        config=config,
        abs_tolerance=1e-2,
        rel_tolerance=1e-2,
    )

    if config.implementation is not None and float_count + half_count == 0:
        raise ValueError(f"Unknown implementation '{config.implementation}'.")

    return float_passed and half_passed
