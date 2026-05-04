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


DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_QUERY_LEN = 128
DEFAULT_KEY_LEN = (1 << 17)


def _resolve_shape(dims: list[int]) -> tuple[int, int, int, int]:
    if not dims:
        return (
            DEFAULT_BATCH_SIZE,
            DEFAULT_NUM_HEADS,
            DEFAULT_QUERY_LEN,
            DEFAULT_KEY_LEN,
        )
    if len(dims) != 4:
        raise ValueError(
            "softmax expects four dimensions: "
            "softmax <batch_size> <num_heads> <query_len> <key_len>"
        )
    batch_size, num_heads, query_len, key_len = dims
    if batch_size <= 0 or num_heads <= 0 or query_len <= 0 or key_len <= 0:
        raise ValueError("softmax dimensions must all be > 0.")
    return batch_size, num_heads, query_len, key_len


def _make_input(
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
) -> torch.Tensor:
    element_count = batch_size * num_heads * query_len * key_len
    return (
        torch.arange(element_count, dtype=torch.float32)
        .reshape(batch_size, num_heads, query_len, key_len)
        .remainder(257)
        .sub(128)
        .mul(0.03125)
    )


def _reference(input_tensor: torch.Tensor, casual: bool = False) -> torch.Tensor:
    logits = input_tensor.float()
    if casual:
        query_len = logits.size(-2)
        key_len = logits.size(-1)
        query_positions = torch.arange(
            query_len,
            device=logits.device,
        ).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=logits.device)
        mask = key_positions <= query_positions
        logits = logits.masked_fill(~mask, -torch.inf)
    return torch.softmax(logits, dim=-1).to(dtype=input_tensor.dtype)


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    batch_size, num_heads, query_len, key_len = _resolve_shape(dims)
    element_count = batch_size * num_heads * query_len * key_len

    host_input = _make_input(batch_size, num_heads, query_len, key_len)
    device_input = host_input.to(device="cuda")
    device_output = torch.empty_like(device_input)
    ref = _reference(device_input, config.casual)

    implementations = load_backend_implementations(["kernel.softmax.softmax_cuda"])
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda x, out, casual=False: out.copy_(_reference(x, casual)),
            source="benchmark/bench_softmax.py",
        )
    )
    implementations = filter_implementations(implementations, config)

    all_passed = True
    for implementation in implementations:
        passed = run_implementation(
            op_name="softmax",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(
                device_input,
                device_output,
                config.casual,
            ),
            verify=lambda: compare_tensors(
                device_output,
                ref,
                abs_tolerance=1e-5,
                rel_tolerance=1e-5,
            ),
            work_units=float(element_count) * 5.0,
            work_unit_name="OP",
            num_bytes=float(element_count) * 2.0 * device_input.element_size(),
        )
        all_passed = passed and all_passed
    return all_passed
