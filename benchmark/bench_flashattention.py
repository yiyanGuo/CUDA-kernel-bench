from __future__ import annotations

import math

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


DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_HEADS = 32
DEFAULT_QUERY_LEN = 4096
DEFAULT_KEY_LEN = 4096
HEAD_DIM = 64
MMA_SOURCE = "kernel/flashattention/flashattention_mma.cu"


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
            "flashattention expects four dimensions: "
            "flashattention <batch_size> <num_heads> <query_len> <key_len>; "
            "head_dim is fixed to 64."
        )
    batch_size, num_heads, query_len, key_len = dims
    if batch_size <= 0 or num_heads <= 0 or query_len <= 0 or key_len <= 0:
        raise ValueError("flashattention dimensions must all be > 0.")
    if query_len % 32 != 0 or key_len % 32 != 0:
        raise ValueError(
            "flashattention currently expects query_len and key_len to be "
            "multiples of 32."
        )
    return batch_size, num_heads, query_len, key_len


def _make_input(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    element_count = math.prod(shape)
    return (
        torch.arange(element_count, dtype=torch.float32)
        .reshape(shape)
        .add(offset)
        .remainder(257)
        .sub(128)
        .mul(0.0078125)
    )


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v.float()).to(dtype=q.dtype)


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    if config.casual:
        raise ValueError("flashattention benchmark does not support causal mode yet.")

    batch_size, num_heads, query_len, key_len = _resolve_shape(dims)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    implementations = load_backend_implementations(
        ["kernel.flashattention.flashattention_cuda"]
    )
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=lambda q, k, v, out: out.copy_(_reference(q, k, v, scale)),
            source="benchmark/bench_flashattention.py",
        )
    )
    implementations = filter_implementations(implementations, config)

    q_shape = (batch_size, num_heads, query_len, HEAD_DIM)
    kv_shape = (batch_size, num_heads, key_len, HEAD_DIM)
    base_q = _make_input(q_shape, 0).to(device="cuda")
    base_k = _make_input(kv_shape, 17).to(device="cuda")
    base_v = _make_input(kv_shape, 31).to(device="cuda")

    def uses_half(implementation: KernelImplementation) -> bool:
        return implementation.source == MMA_SOURCE

    def verify(
        device_output: torch.Tensor,
        ref: torch.Tensor | None,
        abs_tolerance: float,
        rel_tolerance: float,
    ) -> bool:
        if ref is None:
            return True
        return compare_tensors(
            device_output,
            ref,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )

    all_passed = True
    attention_scores = float(batch_size * num_heads * query_len * key_len)
    matmul_ops = attention_scores * float(HEAD_DIM) * 4.0
    softmax_ops = attention_scores * 5.0

    for implementation in implementations:
        dtype = torch.float16 if uses_half(implementation) else torch.float32
        device_q = base_q.to(dtype=dtype)
        device_k = base_k.to(dtype=dtype)
        device_v = base_v.to(dtype=dtype)
        device_output = torch.empty_like(device_q)
        ref = _reference(device_q, device_k, device_v, scale) if config.verify else None
        abs_tolerance = 2e-3 if dtype == torch.float16 else 1e-4
        rel_tolerance = 2e-3 if dtype == torch.float16 else 1e-4
        num_bytes = float(
            device_q.numel()
            + device_k.numel()
            + device_v.numel()
            + device_output.numel()
        ) * device_q.element_size()

        passed = run_implementation(
            op_name="flashattention",
            implementation=implementation,
            config=config,
            launch=lambda impl=implementation: impl.launch(
                device_q,
                device_k,
                device_v,
                device_output,
            ),
            verify=lambda out=device_output, ref=ref, atol=abs_tolerance, rtol=rel_tolerance: verify(
                out,
                ref,
                atol,
                rtol,
            ),
            work_units=matmul_ops + softmax_ops,
            work_unit_name="OP",
            num_bytes=num_bytes,
            prepare=lambda: device_output.zero_(),
        )
        all_passed = passed and all_passed
    return all_passed
