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
DEFAULT_SEQ_LEN = 1024
DEFAULT_NUM_Q_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_POSITION_OFFSET = 0
ROPE_BASE = 10000.0


def _resolve_shape(dims: list[int]) -> tuple[int, int, int, int, int, int, int]:
    if not dims:
        return (
            DEFAULT_BATCH_SIZE,
            DEFAULT_SEQ_LEN,
            DEFAULT_NUM_Q_HEADS,
            DEFAULT_NUM_KV_HEADS,
            DEFAULT_HEAD_DIM,
            DEFAULT_HEAD_DIM,
            DEFAULT_POSITION_OFFSET,
        )
    if len(dims) not in (5, 6, 7):
        raise ValueError(
            "rope expects: rope <batch> <seq_len> <num_q_heads> "
            "<num_kv_heads> <head_dim> [rotary_dim] [position_offset]"
        )

    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = dims[:5]
    rotary_dim = dims[5] if len(dims) >= 6 else head_dim
    position_offset = dims[6] if len(dims) == 7 else DEFAULT_POSITION_OFFSET

    if min(batch_size, seq_len, num_q_heads, num_kv_heads, head_dim) <= 0:
        raise ValueError("rope dimensions must all be > 0.")
    if head_dim % 2 != 0:
        raise ValueError("rope head_dim must be even.")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError("rope rotary_dim must be even and in (0, head_dim].")
    if position_offset < 0:
        raise ValueError("rope position_offset must be >= 0.")
    return (
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
    )


def _make_tensor(shape: tuple[int, ...], modulo: int, scale: float) -> torch.Tensor:
    element_count = 1
    for dim in shape:
        element_count *= dim
    return (
        torch.arange(element_count, dtype=torch.float32)
        .reshape(shape)
        .remainder(modulo)
        .sub(modulo // 2)
        .mul(scale)
    )


def _make_inputs(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    host_q = _make_tensor(
        (batch_size, seq_len, num_q_heads, head_dim),
        modulo=251,
        scale=0.01,
    )
    host_k = _make_tensor(
        (batch_size, seq_len, num_kv_heads, head_dim),
        modulo=197,
        scale=0.0125,
    )
    position_ids = (
        torch.arange(seq_len, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
        .contiguous()
    )

    pair_dim = rotary_dim // 2
    max_position = int(position_ids.max().item()) + position_offset + 1
    positions = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
    pair_ids = torch.arange(pair_dim, dtype=torch.float32).unsqueeze(0)
    inv_freq = torch.pow(ROPE_BASE, -(2.0 * pair_ids) / float(rotary_dim))
    freqs = positions * inv_freq
    return host_q, host_k, torch.cos(freqs), torch.sin(freqs), position_ids


def _rotate_reference(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    position_offset: int,
) -> torch.Tensor:
    input_f32 = input_tensor.float()
    output = input_f32.clone()

    position = position_ids.long() + position_offset
    cos_broadcast = cos[position].float().unsqueeze(2)
    sin_broadcast = sin[position].float().unsqueeze(2)

    even = input_f32[..., :rotary_dim:2]
    odd = input_f32[..., 1:rotary_dim:2]
    output[..., :rotary_dim:2] = even * cos_broadcast - odd * sin_broadcast
    output[..., 1:rotary_dim:2] = even * sin_broadcast + odd * cos_broadcast
    return output.to(dtype=input_tensor.dtype)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _rotate_reference(q, cos, sin, position_ids, rotary_dim, position_offset),
        _rotate_reference(k, cos, sin, position_ids, rotary_dim, position_offset),
    )


def _apply_reference_in_place(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_dim: int,
    position_offset: int,
) -> None:
    ref_q, ref_k = _reference(q, k, cos, sin, position_ids, rotary_dim, position_offset)
    q.copy_(ref_q)
    k.copy_(ref_k)


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    ensure_cuda_available()
    (
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        position_offset,
    ) = _resolve_shape(dims)

    host_q, host_k, host_cos, host_sin, host_position_ids = _make_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        position_offset=position_offset,
    )
    base_q = host_q.to(device="cuda")
    base_k = host_k.to(device="cuda")
    device_q = torch.empty_like(base_q)
    device_k = torch.empty_like(base_k)
    device_cos = host_cos.to(device="cuda")
    device_sin = host_sin.to(device="cuda")
    device_position_ids = host_position_ids.to(device="cuda")

    implementations = load_backend_implementations(["kernel.RoPE.rope_cuda"])
    implementations.append(
        KernelImplementation(
            name="torch",
            backend="pytorch",
            launch=_apply_reference_in_place,
            source="benchmark/bench_rope.py",
        )
    )
    implementations = filter_implementations(implementations, config)
    ref_q, ref_k = (
        _reference(
            base_q,
            base_k,
            device_cos,
            device_sin,
            device_position_ids,
            rotary_dim,
            position_offset,
        )
        if config.verify
        else (None, None)
    )

    def prepare() -> None:
        device_q.copy_(base_q)
        device_k.copy_(base_k)

    def verify() -> bool:
        if ref_q is None or ref_k is None:
            return True
        q_passed = compare_tensors(
            device_q,
            ref_q,
            abs_tolerance=1e-5,
            rel_tolerance=1e-5,
        )
        k_passed = compare_tensors(
            device_k,
            ref_k,
            abs_tolerance=1e-5,
            rel_tolerance=1e-5,
        )
        return q_passed and k_passed

    all_passed = True
    row_count = float(batch_size * seq_len * (num_q_heads + num_kv_heads))
    pair_count = float(rotary_dim // 2)
    work_units = row_count * pair_count * 6.0
    num_bytes = row_count * pair_count * 6.0 * base_q.element_size()
    for implementation in implementations:
        passed = run_implementation(
            op_name="rope",
            implementation=implementation,
            config=config,
            prepare=prepare,
            launch=lambda impl=implementation: impl.launch(
                device_q,
                device_k,
                device_cos,
                device_sin,
                device_position_ids,
                rotary_dim,
                position_offset,
            ),
            verify=verify,
            work_units=work_units,
            work_unit_name="FLOP",
            num_bytes=num_bytes,
        )
        all_passed = passed and all_passed
    return all_passed
