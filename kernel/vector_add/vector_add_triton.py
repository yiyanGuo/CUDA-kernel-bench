from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernel.api import KernelImplementation


@triton.jit
def _vector_add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    numel,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def _launch(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.shape != b.shape or a.shape != out.shape:
        raise ValueError("vector_add triton backend expects matching tensor shapes.")
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["block_size"]),)
    _vector_add_kernel[grid](a, b, out, numel, block_size=1024)


def get_implementations() -> list[KernelImplementation]:
    return [
        KernelImplementation(
            name="elementwise",
            backend="triton",
            launch=_launch,
            source="kernel/vector_add/vector_add_triton.py",
        )
    ]
