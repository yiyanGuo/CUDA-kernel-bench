from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_rope",
        [
            "kernel/RoPE/rope_binding.cpp",
            "kernel/RoPE/rope_naive.cu",
        ],
    )
    return [
        KernelImplementation(
            name="naive",
            backend="cuda",
            launch=module.rope_naive,
            source="kernel/RoPE/rope_naive.cu",
        ),
    ]
