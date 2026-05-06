from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_flashattention",
        [
            "kernel/flashattention/flashattention_binding.cpp",
            "kernel/flashattention/flashattention_naive.cu",
            "kernel/flashattention/flashattention_mma.cu",
        ],
    )
    return [
        KernelImplementation(
            name="naive",
            backend="cuda",
            launch=module.flash_attention,
            source="kernel/flashattention/flashattention_naive.cu",
        ),
        KernelImplementation(
            name="mma",
            backend="cuda",
            launch=module.flash_attention_mma,
            source="kernel/flashattention/flashattention_mma.cu",
        ),
    ]
