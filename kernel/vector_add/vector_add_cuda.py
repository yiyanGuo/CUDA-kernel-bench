from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_vector_add",
        [
            "kernel/vector_add/vector_add_binding.cpp",
            "kernel/vector_add/vector_add_naive.cu",
            "kernel/vector_add/vector_add_float4.cu",
        ],
    )
    return [
        KernelImplementation(
            name="naive",
            backend="cuda",
            launch=module.vector_add_naive,
            source="kernel/vector_add/vector_add_naive.cu",
        ),
        KernelImplementation(
            name="float4",
            backend="cuda",
            launch=module.vector_add_float4,
            source="kernel/vector_add/vector_add_float4.cu",
        ),
    ]
