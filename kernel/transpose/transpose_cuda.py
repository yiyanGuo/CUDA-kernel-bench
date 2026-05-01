from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_transpose",
        [
            "kernel/transpose/transpose_binding.cpp",
            "kernel/transpose/transpose_naive.cu",
            "kernel/transpose/transpose_tile_float4.cu",
        ],
    )
    return [
        KernelImplementation(
            "naive",
            "cuda",
            module.transpose_naive,
            "kernel/transpose/transpose_naive.cu",
        ),
        KernelImplementation(
            "tile_float4",
            "cuda",
            module.transpose_tile_float4,
            "kernel/transpose/transpose_tile_float4.cu",
        ),
    ]
