from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_reduction",
        [
            "kernel/reduction/reduction_binding.cpp",
            "kernel/reduction/reduction_naive.cu",
            "kernel/reduction/reduction_presum.cu",
            "kernel/reduction/reduction_presum_float4.cu",
            "kernel/reduction/reduction_shuffle.cu",
            "kernel/reduction/reduction_grid_stride.cu",
            "kernel/reduction/reduction_integrate.cu",
        ],
    )
    return [
        KernelImplementation(
            "naive",
            "cuda",
            module.reduction_naive,
            "kernel/reduction/reduction_naive.cu",
        ),
        KernelImplementation(
            "presum",
            "cuda",
            module.reduction_presum,
            "kernel/reduction/reduction_presum.cu",
        ),
        KernelImplementation(
            "presum_float4",
            "cuda",
            module.reduction_presum_float4,
            "kernel/reduction/reduction_presum_float4.cu",
        ),
        KernelImplementation(
            "shuffle",
            "cuda",
            module.reduction_shuffle,
            "kernel/reduction/reduction_shuffle.cu",
        ),
        KernelImplementation(
            "grid_stride",
            "cuda",
            module.reduction_grid_stride,
            "kernel/reduction/reduction_grid_stride.cu",
        ),
        KernelImplementation(
            "integrate",
            "cuda",
            module.reduction_integrate,
            "kernel/reduction/reduction_integrate.cu",
        ),
    ]
