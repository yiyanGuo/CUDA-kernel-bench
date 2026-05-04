from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_softmax",
        [
            "kernel/softmax/softmax_binding.cpp",
            "kernel/softmax/softmax_naive.cu",
            "kernel/softmax/softmax_2_pass.cu"
        ],
    )
    return [
        KernelImplementation(
            name="naive",
            backend="cuda",
            launch=module.softmax_naive,
            source="kernel/softmax/softmax_naive.cu",
        ),
        KernelImplementation(
            name="2_pass",
            backend="cuda",
            launch=module.softmax_2_pass,
            source="kernel/softmax/softmax_2_pass.cu",
        )
    ]
