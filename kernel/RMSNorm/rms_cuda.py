from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_rmsnorm",
        [
            "kernel/RMSNorm/rms_binding.cpp",
            "kernel/RMSNorm/rms_naive.cu",
            "kernel/RMSNorm/rms_naive_v2.cu",
            "kernel/RMSNorm/rms_shared_memory.cu",
        ],
    )
    return [
        KernelImplementation(
            name="naive",
            backend="cuda",
            launch=module.rms_naive,
            source="kernel/RMSNorm/rms_naive.cu",
        ),
        KernelImplementation(
            name="naive_v2",
            backend="cuda",
            launch=module.rms_naive_v2,
            source="kernel/RMSNorm/rms_naive_v2.cu",
        ),
        KernelImplementation(
            name="shared_memory",
            backend="cuda",
            launch=module.rms_shared_memory,
            source="kernel/RMSNorm/rms_shared_memory.cu",
        ),
    ]
