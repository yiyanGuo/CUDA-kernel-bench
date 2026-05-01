from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_rmsnorm_half",
        [
            "kernel/RMSNorm/rms_half_binding.cpp",
            "kernel/RMSNorm/rms_half.cu",
            "kernel/RMSNorm/rms_half2.cu",
        ],
    )
    return [
        KernelImplementation(
            name="half",
            backend="cuda",
            launch=module.rms_half,
            source="kernel/RMSNorm/rms_half.cu",
        ),
        KernelImplementation(
            name="half2",
            backend="cuda",
            launch=module.rms_half2,
            source="kernel/RMSNorm/rms_half2.cu",
        ),
    ]
