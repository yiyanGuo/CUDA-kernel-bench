from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_implementations() -> list[KernelImplementation]:
    module = load_extension_from_iterable(
        "cuda_kernel_bench_scan",
        [
            "kernel/scan/scan_binding.cpp",
            "kernel/scan/scan_naive.cu",
            "kernel/scan/scan_one_block.cu",
            "kernel/scan/scan_multi_block.cu",
            "kernel/scan/scan_warp.cu",
            "kernel/scan/thrust_exclusive_scan.cu",
            "kernel/scan/scan_memory_buffer.cu",
        ],
    )
    return [
        KernelImplementation(
            "naive", "cuda", module.scan_naive, "kernel/scan/scan_naive.cu"
        ),
        KernelImplementation(
            "one_block",
            "cuda",
            module.scan_one_block,
            "kernel/scan/scan_one_block.cu",
        ),
        KernelImplementation(
            "multi_block",
            "cuda",
            module.scan_multi_block,
            "kernel/scan/scan_multi_block.cu",
        ),
        KernelImplementation(
            "warp", "cuda", module.scan_warp, "kernel/scan/scan_warp.cu"
        ),
        KernelImplementation(
            "thrust",
            "cuda",
            module.scan_thrust_exclusive,
            "kernel/scan/thrust_exclusive_scan.cu",
        ),
        KernelImplementation(
            "memory_buffer",
            "cuda",
            module.scan_memory_buffer,
            "kernel/scan/scan_memory_buffer.cu",
        ),
    ]
