from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_module():
    return load_extension_from_iterable(
        "cuda_kernel_bench_vector_add_cutlass",
        ["kernel/vector_add/vector_add_cutlass.cu"],
    )


def get_implementations() -> list[KernelImplementation]:
    module = get_module()
    return [
        KernelImplementation(
            name="cute_x4",
            backend="cutlass",
            launch=module.vector_add_cute_x4,
            source="kernel/vector_add/vector_add_cutlass.cu",
        ),
    ]
