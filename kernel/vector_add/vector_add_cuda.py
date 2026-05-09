from __future__ import annotations

from kernel.api import KernelImplementation
from kernel.cuda_extension import load_extension_from_iterable


def get_module():
    return load_extension_from_iterable(
        "cuda_kernel_bench_vector_add",
        ["kernel/vector_add/vector_add.cu"],
    )


def get_implementations() -> list[KernelImplementation]:
    module = get_module()
    return [
        KernelImplementation(
            name="dispatch",
            backend="cuda",
            launch=module.vector_add,
            source="kernel/vector_add/vector_add.cu",
        ),
        KernelImplementation(
            name="f32x4",
            backend="cuda",
            launch=module.vector_add_f32x4,
            source="kernel/vector_add/vector_add.cu",
        ),
        KernelImplementation(
            name="f16x2",
            backend="cuda",
            launch=module.vector_add_f16x2,
            source="kernel/vector_add/vector_add.cu",
        ),
    ]
