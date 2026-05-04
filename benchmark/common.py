from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from kernel.api import KernelImplementation


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int = 2
    repeat: int = 5
    mode: str = "compare"
    implementation: str | None = None
    casual: bool = False


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment.")


def load_backend_implementations(module_names: Iterable[str]) -> list[KernelImplementation]:
    implementations: list[KernelImplementation] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise
            print(
                f"[skip] {module_name}: missing dependency '{exc.name}'",
                file=sys.stderr,
            )
            continue
        implementations.extend(module.get_implementations())
    return implementations


def implementation_key(implementation: KernelImplementation) -> str:
    return f"{implementation.backend}:{implementation.name}"


def implementation_labels(implementation: KernelImplementation) -> set[str]:
    labels = {
        implementation.name,
        implementation_key(implementation),
    }
    if implementation.source is not None:
        source_path = Path(implementation.source)
        labels.add(implementation.source)
        labels.add(source_path.name)
        labels.add(source_path.stem)
    return labels


def implementation_matches(
    implementation: KernelImplementation,
    selected_implementation: str | None,
) -> bool:
    if selected_implementation is None:
        return True
    return selected_implementation in implementation_labels(implementation)


def filter_implementations(
    implementations: Iterable[KernelImplementation],
    config: BenchmarkConfig,
    *,
    allow_empty: bool = False,
) -> list[KernelImplementation]:
    implementations = list(implementations)
    filtered = [
        implementation
        for implementation in implementations
        if implementation_matches(implementation, config.implementation)
    ]
    if config.implementation is not None and not filtered and not allow_empty:
        available = ", ".join(
            implementation.source or implementation_key(implementation)
            for implementation in implementations
        )
        raise ValueError(
            f"Unknown implementation '{config.implementation}'. "
            f"Available implementations: {available}"
        )
    return filtered


def nearly_equal(
    lhs: float,
    rhs: float,
    abs_tolerance: float = 1e-5,
    rel_tolerance: float = 1e-5,
) -> bool:
    diff = abs(lhs - rhs)
    if diff <= abs_tolerance:
        return True
    scale = max(abs(lhs), abs(rhs))
    return diff <= rel_tolerance * scale


def compare_tensors(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    abs_tolerance: float = 1e-5,
    rel_tolerance: float = 1e-5,
) -> bool:
    lhs_cpu = lhs.detach().cpu()
    rhs_cpu = rhs.detach().cpu()

    if torch.allclose(lhs_cpu, rhs_cpu, atol=abs_tolerance, rtol=rel_tolerance):
        return True

    mismatch_mask = ~torch.isclose(
        lhs_cpu,
        rhs_cpu,
        atol=abs_tolerance,
        rtol=rel_tolerance,
    )
    first_mismatch = mismatch_mask.flatten().nonzero()[0].item()
    lhs_value = lhs_cpu.flatten()[first_mismatch].item()
    rhs_value = rhs_cpu.flatten()[first_mismatch].item()
    print(
        f"Mismatch at index {first_mismatch}: lhs={lhs_value:.6f} rhs={rhs_value:.6f}",
        file=sys.stderr,
    )
    return False


def compare_scalars(
    lhs: float,
    rhs: float,
    abs_tolerance: float = 1e-5,
    rel_tolerance: float = 1e-5,
) -> bool:
    if nearly_equal(lhs, rhs, abs_tolerance, rel_tolerance):
        return True
    print(f"Mismatch: lhs={lhs:.6f} rhs={rhs:.6f}", file=sys.stderr)
    return False


def measure_min_elapsed_ms(
    repeat_count: int,
    launch: Callable[[], None],
    prepare: Callable[[], None] | None = None,
) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    best_ms = float("inf")
    for _ in range(repeat_count):
        torch.cuda.synchronize()
        if prepare is not None:
            prepare()
            torch.cuda.synchronize()
        start_event.record()
        launch()
        stop_event.record()
        stop_event.synchronize()
        best_ms = min(best_ms, start_event.elapsed_time(stop_event))
    return best_ms


def to_giga_throughput(units: float, elapsed_ms: float) -> float:
    return units / (elapsed_ms * 1e6)


def to_bandwidth_gb(num_bytes: float, elapsed_ms: float) -> float:
    return num_bytes / (elapsed_ms * 1e6)


def print_benchmark_line(
    op_name: str,
    implementation: KernelImplementation,
    best_ms: float,
    work_units: float,
    work_unit_name: str,
    num_bytes: float,
    passed: bool,
) -> None:
    print(
        f"[{op_name}/{implementation.backend}:{implementation.name}] "
        f"best={best_ms:.4f} ms, "
        f"{to_giga_throughput(work_units, best_ms):.3f} G{work_unit_name}/s, "
        f"{to_bandwidth_gb(num_bytes, best_ms):.3f} GB/s, "
        f"verify={'PASS' if passed else 'FAIL'}"
    )


def run_implementation(
    op_name: str,
    implementation: KernelImplementation,
    config: BenchmarkConfig,
    launch: Callable[[], None],
    verify: Callable[[], bool],
    work_units: float,
    work_unit_name: str,
    num_bytes: float,
    prepare: Callable[[], None] | None = None,
) -> bool:
    for _ in range(config.warmup):
        if prepare is not None:
            prepare()
        launch()
        torch.cuda.synchronize()

    best_ms = measure_min_elapsed_ms(config.repeat, launch, prepare)

    if prepare is not None:
        prepare()
    launch()
    torch.cuda.synchronize()
    passed = verify()
    print_benchmark_line(
        op_name=op_name,
        implementation=implementation,
        best_ms=best_ms,
        work_units=work_units,
        work_unit_name=work_unit_name,
        num_bytes=num_bytes,
        passed=passed,
    )
    return passed
