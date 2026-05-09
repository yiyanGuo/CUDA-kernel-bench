from __future__ import annotations

from benchmark.common import BenchmarkConfig
from kernel.hgemm.benchmark import run_benchmark as _run_hgemm_benchmark


def run_benchmark(dims: list[int], config: BenchmarkConfig) -> bool:
    return _run_hgemm_benchmark(dims, config)
