from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable

from benchmark.common import BenchmarkConfig


BenchmarkFn = Callable[[list[int], BenchmarkConfig], bool]


OPERATOR_MODULES: dict[str, str] = {
    "reduction": "benchmark.bench_reduction",
    "scan": "benchmark.bench_scan",
    "transpose": "benchmark.bench_transpose",
    "vector_add": "benchmark.bench_vector_add",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Dispatch Python benchmarks for CUDA and Triton kernel backends. "
            "Pass the operator name followed by its dimensions."
        )
    )
    parser.add_argument("operator", help="Benchmark operator name, or 'all'.")
    parser.add_argument(
        "dims",
        nargs="*",
        type=int,
        help=(
            "Optional dimensions for the selected operator. "
            "Examples: vector_add 16777216, transpose 2048 4096."
        ),
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup run count.")
    parser.add_argument("--repeat", type=int, default=5, help="Timed run count.")
    return parser


def print_available_operators() -> None:
    print("Available operators:")
    for name in OPERATOR_MODULES:
        print(f"  {name}")


def load_benchmark(name: str) -> BenchmarkFn | None:
    module_name = OPERATOR_MODULES.get(name)
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    return module.run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0.")

    config = BenchmarkConfig(warmup=args.warmup, repeat=args.repeat)
    requested_operator = args.operator

    if requested_operator == "all":
        if args.dims:
            parser.error("'all' does not accept custom dimensions.")
        all_passed = True
        for name in OPERATOR_MODULES:
            run_benchmark = load_benchmark(name)
            if run_benchmark is None:
                continue
            print(f"== {name} ==")
            all_passed = run_benchmark([], config) and all_passed
        return 0 if all_passed else 2

    run_benchmark = load_benchmark(requested_operator)
    if run_benchmark is None:
        print(f"Unknown operator: {requested_operator}", file=sys.stderr)
        print_available_operators()
        return 1

    try:
        passed = run_benchmark(args.dims, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
