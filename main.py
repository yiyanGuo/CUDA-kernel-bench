from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Callable

from benchmark.common import BenchmarkConfig


BenchmarkFn = Callable[[list[int], BenchmarkConfig], bool]
REPO_ROOT = Path(__file__).resolve().parent


KERNEL_DIR_OPERATOR_ALIASES = {
    "RMSNorm": "rmsnorm",
}


def discover_operator_modules() -> dict[str, str]:
    import benchmark

    operators: dict[str, str] = {}
    for module_info in pkgutil.iter_modules(benchmark.__path__):
        module_name = module_info.name
        if not module_name.startswith("bench_"):
            continue
        operator_name = module_name.removeprefix("bench_")
        operators[operator_name] = f"benchmark.{module_name}"
    return dict(sorted(operators.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Dispatch Python benchmarks for CUDA and Triton kernel backends. "
            "Pass the operator name followed by its dimensions."
        )
    )
    parser.add_argument("operator", nargs="?", help="Benchmark operator name, or 'all'.")
    parser.add_argument(
        "dims",
        nargs="*",
        type=int,
        help=(
            "Optional dimensions for the selected operator. "
            "Examples: vector_add 16777216, transpose 2048 4096."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List auto-registered operators and exit.",
    )
    parser.add_argument(
        "--mode",
        choices=("compare", "single"),
        default="compare",
        help="Run all implementations or a single selected implementation.",
    )
    parser.add_argument(
        "--impl",
        help="Implementation to run in single mode. Accepts name or backend:name.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup run count.")
    parser.add_argument("--repeat", type=int, default=5, help="Timed run count.")
    return parser


def print_available_operators(operator_modules: dict[str, str]) -> None:
    print("Available operators:")
    for name in operator_modules:
        print(f"  {name}")


def load_benchmark(name: str, operator_modules: dict[str, str]) -> BenchmarkFn | None:
    module_name = operator_modules.get(name)
    if module_name is None:
        return None
    module = importlib.import_module(module_name)
    return module.run_benchmark


def infer_operator_from_impl(
    implementation: str,
    operator_modules: dict[str, str],
) -> str | None:
    candidates = list(REPO_ROOT.rglob(implementation))
    candidates = [
        candidate
        for candidate in candidates
        if candidate.is_file() and "build" not in candidate.parts
    ]
    if len(candidates) != 1:
        return None

    try:
        relative = candidates[0].relative_to(REPO_ROOT)
    except ValueError:
        return None

    if relative.parts[0] == "benchmark" and relative.name.startswith("bench_"):
        operator_name = relative.stem.removeprefix("bench_")
    elif relative.parts[0] == "kernel" and len(relative.parts) >= 3:
        kernel_dir = relative.parts[1]
        operator_name = KERNEL_DIR_OPERATOR_ALIASES.get(kernel_dir, kernel_dir)
    else:
        return None

    return operator_name if operator_name in operator_modules else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    operator_modules = discover_operator_modules()

    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be >= 0 and --repeat must be > 0.")

    if args.list:
        print_available_operators(operator_modules)
        return 0

    if args.operator is None:
        if args.mode == "single" and args.impl:
            inferred_operator = infer_operator_from_impl(args.impl, operator_modules)
            if inferred_operator is None:
                parser.error(
                    "operator is required unless --list is used, or --impl uniquely "
                    "identifies a registered kernel/benchmark file."
                )
            args.operator = inferred_operator
        else:
            parser.error("operator is required unless --list is used.")
    elif args.operator not in operator_modules and args.operator != "all":
        inferred_operator = None
        if args.mode == "single" and args.impl:
            inferred_operator = infer_operator_from_impl(args.impl, operator_modules)
        if inferred_operator is not None:
            try:
                inferred_first_dim = int(args.operator)
            except ValueError:
                inferred_first_dim = None
            if inferred_first_dim is not None:
                args.dims = [inferred_first_dim, *args.dims]
                args.operator = inferred_operator

    if args.mode == "single" and not args.impl:
        parser.error("--mode single requires --impl.")
    if args.mode == "compare" and args.impl:
        parser.error("--impl is only valid with --mode single.")

    config = BenchmarkConfig(
        warmup=args.warmup,
        repeat=args.repeat,
        mode=args.mode,
        implementation=args.impl,
    )
    requested_operator = args.operator

    if requested_operator == "all":
        if args.dims:
            parser.error("'all' does not accept custom dimensions.")
        if args.mode == "single":
            parser.error("'all' does not support --mode single.")
        all_passed = True
        for name in operator_modules:
            run_benchmark = load_benchmark(name, operator_modules)
            if run_benchmark is None:
                continue
            print(f"== {name} ==")
            all_passed = run_benchmark([], config) and all_passed
        return 0 if all_passed else 2

    run_benchmark = load_benchmark(requested_operator, operator_modules)
    if run_benchmark is None:
        print(f"Unknown operator: {requested_operator}", file=sys.stderr)
        print_available_operators(operator_modules)
        return 1

    try:
        passed = run_benchmark(args.dims, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
