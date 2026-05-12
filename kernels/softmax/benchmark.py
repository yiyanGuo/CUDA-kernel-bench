from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch
from torch.utils.cpp_extension import load


KERNEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = KERNEL_DIR.parents[1]
INCLUDE_DIR = REPO_ROOT / "include"

BATCH_SIZE = 4
NUM_HEADS = 8
QUERY_LEN = 128
KEY_LEN = 1 << 17
CASUAL = False
WARMUP = 2
REPEAT = 5
VERIFY = True

KERNELS = [
    ("naive", "softmax_naive.cu", "softmax_naive"),
    ("2_pass", "softmax_2_pass.cu", "softmax_2_pass"),
]


@lru_cache(maxsize=None)
def load_kernel(name: str, source: str, symbol: str):
    module_name = f"cuda_kernel_bench_softmax_{name}"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / "softmax_single_binding.cpp"), str(KERNEL_DIR / source)],
        extra_include_paths=[str(INCLUDE_DIR), str(KERNEL_DIR)],
        extra_cflags=["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}"],
        extra_cuda_cflags=["-O3", "-std=c++17", f"-DKERNEL_SYMBOL={symbol}"],
        build_directory=str(build_dir),
        verbose=False,
    )


def make_launch(name: str, source: str, symbol: str) -> Callable[[torch.Tensor, torch.Tensor, bool], None]:
    cache = {}

    def launch(x: torch.Tensor, out: torch.Tensor, casual: bool) -> None:
        if "fn" not in cache:
            cache["fn"] = getattr(load_kernel(name, source, symbol), symbol)
        cache["fn"](x, out, casual)

    return launch


def reference(input_tensor: torch.Tensor) -> torch.Tensor:
    logits = input_tensor.float()
    if CASUAL:
        query_positions = torch.arange(QUERY_LEN, device=logits.device).unsqueeze(-1)
        key_positions = torch.arange(KEY_LEN, device=logits.device)
        logits = logits.masked_fill(~(key_positions <= query_positions), -torch.inf)
    return torch.softmax(logits, dim=-1).to(dtype=input_tensor.dtype)


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5):
        return True
    mismatch = (~torch.isclose(lhs, rhs, atol=1e-5, rtol=1e-5)).flatten().nonzero()[0].item()
    print(f"Mismatch at index {mismatch}: lhs={lhs.flatten()[mismatch].item():.6f} rhs={rhs.flatten()[mismatch].item():.6f}")
    return False


def measure_ms(launch: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best


def run_impl(name: str, backend: str, launch: Callable[[], None], verify: Callable[[], bool], num_bytes: float) -> bool:
    for _ in range(WARMUP):
        launch()
        torch.cuda.synchronize()
    best_ms = measure_ms(launch)
    passed = None
    if VERIFY:
        launch()
        torch.cuda.synchronize()
        passed = verify()
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    element_count = float(BATCH_SIZE * NUM_HEADS * QUERY_LEN * KEY_LEN)
    gops = element_count * 5.0 / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[softmax/{backend}:{name}] best={best_ms:.4f} ms, {gops:.3f} GOP/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    element_count = BATCH_SIZE * NUM_HEADS * QUERY_LEN * KEY_LEN
    host_input = torch.arange(element_count, dtype=torch.float32).reshape(BATCH_SIZE, NUM_HEADS, QUERY_LEN, KEY_LEN).remainder(257).sub(128).mul(0.03125)
    device_input = host_input.cuda()
    device_output = torch.empty_like(device_input)
    ref = reference(device_input) if VERIFY else None
    num_bytes = float(element_count) * 2.0 * device_input.element_size()

    implementations = [(name, "cuda", make_launch(name, source, symbol)) for name, source, symbol in KERNELS]
    implementations.append(("torch", "pytorch", lambda x, out, casual: out.copy_(reference(x))))

    all_passed = True
    for name, backend, fn in implementations:
        passed = run_impl(
            name,
            backend,
            lambda fn=fn: fn(device_input, device_output, CASUAL),
            lambda: True if ref is None else compare_tensors(device_output, ref),
            num_bytes,
        )
        all_passed = passed and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
