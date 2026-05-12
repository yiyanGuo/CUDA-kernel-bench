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
SEQ_LEN = 1024
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
ROTARY_DIM = 128
POSITION_OFFSET = 0
ROPE_BASE = 10000.0
WARMUP = 2
REPEAT = 5
VERIFY = True


@lru_cache(maxsize=None)
def load_kernel():
    module_name = "cuda_kernel_bench_rope_naive"
    build_dir = KERNEL_DIR / ".torch_extensions" / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=module_name,
        sources=[str(KERNEL_DIR / "rope_single_binding.cpp"), str(KERNEL_DIR / "rope_naive.cu")],
        extra_include_paths=[str(INCLUDE_DIR), str(KERNEL_DIR)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_dir),
        verbose=False,
    )


def rope_cuda(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> None:
    load_kernel().rope_naive(q, k, cos, sin, position_ids, ROTARY_DIM, POSITION_OFFSET)


def make_tensor(shape: tuple[int, ...], modulo: int, scale: float) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.arange(numel, dtype=torch.float32).reshape(shape).remainder(modulo).sub(modulo // 2).mul(scale)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = make_tensor((BATCH_SIZE, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM), 251, 0.01)
    k = make_tensor((BATCH_SIZE, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM), 197, 0.0125)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.int32).unsqueeze(0).expand(BATCH_SIZE, SEQ_LEN).contiguous()
    pair_dim = ROTARY_DIM // 2
    max_position = int(position_ids.max().item()) + POSITION_OFFSET + 1
    positions = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
    pair_ids = torch.arange(pair_dim, dtype=torch.float32).unsqueeze(0)
    inv_freq = torch.pow(ROPE_BASE, -(2.0 * pair_ids) / float(ROTARY_DIM))
    freqs = positions * inv_freq
    return q.cuda(), k.cuda(), torch.cos(freqs).cuda(), torch.sin(freqs).cuda(), position_ids.cuda()


def rotate_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    out = xf.clone()
    position = position_ids.long() + POSITION_OFFSET
    c = cos[position].float().unsqueeze(2)
    s = sin[position].float().unsqueeze(2)
    even = xf[..., :ROTARY_DIM:2]
    odd = xf[..., 1:ROTARY_DIM:2]
    out[..., :ROTARY_DIM:2] = even * c - odd * s
    out[..., 1:ROTARY_DIM:2] = even * s + odd * c
    return out.to(dtype=x.dtype)


def apply_reference(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> None:
    q.copy_(rotate_reference(q, cos, sin, position_ids))
    k.copy_(rotate_reference(k, cos, sin, position_ids))


def compare_tensors(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5):
        return True
    mismatch = (~torch.isclose(lhs, rhs, atol=1e-5, rtol=1e-5)).flatten().nonzero()[0].item()
    print(f"Mismatch at index {mismatch}: lhs={lhs.flatten()[mismatch].item():.6f} rhs={rhs.flatten()[mismatch].item():.6f}")
    return False


def measure_ms(launch: Callable[[], None], prepare: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(REPEAT):
        prepare()
        torch.cuda.synchronize()
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        best = min(best, start.elapsed_time(stop))
    return best


def run_impl(name: str, backend: str, launch: Callable[[], None], prepare: Callable[[], None], verify: Callable[[], bool], num_bytes: float) -> bool:
    for _ in range(WARMUP):
        prepare()
        launch()
        torch.cuda.synchronize()
    best_ms = measure_ms(launch, prepare)
    passed = None
    if VERIFY:
        prepare()
        launch()
        torch.cuda.synchronize()
        passed = verify()
    status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    rows = float(BATCH_SIZE * SEQ_LEN * (NUM_Q_HEADS + NUM_KV_HEADS))
    flops = rows * float(ROTARY_DIM // 2) * 6.0 / (best_ms * 1e6)
    bandwidth = num_bytes / (best_ms * 1e6)
    print(f"[rope/{backend}:{name}] best={best_ms:.4f} ms, {flops:.3f} GFLOP/s, {bandwidth:.3f} GB/s, verify={status}")
    return True if passed is None else passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    base_q, base_k, cos, sin, position_ids = make_inputs()
    q = torch.empty_like(base_q)
    k = torch.empty_like(base_k)
    ref_q = rotate_reference(base_q, cos, sin, position_ids) if VERIFY else None
    ref_k = rotate_reference(base_k, cos, sin, position_ids) if VERIFY else None

    def prepare() -> None:
        q.copy_(base_q)
        k.copy_(base_k)

    rows = float(BATCH_SIZE * SEQ_LEN * (NUM_Q_HEADS + NUM_KV_HEADS))
    num_bytes = rows * float(ROTARY_DIM // 2) * 6.0 * base_q.element_size()
    implementations = [
        ("naive", "cuda", lambda: rope_cuda(q, k, cos, sin, position_ids)),
        ("torch", "pytorch", lambda: apply_reference(q, k, cos, sin, position_ids)),
    ]

    all_passed = True
    for name, backend, fn in implementations:
        passed = run_impl(
            name,
            backend,
            fn,
            prepare,
            lambda: True if ref_q is None or ref_k is None else compare_tensors(q, ref_q) and compare_tensors(k, ref_k),
            num_bytes,
        )
        all_passed = passed and all_passed
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
