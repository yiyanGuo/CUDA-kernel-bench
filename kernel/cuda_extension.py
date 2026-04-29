from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from torch.utils.cpp_extension import load


REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIR = REPO_ROOT / "include"
BUILD_DIR = REPO_ROOT / "build" / "torch_extensions"


@lru_cache(maxsize=None)
def load_extension(module_name: str, sources: tuple[str, ...]):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    build_directory = BUILD_DIR / module_name
    build_directory.mkdir(parents=True, exist_ok=True)

    absolute_sources = [str(REPO_ROOT / source) for source in sources]
    return load(
        name=module_name,
        sources=absolute_sources,
        extra_include_paths=[str(INCLUDE_DIR)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        build_directory=str(build_directory),
        verbose=False,
    )


def load_extension_from_iterable(module_name: str, sources: Iterable[str]):
    return load_extension(module_name, tuple(sources))
