from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


LaunchFn = Callable[..., None]


@dataclass(frozen=True)
class KernelImplementation:
    name: str
    backend: str
    launch: LaunchFn
