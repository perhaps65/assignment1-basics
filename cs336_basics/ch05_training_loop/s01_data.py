"""5.1 数据加载。仅接口骨架，待自行实现。"""

from __future__ import annotations

import numpy.typing as npt
from torch import Tensor


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError
