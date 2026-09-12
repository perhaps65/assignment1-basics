"""4.5 梯度裁剪。仅接口骨架，待自行实现。"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    raise NotImplementedError
