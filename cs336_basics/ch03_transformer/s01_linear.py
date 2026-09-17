"""3.3.2 Linear 模块。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
