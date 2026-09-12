"""3.4.1 RMSNorm。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
