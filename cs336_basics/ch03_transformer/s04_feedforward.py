"""3.4.2 SiLU 与 SwiGLU。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


def silu(x: Tensor) -> Tensor:
    raise NotImplementedError

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
