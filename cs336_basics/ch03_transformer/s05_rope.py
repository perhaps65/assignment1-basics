"""3.4.3 旋转位置编码。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        raise NotImplementedError
