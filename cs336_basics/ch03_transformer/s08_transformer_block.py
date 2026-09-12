"""3.4–3.5 Transformer Block。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
