"""3.4.4–3.4.5 缩放点积注意力与多头自注意力。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None,
) -> Tensor:
    raise NotImplementedError

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        raise NotImplementedError
