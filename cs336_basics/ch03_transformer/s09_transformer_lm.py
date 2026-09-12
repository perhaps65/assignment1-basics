"""3.5 完整 Transformer LM。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError
