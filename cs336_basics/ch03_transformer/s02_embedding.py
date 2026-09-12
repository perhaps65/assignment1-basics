"""3.3.3 Embedding 模块。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        raise NotImplementedError

    def forward(self, token_ids: Tensor) -> Tensor:
        raise NotImplementedError
