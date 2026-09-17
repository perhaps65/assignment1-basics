"""3.3.2 Linear 模块。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor, nn
import torch
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                device=device,
                dtype=dtype
            )
        )
        sigma = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T
