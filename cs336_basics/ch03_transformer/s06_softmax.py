"""3.4.4 Softmax。仅接口骨架，待自行实现。"""

from __future__ import annotations

from torch import Tensor
import torch


def softmax(x: Tensor, dim: int) -> Tensor:
    max_value = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_value)
    sum_value = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_value

